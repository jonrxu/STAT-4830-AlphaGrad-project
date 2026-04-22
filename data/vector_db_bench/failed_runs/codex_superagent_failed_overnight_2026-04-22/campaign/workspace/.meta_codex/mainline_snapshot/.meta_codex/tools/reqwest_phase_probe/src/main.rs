use clap::{Parser, ValueEnum};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    server_url: String,

    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    #[arg(long, default_value_t = 100)]
    warmup: usize,

    #[arg(long, default_value_t = 1000)]
    max_queries: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value = "/opt/vdb-data/query_vectors.json")]
    query_vectors: PathBuf,

    #[arg(long)]
    base_vectors: Option<PathBuf>,

    #[arg(long, default_value_t = 1000)]
    insert_batch_size: usize,

    #[arg(long, default_value_t = 0)]
    preload_max_vectors: usize,

    #[arg(long, value_enum, default_value_t = CompletionMode::Json)]
    completion_mode: CompletionMode,

    #[arg(long, value_enum, default_value_t = WarmupCompletionMode::MatchTimed)]
    warmup_completion_mode: WarmupCompletionMode,
}

#[derive(Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompletionMode {
    Json,
    Bytes,
}

#[derive(Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum WarmupCompletionMode {
    MatchTimed,
    None,
    Json,
    Bytes,
}

#[derive(Clone, Deserialize)]
struct IndexedVector {
    #[serde(default)]
    id: u64,
    vector: Vec<f32>,
}

#[derive(Clone, Serialize)]
struct SearchRequest {
    vector: Vec<f32>,
    top_k: u32,
}

#[derive(Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Deserialize)]
struct SearchResult {
    id: u64,
    distance: f64,
}

#[derive(Serialize)]
struct InsertItem {
    id: u64,
    vector: Vec<f32>,
}

#[derive(Serialize)]
struct BulkInsertRequest {
    vectors: Vec<InsertItem>,
}

#[derive(Deserialize)]
struct BulkInsertResponse {
    inserted: usize,
}

#[derive(Serialize)]
struct QueryTiming {
    ok: bool,
    send_ms: f64,
    body_ms: f64,
    total_ms: f64,
    send_completed_ms: f64,
    response_len: usize,
    results_len: usize,
}

#[derive(Serialize)]
struct MetricSummary {
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[derive(Serialize)]
struct PreloadSummary {
    vectors_loaded: usize,
    vectors_inserted: usize,
    batch_size: usize,
    duration_secs: f64,
}

#[derive(Serialize)]
struct ProbeOutput {
    type_name: &'static str,
    completion_mode: CompletionMode,
    warmup_completion_mode: WarmupCompletionMode,
    preload: Option<PreloadSummary>,
    total_queries: usize,
    warmup: usize,
    concurrency: usize,
    send_phase_duration_secs: f64,
    completion_duration_secs: f64,
    send_phase_qps: f64,
    completion_qps: f64,
    ok_queries: usize,
    failed_queries: usize,
    send_latency: MetricSummary,
    body_latency: MetricSummary,
    total_latency: MetricSummary,
    response_len_bytes: MetricSummary,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let concurrency = cli.concurrency.max(1);
    let queries = load_queries(&cli.query_vectors)?;
    let queries = limit_queries(queries, cli.max_queries);
    let client = reqwest::Client::new();

    let preload = if let Some(base_vectors) = cli.base_vectors.as_ref() {
        Some(
            preload_base_vectors(
                &client,
                &cli.server_url,
                base_vectors,
                cli.preload_max_vectors,
                cli.insert_batch_size.max(1),
            )
            .await?,
        )
    } else {
        None
    };

    run_warmup(
        &client,
        &cli.server_url,
        &queries,
        cli.warmup,
        cli.completion_mode,
        cli.warmup_completion_mode,
    )
    .await;

    let started = Instant::now();
    let timings = run_queries(
        &client,
        &cli.server_url,
        &queries,
        concurrency,
        cli.seed,
        cli.completion_mode,
        started,
    )
    .await;
    let completion_duration_secs = started.elapsed().as_secs_f64();
    let send_phase_duration_secs = timings
        .iter()
        .map(|timing| timing.send_completed_ms)
        .fold(0.0f64, f64::max)
        / 1000.0;

    let ok_queries = timings.iter().filter(|timing| timing.ok).count();
    let failed_queries = timings.len().saturating_sub(ok_queries);

    let output = ProbeOutput {
        type_name: "ReqwestPhaseProbe",
        completion_mode: cli.completion_mode,
        warmup_completion_mode: cli.warmup_completion_mode,
        preload,
        total_queries: timings.len(),
        warmup: cli.warmup.min(queries.len()),
        concurrency,
        send_phase_duration_secs,
        completion_duration_secs,
        send_phase_qps: qps(timings.len(), send_phase_duration_secs),
        completion_qps: qps(timings.len(), completion_duration_secs),
        ok_queries,
        failed_queries,
        send_latency: summarize_metric(timings.iter().map(|timing| timing.send_ms).collect()),
        body_latency: summarize_metric(timings.iter().map(|timing| timing.body_ms).collect()),
        total_latency: summarize_metric(timings.iter().map(|timing| timing.total_ms).collect()),
        response_len_bytes: summarize_metric(
            timings
                .iter()
                .map(|timing| timing.response_len as f64)
                .collect(),
        ),
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn load_queries(path: &PathBuf) -> Result<Vec<IndexedVector>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let queries = serde_json::from_slice(&bytes)?;
    Ok(queries)
}

fn load_vectors(path: &PathBuf) -> Result<Vec<IndexedVector>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let vectors = serde_json::from_slice(&bytes)?;
    Ok(vectors)
}

fn limit_queries(queries: Vec<IndexedVector>, max_queries: usize) -> Vec<IndexedVector> {
    if max_queries == 0 || max_queries >= queries.len() {
        return queries;
    }
    queries.into_iter().take(max_queries).collect()
}

fn limit_vectors(vectors: Vec<IndexedVector>, max_vectors: usize) -> Vec<IndexedVector> {
    if max_vectors == 0 || max_vectors >= vectors.len() {
        return vectors;
    }
    vectors.into_iter().take(max_vectors).collect()
}

async fn preload_base_vectors(
    client: &reqwest::Client,
    server_url: &str,
    base_vectors_path: &PathBuf,
    preload_max_vectors: usize,
    insert_batch_size: usize,
) -> Result<PreloadSummary, Box<dyn std::error::Error>> {
    let vectors = limit_vectors(load_vectors(base_vectors_path)?, preload_max_vectors);
    let started = Instant::now();
    let inserted = bulk_insert_vectors(client, server_url, &vectors, insert_batch_size).await?;

    Ok(PreloadSummary {
        vectors_loaded: vectors.len(),
        vectors_inserted: inserted,
        batch_size: insert_batch_size,
        duration_secs: started.elapsed().as_secs_f64(),
    })
}

async fn bulk_insert_vectors(
    client: &reqwest::Client,
    server_url: &str,
    vectors: &[IndexedVector],
    batch_size: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let url = format!("{}/bulk_insert", server_url.trim_end_matches('/'));
    let mut total_inserted = 0usize;

    for chunk in vectors.chunks(batch_size.max(1)) {
        let request = BulkInsertRequest {
            vectors: chunk
                .iter()
                .map(|vector| InsertItem {
                    id: vector.id,
                    vector: vector.vector.clone(),
                })
                .collect(),
        };

        let response = client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<BulkInsertResponse>()
            .await?;
        total_inserted += response.inserted;
    }

    Ok(total_inserted)
}

async fn run_warmup(
    client: &reqwest::Client,
    server_url: &str,
    queries: &[IndexedVector],
    warmup_count: usize,
    completion_mode: CompletionMode,
    warmup_completion_mode: WarmupCompletionMode,
) {
    let url = format!("{}/search", server_url.trim_end_matches('/'));
    let count = warmup_count.min(queries.len());

    for query in queries.iter().take(count) {
        let request = SearchRequest {
            vector: query.vector.clone(),
            top_k: 10,
        };
        if let Ok(response) = client.post(&url).json(&request).send().await {
            match warmup_completion_mode {
                WarmupCompletionMode::MatchTimed => {
                    let _ = complete_response(response, completion_mode).await;
                }
                WarmupCompletionMode::None => {}
                WarmupCompletionMode::Json => {
                    let _ = complete_response(response, CompletionMode::Json).await;
                }
                WarmupCompletionMode::Bytes => {
                    let _ = complete_response(response, CompletionMode::Bytes).await;
                }
            }
        }
    }
}

async fn run_queries(
    client: &reqwest::Client,
    server_url: &str,
    queries: &[IndexedVector],
    concurrency: usize,
    seed: u64,
    completion_mode: CompletionMode,
    started: Instant,
) -> Vec<QueryTiming> {
    let url = Arc::new(format!("{}/search", server_url.trim_end_matches('/')));
    let mut indexed: Vec<&IndexedVector> = queries.iter().collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indexed.shuffle(&mut rng);

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::with_capacity(indexed.len());

    for query in indexed {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let url = Arc::clone(&url);
        let vector = query.vector.clone();

        handles.push(tokio::spawn(async move {
            let request = SearchRequest { vector, top_k: 10 };
            let query_started = Instant::now();
            let response = client.post(url.as_str()).json(&request).send().await;
            let send_ms = query_started.elapsed().as_secs_f64() * 1000.0;
            let send_completed_ms = started.elapsed().as_secs_f64() * 1000.0;

            drop(permit);

            let body_started = Instant::now();
            let (ok, response_len, results_len) = match response {
                Ok(response) => complete_response(response, completion_mode).await,
                Err(_) => (false, 0, 0),
            };
            let body_ms = body_started.elapsed().as_secs_f64() * 1000.0;
            let total_ms = query_started.elapsed().as_secs_f64() * 1000.0;

            QueryTiming {
                ok,
                send_ms,
                body_ms,
                total_ms,
                send_completed_ms,
                response_len,
                results_len,
            }
        }));
    }

    let mut timings = Vec::with_capacity(handles.len());
    for handle in handles {
        if let Ok(timing) = handle.await {
            timings.push(timing);
        }
    }
    timings
}

async fn complete_response(
    response: reqwest::Response,
    completion_mode: CompletionMode,
) -> (bool, usize, usize) {
    let expected_len = response.content_length().unwrap_or(0) as usize;

    match completion_mode {
        CompletionMode::Json => match response.json::<SearchResponse>().await {
            Ok(parsed) => {
                let _distance_checksum: f64 = parsed.results.iter().map(|result| result.distance).sum();
                let _id_checksum: u64 = parsed.results.iter().map(|result| result.id).sum();
                (true, expected_len, parsed.results.len())
            }
            Err(_) => (false, expected_len, 0),
        },
        CompletionMode::Bytes => match response.bytes().await {
            Ok(bytes) => (true, bytes.len(), 0),
            Err(_) => (false, 0, 0),
        },
    }
}

fn summarize_metric(values: Vec<f64>) -> MetricSummary {
    if values.is_empty() {
        return MetricSummary {
            avg_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            max_ms: 0.0,
        };
    }

    let avg_ms = values.iter().sum::<f64>() / values.len() as f64;
    let max_ms = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = values;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    MetricSummary {
        avg_ms,
        p50_ms: percentile(&sorted, 50.0),
        p95_ms: percentile(&sorted, 95.0),
        p99_ms: percentile(&sorted, 99.0),
        max_ms,
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let rank = (p / 100.0) * (values.len().saturating_sub(1)) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return values[lower];
    }

    let frac = rank - lower as f64;
    values[lower] * (1.0 - frac) + values[upper] * frac
}

fn qps(total_queries: usize, duration_secs: f64) -> f64 {
    if duration_secs <= 0.0 {
        0.0
    } else {
        total_queries as f64 / duration_secs
    }
}
