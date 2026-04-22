use serde::Deserialize;
use serde::Serialize;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const DEFAULT_DATA_DIR: &str = "/opt/vdb-data";

#[derive(Deserialize)]
struct QueryVector {
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct GroundTruthEntry {
    query_id: usize,
    neighbors: Vec<u64>,
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    vector: &'a [f32],
    top_k: u32,
}

#[derive(Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Deserialize)]
struct SearchResult {
    id: u64,
}

struct Args {
    server_url: String,
    data_dir: PathBuf,
    ground_truth: Option<PathBuf>,
    queries: usize,
    warmup: usize,
    concurrency: usize,
}

struct ReplaySummary {
    duration_secs: f64,
    matched: u64,
    expected: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let bodies = load_bodies(&args.data_dir)?;
    let ground_truth = if let Some(path) = args.ground_truth.as_ref() {
        Some(load_ground_truth(path)?)
    } else {
        None
    };
    let warmup_count = args.warmup.min(bodies.len());
    let measured_count = args.queries.min(bodies.len().saturating_sub(warmup_count));

    let warmup = &bodies[..warmup_count];
    let measured = &bodies[warmup_count..warmup_count + measured_count];
    let measured_ground_truth = ground_truth
        .as_ref()
        .map(|entries| &entries[warmup_count..warmup_count + measured_count]);

    if !warmup.is_empty() {
        let _ = replay_requests(&args.server_url, warmup, args.concurrency)?;
    }

    let summary = replay_requests_with_recall(
        &args.server_url,
        measured,
        measured_ground_truth,
        args.concurrency,
    )?;
    let qps = if summary.duration_secs > 0.0 {
        measured_count as f64 / summary.duration_secs
    } else {
        0.0
    };
    let recall = if summary.expected > 0 {
        summary.matched as f64 / summary.expected as f64
    } else {
        0.0
    };

    let payload = serde_json::json!({
        "type": "ReplaySearchBodiesFast",
        "server_url": args.server_url,
        "data_dir": args.data_dir,
        "ground_truth": args.ground_truth,
        "warmup": warmup_count,
        "total_queries": measured_count,
        "concurrency": args.concurrency,
        "duration_secs": summary.duration_secs,
        "qps": qps,
        "matched_neighbors": summary.matched,
        "expected_neighbors": summary.expected,
        "recall": recall,
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut server_url = String::from("http://127.0.0.1:8080");
    let mut data_dir = PathBuf::from(env::var("VDB_DATA_DIR").unwrap_or_else(|_| DEFAULT_DATA_DIR.to_string()));
    let mut ground_truth = None;
    let mut queries = 1000usize;
    let mut warmup = 100usize;
    let mut concurrency = 4usize;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        let value = match arg.as_str() {
            "--server-url" | "--data-dir" | "--ground-truth" | "--queries" | "--warmup" | "--concurrency" => {
                iter.next().ok_or_else(|| format!("missing value for {arg}"))?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        };

        match arg.as_str() {
            "--server-url" => server_url = value,
            "--data-dir" => data_dir = PathBuf::from(value),
            "--ground-truth" => ground_truth = Some(PathBuf::from(value)),
            "--queries" => queries = value.parse()?,
            "--warmup" => warmup = value.parse()?,
            "--concurrency" => concurrency = value.parse()?,
            _ => unreachable!(),
        }
    }

    Ok(Args {
        server_url,
        data_dir,
        ground_truth,
        queries,
        warmup,
        concurrency,
    })
}

fn print_help() {
    eprintln!("Usage: replay_search_bodies_fast [--server-url URL] [--data-dir DIR] [--ground-truth PATH] [--queries N] [--warmup N] [--concurrency N]");
}

fn load_bodies(data_dir: &Path) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let query_path = data_dir.join("query_vectors.json");
    let query_bytes = fs::read(&query_path)?;
    let queries: Vec<QueryVector> = serde_json::from_slice(&query_bytes)?;
    let mut bodies = Vec::with_capacity(queries.len());

    for query in &queries {
        let body = serde_json::to_vec(&SearchRequest {
            vector: &query.vector,
            top_k: 10,
        })?;
        bodies.push(body);
    }

    Ok(bodies)
}

fn load_ground_truth(path: &Path) -> Result<Vec<Vec<u64>>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let entries: Vec<GroundTruthEntry> = serde_json::from_slice(&bytes)?;
    let mut ground_truth = vec![Vec::new(); entries.len()];

    for entry in entries {
        if entry.query_id < ground_truth.len() {
            ground_truth[entry.query_id] = entry.neighbors.into_iter().take(10).collect();
        }
    }

    Ok(ground_truth)
}

fn replay_requests(
    server_url: &str,
    bodies: &[Vec<u8>],
    concurrency: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(replay_requests_with_recall(server_url, bodies, None, concurrency)?.duration_secs)
}

fn replay_requests_with_recall(
    server_url: &str,
    bodies: &[Vec<u8>],
    ground_truth: Option<&[Vec<u64>]>,
    concurrency: usize,
) -> Result<ReplaySummary, Box<dyn std::error::Error>> {
    let (host, port, host_header) = parse_server_url(server_url)?;
    let requests: Arc<Vec<Vec<u8>>> = Arc::new(
        bodies
            .iter()
            .map(|body| build_http_request(&host_header, body))
            .collect(),
    );
    let ground_truth: Option<Arc<Vec<Vec<u64>>>> = ground_truth.map(|entries| Arc::new(entries.to_vec()));

    let worker_count = concurrency.max(1);
    let mut partitions = vec![Vec::new(); worker_count];
    for (idx, request_idx) in (0..requests.len()).enumerate() {
        partitions[idx % worker_count].push(request_idx);
    }

    let start = Instant::now();
    let mut handles = Vec::with_capacity(worker_count);
    for request_indexes in partitions {
        let requests = Arc::clone(&requests);
        let ground_truth = ground_truth.clone();
        let host = host.clone();
        handles.push(thread::spawn(move || {
            run_worker(
                &host,
                port,
                &requests,
                ground_truth.as_deref().map(|entries| entries.as_slice()),
                &request_indexes,
            )
        }));
    }

    let mut matched = 0u64;
    let mut expected = 0u64;
    for handle in handles {
        let worker = handle.join().map_err(|_| "worker thread panicked")??;
        matched += worker.matched;
        expected += worker.expected;
    }

    Ok(ReplaySummary {
        duration_secs: start.elapsed().as_secs_f64(),
        matched,
        expected,
    })
}

fn parse_server_url(server_url: &str) -> Result<(String, u16, String), Box<dyn std::error::Error>> {
    let rest = server_url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only plain http:// is supported, got: {server_url}"))?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let mut parts = authority.splitn(2, ':');
    let host = parts.next().unwrap_or("127.0.0.1").to_string();
    let port = parts
        .next()
        .map(str::parse)
        .transpose()?
        .unwrap_or(80u16);
    Ok((host, port, authority.to_string()))
}

fn build_http_request(host_header: &str, body: &[u8]) -> Vec<u8> {
    let headers = format!(
        "POST /search HTTP/1.1\r\ncontent-type: application/json\r\naccept: */*\r\nhost: {host_header}\r\ncontent-length: {}\r\n\r\n",
        body.len()
    );

    let mut request = Vec::with_capacity(headers.len() + body.len());
    request.extend_from_slice(headers.as_bytes());
    request.extend_from_slice(body);
    request
}

fn run_worker(
    host: &str,
    port: u16,
    requests: &[Vec<u8>],
    ground_truth: Option<&[Vec<u64>]>,
    request_indexes: &[usize],
) -> io::Result<ReplaySummary> {
    let mut stream = TcpStream::connect((host, port))?;
    stream.set_nodelay(true)?;
    let mut buffer = Vec::with_capacity(64 * 1024);
    let mut matched = 0u64;
    let mut expected = 0u64;

    for &request_idx in request_indexes {
        stream.write_all(&requests[request_idx])?;
        let body = read_response(&mut stream, &mut buffer, ground_truth.is_some())?;
        if let (Some(entries), Some(body)) = (ground_truth, body) {
            let response: SearchResponse = serde_json::from_slice(&body).map_err(invalid_json_error)?;
            let truth = &entries[request_idx];
            expected += truth.len() as u64;
            for result in response.results {
                if truth.iter().any(|&expected_id| expected_id == result.id) {
                    matched += 1;
                }
            }
        }
    }

    Ok(ReplaySummary {
        duration_secs: 0.0,
        matched,
        expected,
    })
}

fn read_response(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    capture_body: bool,
) -> io::Result<Option<Vec<u8>>> {
    loop {
        if let Some(header_end) = find_header_end(buffer) {
            let body_start = header_end + 4;
            let content_length =
                parse_content_length(&buffer[..header_end]).ok_or_else(bad_response_error)?;
            let body_end = body_start
                .checked_add(content_length)
                .ok_or_else(bad_response_error)?;

            if buffer.len() >= body_end {
                let body = if capture_body {
                    Some(buffer[body_start..body_end].to_vec())
                } else {
                    None
                };
                consume_buffer(buffer, body_end);
                return Ok(body);
            }
        }

        let mut chunk = [0u8; 64 * 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed while waiting for response",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(header_block: &[u8]) -> Option<usize> {
    for line in header_block.split(|&byte| byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(colon) = line.iter().position(|&byte| byte == b':') else {
            continue;
        };
        let (name, value) = line.split_at(colon);
        if name.eq_ignore_ascii_case(b"content-length") {
            return std::str::from_utf8(value)
                .ok()?
                .trim_start_matches(':')
                .trim()
                .parse()
                .ok();
        }
    }
    None
}

fn consume_buffer(buffer: &mut Vec<u8>, consumed: usize) {
    if consumed >= buffer.len() {
        buffer.clear();
        return;
    }

    buffer.copy_within(consumed.., 0);
    buffer.truncate(buffer.len() - consumed);
}

fn bad_response_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid HTTP response")
}

fn invalid_json_error(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
