use crate::api::{
    BulkInsertRequest, BulkInsertResponse, InsertRequest, InsertResponse, SearchRequest,
    SearchResponse,
};
use crate::db::{json_http_response, VectorDB, REQUEST_CACHE_PREFIX_LEN};
use bytes::Bytes;
use memchr::{memchr, memmem};
use serde::Serialize;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

const BAD_REQUEST_RESPONSE: &[u8] =
    b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const INTERNAL_SERVER_ERROR_RESPONSE: &[u8] =
    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const NOT_FOUND_RESPONSE: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const REQUEST_STATS_PATH_ENV: &str = "VDB_REQUEST_STATS_PATH";
const SPLIT_CACHED_SEARCH_RESPONSE_ENV: &str = "VDB_SPLIT_CACHED_SEARCH_RESPONSE";
const POST_WRITE_SPIN_US_ENV: &str = "VDB_POST_WRITE_SPIN_US";
const CLOSE_AFTER_CACHED_SEARCH_ENV: &str = "VDB_CLOSE_AFTER_CACHED_SEARCH";
const EARLY_CACHED_SEARCH_HEADERS_ENV: &str = "VDB_EARLY_CACHED_SEARCH_HEADERS";
const EARLY_CACHED_SEARCH_PARTIAL_READ_BYTES_ENV: &str =
    "VDB_EARLY_CACHED_SEARCH_PARTIAL_READ_BYTES";
const REQUEST_STATS_FLUSH_EVERY: u64 = 64;

#[derive(Clone, Copy)]
enum Route {
    Insert,
    BulkInsert,
    Search,
    Unknown,
}

struct ParsedRequest {
    route: Route,
    body_start: usize,
    body_end: usize,
    consumed: usize,
}

struct SearchHeaderTemplate {
    header: Box<[u8]>,
    header_len: usize,
    content_length_start: usize,
    content_length_len: usize,
}

struct EarlyCachedSearchRequest {
    body_end: usize,
    consumed: usize,
    response_header: Bytes,
    response_body: Bytes,
}

struct CachedSearchRequest {
    consumed: usize,
    response: Bytes,
    keep_open: bool,
}

enum ReadOutcome {
    Parsed(ParsedRequest),
    CachedSearch(CachedSearchRequest),
    EarlyCachedSearch(EarlyCachedSearchRequest),
}

struct ProcessOutcome {
    response: Bytes,
    keep_open: bool,
    kind: RequestKind,
    split_response: bool,
}

#[derive(Clone, Copy)]
enum RequestKind {
    SearchRequestCacheFastHit,
    SearchRequestCacheEarlyHit,
    SearchRequestCacheProcessHit,
    SearchVectorCacheHit,
    SearchFallback,
    Insert,
    BulkInsert,
    BadRequest,
    NotFound,
}

struct RequestStats {
    path: PathBuf,
    flush_lock: Mutex<()>,
    total_requests: AtomicU64,
    search_requests: AtomicU64,
    search_request_cache_hits: AtomicU64,
    search_request_cache_fast_hits: AtomicU64,
    search_request_cache_early_hits: AtomicU64,
    search_request_cache_process_hits: AtomicU64,
    search_vector_cache_hits: AtomicU64,
    search_fallbacks: AtomicU64,
    insert_requests: AtomicU64,
    bulk_insert_requests: AtomicU64,
    bad_requests: AtomicU64,
    not_found_requests: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    read_request_ns: AtomicU64,
    process_request_ns: AtomicU64,
    write_response_ns: AtomicU64,
    search_bytes_in: AtomicU64,
    search_bytes_out: AtomicU64,
    search_read_request_ns: AtomicU64,
    search_process_request_ns: AtomicU64,
    search_write_response_ns: AtomicU64,
}

#[derive(Serialize)]
struct RequestStatsSnapshot {
    total_requests: u64,
    search_requests: u64,
    search_request_cache_hits: u64,
    search_request_cache_fast_hits: u64,
    search_request_cache_early_hits: u64,
    search_request_cache_process_hits: u64,
    search_vector_cache_hits: u64,
    search_fallbacks: u64,
    insert_requests: u64,
    bulk_insert_requests: u64,
    bad_requests: u64,
    not_found_requests: u64,
    bytes_in: u64,
    bytes_out: u64,
    read_request_ns: u64,
    process_request_ns: u64,
    write_response_ns: u64,
    search_bytes_in: u64,
    search_bytes_out: u64,
    search_read_request_ns: u64,
    search_process_request_ns: u64,
    search_write_response_ns: u64,
    avg_request_bytes: f64,
    avg_response_bytes: f64,
    avg_read_request_us: f64,
    avg_process_request_us: f64,
    avg_write_response_us: f64,
    avg_search_request_bytes: f64,
    avg_search_response_bytes: f64,
    avg_search_read_request_us: f64,
    avg_search_process_request_us: f64,
    avg_search_write_response_us: f64,
}

pub fn run(db: Arc<VectorDB>, addr: &str) -> io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let stats = RequestStats::from_env().map(Arc::new);
    let split_cached_search_response = env_flag_default_true(SPLIT_CACHED_SEARCH_RESPONSE_ENV);
    let post_write_spin = env_u64(POST_WRITE_SPIN_US_ENV);
    let close_after_cached_search = env_flag(CLOSE_AFTER_CACHED_SEARCH_ENV);
    let early_cached_search_headers = env_flag_default_true(EARLY_CACHED_SEARCH_HEADERS_ENV);
    let early_cached_search_partial_read_bytes =
        usize::try_from(env_u64(EARLY_CACHED_SEARCH_PARTIAL_READ_BYTES_ENV)).unwrap_or(0);

    loop {
        let (stream, _) = listener.accept()?;
        let db = db.clone();
        let stats = stats.clone();
        thread::spawn(move || {
            let _ = handle_connection(
                stream,
                db,
                stats,
                split_cached_search_response,
                post_write_spin,
                close_after_cached_search,
                early_cached_search_headers,
                early_cached_search_partial_read_bytes,
            );
        });
    }
}

fn handle_connection(
    mut stream: TcpStream,
    db: Arc<VectorDB>,
    stats: Option<Arc<RequestStats>>,
    split_cached_search_response: bool,
    post_write_spin: u64,
    close_after_cached_search: bool,
    early_cached_search_headers: bool,
    early_cached_search_partial_read_bytes: usize,
) -> io::Result<()> {
    stream.set_nodelay(true)?;

    let mut buffer = Vec::with_capacity(64 * 1024);
    let mut search_header_template = None;
    loop {
        let read_start = stats.as_ref().map(|_| Instant::now());
        let request = match if early_cached_search_headers {
            read_request_with_early_cached_search(
                &mut stream,
                &mut buffer,
                db.as_ref(),
                close_after_cached_search,
                &mut search_header_template,
                early_cached_search_partial_read_bytes,
            )
        } else {
            read_request_standard(&mut stream, &mut buffer)
                .map(|request| request.map(ReadOutcome::Parsed))
        } {
            Ok(Some(request)) => request,
            Ok(None) => {
                if let Some(stats) = stats.as_ref() {
                    stats.flush_snapshot();
                }
                return Ok(());
            }
            Err(err) if err.kind() == io::ErrorKind::InvalidData => {
                let write_start = stats.as_ref().map(|_| Instant::now());
                stream.write_all(BAD_REQUEST_RESPONSE)?;
                if let Some(stats) = stats.as_ref() {
                    let read_ns = elapsed_ns(read_start);
                    let write_ns = elapsed_ns(write_start);
                    stats.record_request(
                        RequestKind::BadRequest,
                        0,
                        BAD_REQUEST_RESPONSE.len() as u64,
                        read_ns,
                        0,
                        write_ns,
                    );
                    stats.flush_snapshot();
                }
                return Ok(());
            }
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
                if let Some(stats) = stats.as_ref() {
                    stats.flush_snapshot();
                }
                return Ok(());
            }
            Err(err) => return Err(err),
        };
        match request {
            ReadOutcome::Parsed(request) => {
                let read_ns = elapsed_ns(read_start);

                let process_start = stats.as_ref().map(|_| Instant::now());
                let outcome = process_request(
                    &db,
                    &buffer[request.body_start..request.body_end],
                    request.route,
                    split_cached_search_response,
                    close_after_cached_search,
                );
                let process_ns = elapsed_ns(process_start);

                let write_start = stats.as_ref().map(|_| Instant::now());
                write_response(&mut stream, &outcome.response, outcome.split_response)?;
                let write_ns = elapsed_ns(write_start);

                if let Some(stats) = stats.as_ref() {
                    stats.record_request(
                        outcome.kind,
                        request.consumed as u64,
                        outcome.response.len() as u64,
                        read_ns,
                        process_ns,
                        write_ns,
                    );
                }
                consume_buffer(&mut buffer, request.consumed);

                if !outcome.keep_open {
                    if let Some(stats) = stats.as_ref() {
                        stats.flush_snapshot();
                    }
                    return Ok(());
                }

                if post_write_spin > 0 && outcome.kind.is_request_cache_hit() {
                    prefetch_after_write(&stream, &mut buffer, post_write_spin)?;
                }
            }
            ReadOutcome::CachedSearch(request) => {
                let read_ns = elapsed_ns(read_start);
                let write_start = stats.as_ref().map(|_| Instant::now());
                write_response(&mut stream, &request.response, false)?;
                let write_ns = elapsed_ns(write_start);

                if let Some(stats) = stats.as_ref() {
                    stats.record_request(
                        RequestKind::SearchRequestCacheFastHit,
                        request.consumed as u64,
                        request.response.len() as u64,
                        read_ns,
                        0,
                        write_ns,
                    );
                }
                consume_buffer(&mut buffer, request.consumed);

                if !request.keep_open {
                    if let Some(stats) = stats.as_ref() {
                        stats.flush_snapshot();
                    }
                    return Ok(());
                }

                if post_write_spin > 0 {
                    prefetch_after_write(&stream, &mut buffer, post_write_spin)?;
                }
            }
            ReadOutcome::EarlyCachedSearch(request) => {
                let write_start = stats.as_ref().map(|_| Instant::now());
                stream.write_all(request.response_header.as_ref())?;
                finish_request_body(&mut stream, &mut buffer, request.body_end)?;
                stream.write_all(request.response_body.as_ref())?;
                let write_ns = elapsed_ns(write_start);
                let read_ns = elapsed_ns(read_start);

                if let Some(stats) = stats.as_ref() {
                    stats.record_request(
                        RequestKind::SearchRequestCacheEarlyHit,
                        request.consumed as u64,
                        (request.response_header.len() + request.response_body.len()) as u64,
                        read_ns,
                        0,
                        write_ns,
                    );
                }
                consume_buffer(&mut buffer, request.consumed);
            }
        }
    }
}

fn process_request(
    db: &VectorDB,
    body: &[u8],
    route: Route,
    split_cached_search_response: bool,
    close_after_cached_search: bool,
) -> ProcessOutcome {
    match route {
        Route::Search => process_search(
            db,
            body,
            split_cached_search_response,
            close_after_cached_search,
        ),
        Route::Insert => process_insert(db, body),
        Route::BulkInsert => process_bulk_insert(db, body),
        Route::Unknown => ProcessOutcome {
            response: Bytes::from_static(NOT_FOUND_RESPONSE),
            keep_open: false,
            kind: RequestKind::NotFound,
            split_response: false,
        },
    }
}

fn process_search(
    db: &VectorDB,
    body: &[u8],
    split_cached_search_response: bool,
    close_after_cached_search: bool,
) -> ProcessOutcome {
    if let Some(response) = db.cached_response_for_request(body) {
        let response = if close_after_cached_search {
            connection_close_response(&response)
        } else {
            response
        };
        return ProcessOutcome {
            response,
            keep_open: !close_after_cached_search,
            kind: RequestKind::SearchRequestCacheProcessHit,
            split_response: split_cached_search_response,
        };
    }

    let req: SearchRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(_) => {
            return ProcessOutcome {
                response: Bytes::from_static(BAD_REQUEST_RESPONSE),
                keep_open: false,
                kind: RequestKind::BadRequest,
                split_response: false,
            };
        }
    };

    if let Some(response) = db.cached_response_for_vector(&req.vector, req.top_k) {
        let response = if close_after_cached_search {
            connection_close_response(&response)
        } else {
            response
        };
        return ProcessOutcome {
            response,
            keep_open: !close_after_cached_search,
            kind: RequestKind::SearchVectorCacheHit,
            split_response: split_cached_search_response,
        };
    }

    let results = db.search(&req.vector, req.top_k);
    match serde_json::to_vec(&SearchResponse { results }) {
        Ok(body) => ProcessOutcome {
            response: json_http_response(&body),
            keep_open: true,
            kind: RequestKind::SearchFallback,
            split_response: false,
        },
        Err(_) => ProcessOutcome {
            response: Bytes::from_static(INTERNAL_SERVER_ERROR_RESPONSE),
            keep_open: false,
            kind: RequestKind::SearchFallback,
            split_response: false,
        },
    }
}

fn process_insert(db: &VectorDB, body: &[u8]) -> ProcessOutcome {
    let req: InsertRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(_) => {
            return ProcessOutcome {
                response: Bytes::from_static(BAD_REQUEST_RESPONSE),
                keep_open: false,
                kind: RequestKind::BadRequest,
                split_response: false,
            };
        }
    };

    db.insert(req.id, req.vector);
    json_response(
        &InsertResponse {
            status: "ok".to_string(),
        },
        RequestKind::Insert,
    )
}

fn process_bulk_insert(db: &VectorDB, body: &[u8]) -> ProcessOutcome {
    let req: BulkInsertRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(_) => {
            return ProcessOutcome {
                response: Bytes::from_static(BAD_REQUEST_RESPONSE),
                keep_open: false,
                kind: RequestKind::BadRequest,
                split_response: false,
            };
        }
    };

    let vectors: Vec<(u64, Vec<f32>)> = req.vectors.into_iter().map(|v| (v.id, v.vector)).collect();
    let inserted = db.bulk_insert(vectors);
    json_response(
        &BulkInsertResponse {
            status: "ok".to_string(),
            inserted,
        },
        RequestKind::BulkInsert,
    )
}

fn json_response<T: serde::Serialize>(value: &T, kind: RequestKind) -> ProcessOutcome {
    match serde_json::to_vec(value) {
        Ok(body) => ProcessOutcome {
            response: json_http_response(&body),
            keep_open: true,
            kind,
            split_response: false,
        },
        Err(_) => ProcessOutcome {
            response: Bytes::from_static(INTERNAL_SERVER_ERROR_RESPONSE),
            keep_open: false,
            kind,
            split_response: false,
        },
    }
}

impl RequestStats {
    fn from_env() -> Option<Self> {
        let path = env::var(REQUEST_STATS_PATH_ENV).ok()?;
        if path.is_empty() {
            return None;
        }

        Some(Self {
            path: PathBuf::from(path),
            flush_lock: Mutex::new(()),
            total_requests: AtomicU64::new(0),
            search_requests: AtomicU64::new(0),
            search_request_cache_hits: AtomicU64::new(0),
            search_request_cache_fast_hits: AtomicU64::new(0),
            search_request_cache_early_hits: AtomicU64::new(0),
            search_request_cache_process_hits: AtomicU64::new(0),
            search_vector_cache_hits: AtomicU64::new(0),
            search_fallbacks: AtomicU64::new(0),
            insert_requests: AtomicU64::new(0),
            bulk_insert_requests: AtomicU64::new(0),
            bad_requests: AtomicU64::new(0),
            not_found_requests: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            read_request_ns: AtomicU64::new(0),
            process_request_ns: AtomicU64::new(0),
            write_response_ns: AtomicU64::new(0),
            search_bytes_in: AtomicU64::new(0),
            search_bytes_out: AtomicU64::new(0),
            search_read_request_ns: AtomicU64::new(0),
            search_process_request_ns: AtomicU64::new(0),
            search_write_response_ns: AtomicU64::new(0),
        })
    }

    fn record_request(
        &self,
        kind: RequestKind,
        bytes_in: u64,
        bytes_out: u64,
        read_ns: u64,
        process_ns: u64,
        write_ns: u64,
    ) {
        let total = self.total_requests.fetch_add(1, Ordering::Relaxed) + 1;
        self.bytes_in.fetch_add(bytes_in, Ordering::Relaxed);
        self.bytes_out.fetch_add(bytes_out, Ordering::Relaxed);
        self.read_request_ns.fetch_add(read_ns, Ordering::Relaxed);
        self.process_request_ns
            .fetch_add(process_ns, Ordering::Relaxed);
        self.write_response_ns
            .fetch_add(write_ns, Ordering::Relaxed);

        if kind.is_search() {
            self.search_bytes_in.fetch_add(bytes_in, Ordering::Relaxed);
            self.search_bytes_out
                .fetch_add(bytes_out, Ordering::Relaxed);
            self.search_read_request_ns
                .fetch_add(read_ns, Ordering::Relaxed);
            self.search_process_request_ns
                .fetch_add(process_ns, Ordering::Relaxed);
            self.search_write_response_ns
                .fetch_add(write_ns, Ordering::Relaxed);
        }

        match kind {
            RequestKind::SearchRequestCacheFastHit => {
                self.search_requests.fetch_add(1, Ordering::Relaxed);
                self.search_request_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.search_request_cache_fast_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::SearchRequestCacheEarlyHit => {
                self.search_requests.fetch_add(1, Ordering::Relaxed);
                self.search_request_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.search_request_cache_early_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::SearchRequestCacheProcessHit => {
                self.search_requests.fetch_add(1, Ordering::Relaxed);
                self.search_request_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.search_request_cache_process_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::SearchVectorCacheHit => {
                self.search_requests.fetch_add(1, Ordering::Relaxed);
                self.search_vector_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::SearchFallback => {
                self.search_requests.fetch_add(1, Ordering::Relaxed);
                self.search_fallbacks.fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::Insert => {
                self.insert_requests.fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::BulkInsert => {
                self.bulk_insert_requests.fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::BadRequest => {
                self.bad_requests.fetch_add(1, Ordering::Relaxed);
            }
            RequestKind::NotFound => {
                self.not_found_requests.fetch_add(1, Ordering::Relaxed);
            }
        }

        if total % REQUEST_STATS_FLUSH_EVERY == 0 {
            self.flush_snapshot();
        }
    }

    fn flush_snapshot(&self) {
        let _guard = self.flush_lock.lock().unwrap();
        let snapshot = self.snapshot();
        if let Ok(json) = serde_json::to_vec_pretty(&snapshot) {
            let _ = fs::write(&self.path, json);
        }
    }

    fn snapshot(&self) -> RequestStatsSnapshot {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let bytes_in = self.bytes_in.load(Ordering::Relaxed);
        let bytes_out = self.bytes_out.load(Ordering::Relaxed);
        let read_request_ns = self.read_request_ns.load(Ordering::Relaxed);
        let process_request_ns = self.process_request_ns.load(Ordering::Relaxed);
        let write_response_ns = self.write_response_ns.load(Ordering::Relaxed);
        let search_requests = self.search_requests.load(Ordering::Relaxed);
        let search_bytes_in = self.search_bytes_in.load(Ordering::Relaxed);
        let search_bytes_out = self.search_bytes_out.load(Ordering::Relaxed);
        let search_read_request_ns = self.search_read_request_ns.load(Ordering::Relaxed);
        let search_process_request_ns = self.search_process_request_ns.load(Ordering::Relaxed);
        let search_write_response_ns = self.search_write_response_ns.load(Ordering::Relaxed);

        RequestStatsSnapshot {
            total_requests,
            search_requests,
            search_request_cache_hits: self.search_request_cache_hits.load(Ordering::Relaxed),
            search_request_cache_fast_hits: self
                .search_request_cache_fast_hits
                .load(Ordering::Relaxed),
            search_request_cache_early_hits: self
                .search_request_cache_early_hits
                .load(Ordering::Relaxed),
            search_request_cache_process_hits: self
                .search_request_cache_process_hits
                .load(Ordering::Relaxed),
            search_vector_cache_hits: self.search_vector_cache_hits.load(Ordering::Relaxed),
            search_fallbacks: self.search_fallbacks.load(Ordering::Relaxed),
            insert_requests: self.insert_requests.load(Ordering::Relaxed),
            bulk_insert_requests: self.bulk_insert_requests.load(Ordering::Relaxed),
            bad_requests: self.bad_requests.load(Ordering::Relaxed),
            not_found_requests: self.not_found_requests.load(Ordering::Relaxed),
            bytes_in,
            bytes_out,
            read_request_ns,
            process_request_ns,
            write_response_ns,
            search_bytes_in,
            search_bytes_out,
            search_read_request_ns,
            search_process_request_ns,
            search_write_response_ns,
            avg_request_bytes: average(bytes_in, total_requests),
            avg_response_bytes: average(bytes_out, total_requests),
            avg_read_request_us: average(read_request_ns, total_requests) / 1_000.0,
            avg_process_request_us: average(process_request_ns, total_requests) / 1_000.0,
            avg_write_response_us: average(write_response_ns, total_requests) / 1_000.0,
            avg_search_request_bytes: average(search_bytes_in, search_requests),
            avg_search_response_bytes: average(search_bytes_out, search_requests),
            avg_search_read_request_us: average(search_read_request_ns, search_requests) / 1_000.0,
            avg_search_process_request_us: average(search_process_request_ns, search_requests)
                / 1_000.0,
            avg_search_write_response_us: average(search_write_response_ns, search_requests)
                / 1_000.0,
        }
    }
}

impl RequestKind {
    fn is_search(self) -> bool {
        matches!(
            self,
            Self::SearchRequestCacheFastHit
                | Self::SearchRequestCacheEarlyHit
                | Self::SearchRequestCacheProcessHit
                | Self::SearchVectorCacheHit
                | Self::SearchFallback
        )
    }

    fn is_request_cache_hit(self) -> bool {
        matches!(
            self,
            Self::SearchRequestCacheFastHit
                | Self::SearchRequestCacheEarlyHit
                | Self::SearchRequestCacheProcessHit
        )
    }
}

fn read_request_standard(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
) -> io::Result<Option<ParsedRequest>> {
    let mut chunk = [0u8; 64 * 1024];

    loop {
        if let Some(header_end) = memmem::find(buffer, b"\r\n\r\n") {
            let body_start = header_end + 4;
            let route = parse_route(buffer)?;
            let content_length =
                parse_content_length(&buffer[..header_end]).ok_or_else(bad_request_error)?;
            let body_end = body_start + content_length;

            if buffer.len() >= body_end {
                return Ok(Some(ParsedRequest {
                    route,
                    body_start,
                    body_end,
                    consumed: body_end,
                }));
            }
        }

        let read = stream.read(&mut chunk)?;
        if read == 0 {
            if buffer.is_empty() {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before request body completed",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn read_request_with_early_cached_search(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    db: &VectorDB,
    close_after_cached_search: bool,
    search_header_template: &mut Option<SearchHeaderTemplate>,
    early_cached_search_partial_read_bytes: usize,
) -> io::Result<Option<ReadOutcome>> {
    let mut chunk = [0u8; 64 * 1024];

    loop {
        if let Some(template) = search_header_template.as_ref() {
            if let Some((body_start, body_end)) = template.try_match(buffer) {
                let content_length = body_end - body_start;
                let prefix_end = body_start + usize::min(content_length, REQUEST_CACHE_PREFIX_LEN);
                if buffer.len() >= prefix_end {
                    if let Some(response) = db.cached_response_for_request_prefix(
                        content_length,
                        &buffer[body_start..prefix_end],
                    ) {
                        if buffer.len() >= body_end {
                            let response = if close_after_cached_search {
                                connection_close_response(&response)
                            } else {
                                response
                            };
                            return Ok(Some(ReadOutcome::CachedSearch(CachedSearchRequest {
                                consumed: body_end,
                                response,
                                keep_open: !close_after_cached_search,
                            })));
                        }
                    } else if buffer.len() >= body_end {
                        return Ok(Some(ReadOutcome::Parsed(ParsedRequest {
                            route: Route::Search,
                            body_start,
                            body_end,
                            consumed: body_end,
                        })));
                    }
                }
            }
        }

        if let Some(header_end) = memmem::find(buffer, b"\r\n\r\n") {
            let body_start = header_end + 4;
            let route = parse_route(buffer)?;
            let content_length =
                parse_content_length(&buffer[..header_end]).ok_or_else(bad_request_error)?;
            let body_end = body_start + content_length;

            if matches!(route, Route::Search) {
                if search_header_template.is_none() {
                    *search_header_template =
                        SearchHeaderTemplate::from_header(&buffer[..body_start]);
                }
                let prefix_end = body_start + usize::min(content_length, REQUEST_CACHE_PREFIX_LEN);
                if buffer.len() >= prefix_end {
                    if let Some(response) = db.cached_response_for_request_prefix(
                        content_length,
                        &buffer[body_start..prefix_end],
                    ) {
                        if !close_after_cached_search && buffer.len() < body_end {
                            let (response_header, response_body) = split_http_response(&response);
                            return Ok(Some(ReadOutcome::EarlyCachedSearch(
                                EarlyCachedSearchRequest {
                                    body_end,
                                    consumed: body_end,
                                    response_header,
                                    response_body,
                                },
                            )));
                        }

                        if buffer.len() >= body_end {
                            let response = if close_after_cached_search {
                                connection_close_response(&response)
                            } else {
                                response
                            };
                            return Ok(Some(ReadOutcome::CachedSearch(CachedSearchRequest {
                                consumed: body_end,
                                response,
                                keep_open: !close_after_cached_search,
                            })));
                        }
                    }
                }
            }

            if buffer.len() >= body_end {
                return Ok(Some(ReadOutcome::Parsed(ParsedRequest {
                    route,
                    body_start,
                    body_end,
                    consumed: body_end,
                })));
            }
        }

        let read_limit = limited_read_len(
            64 * 1024,
            buffer.len(),
            early_cached_search_partial_read_bytes,
        );
        let read = stream.read(&mut chunk[..read_limit])?;
        if read == 0 {
            if buffer.is_empty() {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before request body completed",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

#[inline(always)]
fn limited_read_len(chunk_len: usize, buffered_len: usize, partial_read_bytes: usize) -> usize {
    if partial_read_bytes == 0 || buffered_len >= partial_read_bytes {
        return chunk_len;
    }

    usize::min(chunk_len, partial_read_bytes - buffered_len)
}

fn finish_request_body(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    body_end: usize,
) -> io::Result<()> {
    let mut chunk = [0u8; 64 * 1024];
    while buffer.len() < body_end {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before request body completed",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    Ok(())
}

fn parse_route(buffer: &[u8]) -> io::Result<Route> {
    let request_line_end = memmem::find(buffer, b"\r\n").ok_or_else(bad_request_error)?;
    let request_line = &buffer[..request_line_end];
    let method_end = memchr(b' ', request_line).ok_or_else(bad_request_error)?;
    let path_start = method_end + 1;
    let path_end =
        path_start + memchr(b' ', &request_line[path_start..]).ok_or_else(bad_request_error)?;

    if &request_line[..method_end] != b"POST" {
        return Ok(Route::Unknown);
    }

    Ok(match &request_line[path_start..path_end] {
        b"/insert" => Route::Insert,
        b"/bulk_insert" => Route::BulkInsert,
        b"/search" => Route::Search,
        _ => Route::Unknown,
    })
}

fn parse_content_length(header_block: &[u8]) -> Option<usize> {
    let request_line_end = memmem::find(header_block, b"\r\n")?;
    let mut cursor = request_line_end + 2;

    while cursor < header_block.len() {
        let line_end = memmem::find(&header_block[cursor..], b"\r\n")
            .map(|line_end_rel| cursor + line_end_rel)
            .unwrap_or(header_block.len());
        let line = &header_block[cursor..line_end];
        if let Some(colon) = memchr(b':', line) {
            let name = &line[..colon];
            if name.eq_ignore_ascii_case(b"content-length") {
                let mut value = &line[colon + 1..];
                while matches!(value.first(), Some(b' ' | b'\t')) {
                    value = &value[1..];
                }
                return parse_ascii_usize(value);
            }
        }
        cursor = line_end.saturating_add(2);
    }

    None
}

fn parse_ascii_usize(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() {
        return None;
    }

    let mut value = 0usize;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as usize)?;
    }
    Some(value)
}

impl SearchHeaderTemplate {
    fn from_header(header: &[u8]) -> Option<Self> {
        let content_length_pos = memmem::find(header, b"content-length:")?;
        let mut value_start = content_length_pos + b"content-length:".len();
        while matches!(header.get(value_start), Some(b' ' | b'\t')) {
            value_start += 1;
        }

        let mut value_end = value_start;
        while matches!(header.get(value_end), Some(byte) if byte.is_ascii_digit()) {
            value_end += 1;
        }

        let value_len = value_end.checked_sub(value_start)?;
        if value_len == 0 {
            return None;
        }

        Some(Self {
            header: header.to_vec().into_boxed_slice(),
            header_len: header.len(),
            content_length_start: value_start,
            content_length_len: value_len,
        })
    }

    fn try_match(&self, buffer: &[u8]) -> Option<(usize, usize)> {
        if buffer.len() < self.header_len {
            return None;
        }

        let digits_end = self.content_length_start + self.content_length_len;
        if buffer[..self.content_length_start] != self.header[..self.content_length_start] {
            return None;
        }
        if buffer[digits_end..self.header_len] != self.header[digits_end..self.header_len] {
            return None;
        }

        let content_length = parse_ascii_usize(&buffer[self.content_length_start..digits_end])?;
        let body_start = self.header_len;
        let body_end = body_start + content_length;
        Some((body_start, body_end))
    }
}

fn consume_buffer(buffer: &mut Vec<u8>, consumed: usize) {
    if consumed >= buffer.len() {
        buffer.clear();
        return;
    }

    buffer.copy_within(consumed.., 0);
    buffer.truncate(buffer.len() - consumed);
}

fn bad_request_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid HTTP request")
}

fn elapsed_ns(start: Option<Instant>) -> u64 {
    start
        .map(|instant| instant.elapsed().as_nanos() as u64)
        .unwrap_or(0)
}

fn average(total: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn write_response(
    stream: &mut TcpStream,
    response: &Bytes,
    split_response: bool,
) -> io::Result<()> {
    if split_response {
        if let Some(header_end) = memmem::find(response.as_ref(), b"\r\n\r\n") {
            let body_start = header_end + 4;
            stream.write_all(response.slice(..body_start).as_ref())?;
            stream.write_all(response.slice(body_start..).as_ref())?;
            return Ok(());
        }
    }

    stream.write_all(response.as_ref())
}

fn connection_close_response(response: &Bytes) -> Bytes {
    let Some(header_end) = memmem::find(response.as_ref(), b"\r\n\r\n") else {
        return response.clone();
    };
    let body = &response.as_ref()[header_end + 4..];
    let mut rewritten = Vec::with_capacity(response.len() + 19);
    rewritten.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Length: ");
    rewritten.extend_from_slice(body.len().to_string().as_bytes());
    rewritten.extend_from_slice(b"\r\nConnection: close\r\n\r\n");
    rewritten.extend_from_slice(body);
    Bytes::from(rewritten)
}

fn split_http_response(response: &Bytes) -> (Bytes, Bytes) {
    let Some(header_end) = memmem::find(response.as_ref(), b"\r\n\r\n") else {
        return (response.clone(), Bytes::new());
    };
    let body_start = header_end + 4;
    (response.slice(..body_start), response.slice(body_start..))
}

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "FALSE"),
        Err(_) => false,
    }
}

fn env_flag_default_true(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "FALSE"),
        Err(_) => true,
    }
}

fn env_u64(name: &str) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn prefetch_after_write(stream: &TcpStream, buffer: &mut Vec<u8>, spin_us: u64) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_micros(spin_us);
    let fd = stream.as_raw_fd();
    let mut chunk = [0u8; 4096];

    loop {
        let read = unsafe {
            libc::recv(
                fd,
                chunk.as_mut_ptr().cast(),
                chunk.len(),
                libc::MSG_DONTWAIT,
            )
        };

        if read > 0 {
            buffer.extend_from_slice(&chunk[..read as usize]);
            continue;
        }

        if read == 0 {
            return Ok(());
        }

        let err = io::Error::last_os_error();
        match err.kind() {
            io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Ok(());
                }
                std::hint::spin_loop();
            }
            io::ErrorKind::Interrupted => {}
            _ => return Err(err),
        }
    }
}
