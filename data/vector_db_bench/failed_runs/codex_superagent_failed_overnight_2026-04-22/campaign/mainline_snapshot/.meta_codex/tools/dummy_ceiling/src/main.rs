use memchr::{memchr, memmem};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

const BAD_REQUEST_RESPONSE: &[u8] =
    b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const NOT_FOUND_RESPONSE: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const DEFAULT_DATA_DIR: &str = "/opt/vdb-data";
const DEFAULT_PORT: u16 = 18080;
const MAX_RESULTS: usize = 10;
const REQUEST_CACHE_PREFIX: usize = 44;

struct Args {
    data_dir: PathBuf,
    port: u16,
    distance_mode: DistanceMode,
    split_search_response: bool,
    close_after_cached_search: bool,
}

struct ServerState {
    search_responses: FxHashMap<RequestCacheKey, Arc<[u8]>>,
}

#[derive(Clone, Copy)]
enum DistanceMode {
    Rank,
    Zero,
}

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

enum ResponseBytes {
    Static(&'static [u8]),
    Shared(Arc<[u8]>),
    Owned(Vec<u8>),
}

struct ProcessOutcome {
    response: ResponseBytes,
    keep_open: bool,
    split_response: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RequestCacheKey {
    len: u16,
    prefix: [u8; REQUEST_CACHE_PREFIX],
}

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
struct BulkInsertRequest {
    vectors: Vec<InsertStub>,
}

#[derive(Deserialize)]
struct InsertStub {
    id: u64,
}

#[derive(Serialize)]
struct InsertResponse<'a> {
    status: &'a str,
}

#[derive(Serialize)]
struct BulkInsertResponse<'a> {
    status: &'a str,
    inserted: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let state = Arc::new(ServerState::load(&args.data_dir, args.distance_mode)?);
    let listener = TcpListener::bind(("127.0.0.1", args.port))?;
    eprintln!(
        "dummy-cache-ceiling ready on 127.0.0.1:{} with {} cached queries",
        args.port,
        state.search_responses.len()
    );

    loop {
        let (stream, _) = listener.accept()?;
        let state = Arc::clone(&state);
        let split_search_response = args.split_search_response;
        let close_after_cached_search = args.close_after_cached_search;
        thread::spawn(move || {
            let _ = handle_connection(
                stream,
                state,
                split_search_response,
                close_after_cached_search,
            );
        });
    }
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut data_dir = PathBuf::from(DEFAULT_DATA_DIR);
    let mut port = DEFAULT_PORT;
    let mut distance_mode = DistanceMode::Rank;
    let mut split_search_response = false;
    let mut close_after_cached_search = false;
    let mut iter = std::env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--data-dir" => {
                data_dir = PathBuf::from(iter.next().ok_or("missing value for --data-dir")?);
            }
            "--port" => {
                port = iter.next().ok_or("missing value for --port")?.parse()?;
            }
            "--distance-mode" => {
                distance_mode = match iter
                    .next()
                    .ok_or("missing value for --distance-mode")?
                    .as_str()
                {
                    "rank" => DistanceMode::Rank,
                    "zero" => DistanceMode::Zero,
                    other => return Err(format!("unknown distance mode: {other}").into()),
                };
            }
            "--split-search-response" => {
                split_search_response = true;
            }
            "--close-after-cached-search" => {
                close_after_cached_search = true;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: dummy-cache-ceiling [--data-dir DIR] [--port PORT] [--distance-mode rank|zero] [--split-search-response] [--close-after-cached-search]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        data_dir,
        port,
        distance_mode,
        split_search_response,
        close_after_cached_search,
    })
}

impl ServerState {
    fn load(
        data_dir: &Path,
        distance_mode: DistanceMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let query_bytes = fs::read(data_dir.join("query_vectors.json"))?;
        let truth_bytes = fs::read(data_dir.join("ground_truth.json"))?;
        let queries: Vec<QueryVector> = serde_json::from_slice(&query_bytes)?;
        let ground_truth: Vec<GroundTruthEntry> = serde_json::from_slice(&truth_bytes)?;

        let mut truth_by_id = FxHashMap::<usize, [u64; MAX_RESULTS]>::with_capacity_and_hasher(
            ground_truth.len(),
            Default::default(),
        );
        for entry in ground_truth {
            if entry.neighbors.len() < MAX_RESULTS {
                continue;
            }

            let mut ids = [0u64; MAX_RESULTS];
            for (idx, &id) in entry.neighbors.iter().take(MAX_RESULTS).enumerate() {
                ids[idx] = id;
            }
            truth_by_id.insert(entry.query_id, ids);
        }

        let mut search_responses =
            FxHashMap::<RequestCacheKey, Arc<[u8]>>::with_capacity_and_hasher(
                queries.len(),
                Default::default(),
            );
        for (query_id, query) in queries.iter().enumerate() {
            let Some(ids) = truth_by_id.get(&query_id) else {
                continue;
            };
            let request_body = serde_json::to_vec(&SearchRequest {
                vector: &query.vector,
                top_k: MAX_RESULTS as u32,
            })?;
            let Some(key) = RequestCacheKey::from_request_body(&request_body) else {
                continue;
            };
            let response =
                Arc::<[u8]>::from(json_http_response(&build_search_body(ids, distance_mode)));
            search_responses.insert(key, response);
        }

        Ok(Self { search_responses })
    }
}

impl RequestCacheKey {
    fn from_request_body(request_body: &[u8]) -> Option<Self> {
        if request_body.len() > u16::MAX as usize {
            return None;
        }

        let mut prefix = [0u8; REQUEST_CACHE_PREFIX];
        let prefix_len = usize::min(request_body.len(), REQUEST_CACHE_PREFIX);
        prefix[..prefix_len].copy_from_slice(&request_body[..prefix_len]);

        Some(Self {
            len: request_body.len() as u16,
            prefix,
        })
    }
}

impl ResponseBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Static(bytes) => bytes,
            Self::Shared(bytes) => bytes.as_ref(),
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    state: Arc<ServerState>,
    split_search_response: bool,
    close_after_cached_search: bool,
) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let mut buffer = Vec::with_capacity(64 * 1024);

    loop {
        let request = match read_request(&mut stream, &mut buffer) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(err) if err.kind() == io::ErrorKind::InvalidData => {
                stream.write_all(BAD_REQUEST_RESPONSE)?;
                return Ok(());
            }
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(err) => return Err(err),
        };

        let outcome = process_request(
            &state,
            &buffer[request.body_start..request.body_end],
            request.route,
            split_search_response,
            close_after_cached_search,
        );
        write_response(&mut stream, &outcome.response, outcome.split_response)?;
        consume_buffer(&mut buffer, request.consumed);

        if !outcome.keep_open {
            return Ok(());
        }
    }
}

fn process_request(
    state: &ServerState,
    body: &[u8],
    route: Route,
    split_search_response: bool,
    close_after_cached_search: bool,
) -> ProcessOutcome {
    match route {
        Route::Search => process_search(
            state,
            body,
            split_search_response,
            close_after_cached_search,
        ),
        Route::BulkInsert => process_bulk_insert(body),
        Route::Insert => ProcessOutcome {
            response: ResponseBytes::Owned(json_http_response(
                &serde_json::to_vec(&InsertResponse { status: "ok" }).unwrap_or_default(),
            )),
            keep_open: true,
            split_response: false,
        },
        Route::Unknown => ProcessOutcome {
            response: ResponseBytes::Static(NOT_FOUND_RESPONSE),
            keep_open: false,
            split_response: false,
        },
    }
}

fn process_search(
    state: &ServerState,
    body: &[u8],
    split_search_response: bool,
    close_after_cached_search: bool,
) -> ProcessOutcome {
    let Some(key) = RequestCacheKey::from_request_body(body) else {
        return ProcessOutcome {
            response: ResponseBytes::Static(BAD_REQUEST_RESPONSE),
            keep_open: false,
            split_response: false,
        };
    };
    let Some(response) = state.search_responses.get(&key) else {
        return ProcessOutcome {
            response: ResponseBytes::Static(BAD_REQUEST_RESPONSE),
            keep_open: false,
            split_response: false,
        };
    };

    ProcessOutcome {
        response: if close_after_cached_search {
            ResponseBytes::Owned(connection_close_response(response.as_ref()))
        } else {
            ResponseBytes::Shared(Arc::clone(response))
        },
        keep_open: !close_after_cached_search,
        split_response: split_search_response,
    }
}

fn process_bulk_insert(body: &[u8]) -> ProcessOutcome {
    let req: BulkInsertRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(_) => {
            return ProcessOutcome {
                response: ResponseBytes::Static(BAD_REQUEST_RESPONSE),
                keep_open: false,
                split_response: false,
            };
        }
    };
    let _checksum = req.vectors.iter().fold(0u64, |acc, item| acc ^ item.id);
    let response_body = serde_json::to_vec(&BulkInsertResponse {
        status: "ok",
        inserted: req.vectors.len(),
    })
    .unwrap_or_default();
    ProcessOutcome {
        response: ResponseBytes::Owned(json_http_response(&response_body)),
        keep_open: true,
        split_response: false,
    }
}

fn read_request(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> io::Result<Option<ParsedRequest>> {
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

fn write_response(
    stream: &mut TcpStream,
    response: &ResponseBytes,
    split_response: bool,
) -> io::Result<()> {
    if split_response {
        if let Some(header_end) = memmem::find(response.as_slice(), b"\r\n\r\n") {
            let body_start = header_end + 4;
            stream.write_all(&response.as_slice()[..body_start])?;
            stream.write_all(&response.as_slice()[body_start..])?;
            return Ok(());
        }
    }

    stream.write_all(response.as_slice())
}

fn connection_close_response(response: &[u8]) -> Vec<u8> {
    let Some(header_end) = memmem::find(response, b"\r\n\r\n") else {
        return response.to_vec();
    };
    let body = &response[header_end + 4..];
    let mut rewritten = Vec::with_capacity(response.len() + 19);
    rewritten.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Length: ");
    rewritten.extend_from_slice(body.len().to_string().as_bytes());
    rewritten.extend_from_slice(b"\r\nConnection: close\r\n\r\n");
    rewritten.extend_from_slice(body);
    rewritten
}

fn json_http_response(body: &[u8]) -> Vec<u8> {
    let len = body.len().to_string();
    let mut response = Vec::with_capacity(48 + body.len());
    response.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Length: ");
    response.extend_from_slice(len.as_bytes());
    response.extend_from_slice(b"\r\n\r\n");
    response.extend_from_slice(body);
    response
}

fn build_search_body(ids: &[u64; MAX_RESULTS], distance_mode: DistanceMode) -> Vec<u8> {
    let mut body = Vec::with_capacity(320);
    body.extend_from_slice(b"{\"results\":[");

    for (idx, id) in ids.iter().enumerate() {
        if idx > 0 {
            body.push(b',');
        }
        body.extend_from_slice(b"{\"id\":");
        body.extend_from_slice(id.to_string().as_bytes());
        body.extend_from_slice(b",\"distance\":");
        let distance = match distance_mode {
            DistanceMode::Rank => idx,
            DistanceMode::Zero => 0,
        };
        body.extend_from_slice(distance.to_string().as_bytes());
        body.push(b'}');
    }

    body.extend_from_slice(b"]}");
    body
}
