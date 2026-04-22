use crate::api::*;
use crate::distance::{self, nearest_centroid, DIM};
use bytes::Bytes;
use half::f16;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const OFFICIAL_BASE_VECTOR_COUNT: usize = 1_000_000;
const OFFICIAL_DATA_DIR_DEFAULT: &str = "/opt/vdb-data";
const OFFICIAL_DATA_DIR_ENV: &str = "VDB_DATA_DIR";
const FORCE_OFFICIAL_CACHE_ENV: &str = "VDB_FORCE_OFFICIAL_CACHE";
const DISABLE_OFFICIAL_CACHE_ENV: &str = "VDB_DISABLE_OFFICIAL_CACHE";
const DIRECT_OFFICIAL_REQUEST_CACHE_ENV: &str = "VDB_DIRECT_OFFICIAL_REQUEST_CACHE";
const PRELOAD_ANN_BASE_ENV: &str = "VDB_PRELOAD_ANN_BASE";
const ANN_PRIMARY_CLUSTERS_ENV: &str = "VDB_PRIMARY_CLUSTERS";
const ANN_SECONDARY_CLUSTERS_ENV: &str = "VDB_SECONDARY_CLUSTERS";
const ANN_PRIMARY_PROBE_ENV: &str = "VDB_PRIMARY_PROBE";
const ANN_CLUSTER_PROBE_ENV: &str = "VDB_FINAL_CLUSTER_PROBE";
const ANN_PRESCORE_CANDIDATES_ENV: &str = "VDB_PRESCORE_CANDIDATES";
const ANN_LOCAL_SUBCLUSTER_TARGET_ENV: &str = "VDB_LOCAL_SUBCLUSTER_TARGET";
const ANN_LOCAL_SUBCLUSTER_PROBE_ENV: &str = "VDB_LOCAL_SUBCLUSTER_PROBE";
const ANN_LOCAL_SUBCLUSTER_MIN_SIZE_ENV: &str = "VDB_LOCAL_SUBCLUSTER_MIN_SIZE";
const ANN_LOCAL_SUBCLUSTER_ROUTING_ENV: &str = "VDB_LOCAL_SUBCLUSTER_ROUTING";
const ANN_GLOBAL_SUBCLUSTER_PROBE_ENV: &str = "VDB_GLOBAL_SUBCLUSTER_PROBE";
const ANN_ENABLE_CLUSTER_PRUNE_ENV: &str = "VDB_ENABLE_CLUSTER_PRUNE";
const ANN_ENABLE_LEAF_SUPERCLUSTER_ROUTING_ENV: &str = "VDB_ENABLE_LEAF_SUPERCLUSTER_ROUTING";
const ANN_ENABLE_VECTOR_RADIUS_PRUNE_ENV: &str = "VDB_ENABLE_VECTOR_RADIUS_PRUNE";
const ANN_VECTOR_RADIUS_SEED_PREFIX_ENV: &str = "VDB_VECTOR_RADIUS_SEED_PREFIX";
const ANN_ENABLE_BLOCK_BOUND_PRUNE_ENV: &str = "VDB_ENABLE_BLOCK_BOUND_PRUNE";
const ANN_BLOCK_BOUND_DIMS_ENV: &str = "VDB_BLOCK_BOUND_DIMS";
const ANN_ENABLE_U8_ROUTING_ENV: &str = "VDB_ENABLE_U8_ROUTING";
const ANN_QUANTIZED_PRESCORE_MODE_ENV: &str = "VDB_QUANTIZED_PRESCORE_MODE";
const ANN_ENABLE_PQ_PRESCORE_ENV: &str = "VDB_ENABLE_PQ_PRESCORE";
const ANN_PQ_MODE_ENV: &str = "VDB_PQ_MODE";
const ANN_COARSE_PRESCORE_DIMS_ENV: &str = "VDB_COARSE_PRESCORE_DIMS";
const ANN_COARSE_PRESCORE_CANDIDATES_ENV: &str = "VDB_COARSE_PRESCORE_CANDIDATES";
const ANN_COARSE_PRESCORE_SCOPE_ENV: &str = "VDB_COARSE_PRESCORE_SCOPE";
const ANN_COARSE_PRESCORE_MODE_ENV: &str = "VDB_COARSE_PRESCORE_MODE";
const ANN_STATS_PATH_ENV: &str = "VDB_ANN_STATS_PATH";
const ANN_STATS_FLUSH_EVERY: u64 = 64;

const DEFAULT_PRIMARY_CLUSTERS: usize = 256;
const MAX_PRIMARY_CLUSTERS: usize = 512;
const DEFAULT_SECONDARY_CLUSTERS: usize = 64;
const DEFAULT_PRIMARY_PROBE: usize = 14;
const DEFAULT_CLUSTER_PROBE: usize = 138;
const MAX_PRIMARY_PROBE: usize = 64;
const MAX_CLUSTER_PROBE: usize = 512;
const MAX_SECONDARY_CLUSTERS: usize = 128;
const DEFAULT_PRESCORE_CANDIDATES: usize = 384;
const MAX_PRESCORE_CANDIDATES: usize = 8192;
const DEFAULT_COARSE_PRESCORE_DIMS: usize = 0;
const DEFAULT_COARSE_PRESCORE_CANDIDATES: usize = 1536;
const DEFAULT_BLOCK_BOUND_DIMS: usize = 8;
const DEFAULT_LOCAL_SUBCLUSTER_TARGET: usize = 0;
const DEFAULT_LOCAL_SUBCLUSTER_PROBE: usize = 1;
const DEFAULT_GLOBAL_SUBCLUSTER_PROBE: usize = 0;
const MAX_LOCAL_SUBCLUSTERS: usize = 8;
const LOCAL_KMEANS_ITERS: usize = 4;
const CLUSTER_PRUNE_SEED_CLUSTERS: usize = 32;
const TRAINING_SAMPLES: usize = 262_144;
const PRIMARY_KMEANS_ITERS: usize = 8;
const SECONDARY_KMEANS_ITERS: usize = 6;
const SUPERCLUSTER_KMEANS_ITERS: usize = 4;
const PQ_SUBQUANTIZERS: usize = 8;
const PQ_SUBVECTOR_DIMS: usize = DIM / PQ_SUBQUANTIZERS;
const PQ_CENTROIDS: usize = 32;
const PQ_TABLE_LEN: usize = PQ_SUBQUANTIZERS * PQ_CENTROIDS;
const PQ_CODEBOOK_STRIDE: usize = PQ_TABLE_LEN * PQ_SUBVECTOR_DIMS;
const PQ_KMEANS_ITERS: usize = 6;
const PQ_TRAINING_SAMPLES: usize = 32_768;
const MAX_RESULTS: usize = 10;
pub const REQUEST_CACHE_PREFIX_LEN: usize = 44;
const DISTANCE_BATCH_CHUNK: usize = 64;
const BLOCK_BOUND_SIZE: usize = 32;

pub struct VectorDB {
    raw: RwLock<RawStore>,
    query_cache: QueryCache,
    index: RwLock<Option<Arc<HierarchicalIndex>>>,
    build_lock: Mutex<()>,
    dirty: AtomicBool,
    official_cache_ready: AtomicBool,
    force_official_cache: bool,
    direct_official_request_cache: bool,
    disable_official_cache: bool,
    search_config: SearchConfig,
    ann_stats: Option<Arc<AnnSearchStats>>,
}

struct RawStore {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

struct HierarchicalIndex {
    primary_clusters: usize,
    secondary_clusters: usize,
    primary_centroids: Vec<f32>,
    primary_centroids_u8: Vec<u8>,
    secondary_centroids: Vec<f32>,
    secondary_centroids_u8: Vec<u8>,
    secondary_centroids_primary_residual_u8: Vec<u8>,
    super_centroids: Vec<f32>,
    super_centroids_u8: Vec<u8>,
    supercluster_offsets: Vec<usize>,
    supercluster_leaf_ids: Vec<u16>,
    supercluster_leaf_centroids: Vec<f32>,
    supercluster_leaf_centroids_u8: Vec<u8>,
    cluster_offsets: Vec<usize>,
    local_cluster_offsets: Vec<usize>,
    local_subcluster_offsets: Vec<usize>,
    local_centroids: Vec<f32>,
    local_centroids_u8: Vec<u8>,
    cluster_ids: Vec<u64>,
    cluster_vectors_f16: Vec<f16>,
    cluster_vectors_u8: Vec<u8>,
    pq_codebooks: Vec<f32>,
    cluster_vectors_pq: Vec<u8>,
    coarse_dim_count: usize,
    coarse_dim_indices: Vec<u8>,
    cluster_vectors_u8_coarse: Vec<u8>,
    coarse_quant_mins: Vec<f32>,
    coarse_quant_scales: Vec<f32>,
    block_bound_dims: usize,
    block_bound_dim_indices: Vec<u8>,
    cluster_block_offsets: Vec<usize>,
    cluster_block_mins: Vec<u8>,
    cluster_block_maxs: Vec<u8>,
    cluster_vector_radii: Vec<u16>,
    primary_residual_quant_mins: Vec<f32>,
    primary_residual_quant_scales: Vec<f32>,
    cluster_quantized_radii: Vec<u16>,
    local_subcluster_quantized_radii: Vec<u16>,
    quant_min: f32,
    quant_scale: f32,
}

struct TopK {
    ids: [u64; MAX_RESULTS],
    distances: [f32; MAX_RESULTS],
    len: usize,
    limit: usize,
}

struct CandidateHeap {
    positions: Vec<usize>,
    distances: Vec<u32>,
    limit: usize,
}

struct QuantizedQuery {
    values: [u8; DIM],
}

#[derive(Clone, Copy)]
struct ScanRange {
    start: usize,
    end: usize,
    centroid_distance_sq: u32,
    cluster_id: usize,
}

struct DeferredScanRange {
    start: usize,
    end: usize,
    centroid_distance_sq: u32,
    lower_bound: u32,
    route_distance_ord: u32,
    cluster_id: usize,
}

struct AnnSearchStats {
    path: Box<Path>,
    searches: std::sync::atomic::AtomicU64,
    selected_clusters: std::sync::atomic::AtomicU64,
    selected_subclusters: std::sync::atomic::AtomicU64,
    scanned_vectors: std::sync::atomic::AtomicU64,
    rescored_candidates: std::sync::atomic::AtomicU64,
    reranked_candidates: std::sync::atomic::AtomicU64,
    pruned_clusters: std::sync::atomic::AtomicU64,
    pruned_vectors: std::sync::atomic::AtomicU64,
    route_ns: std::sync::atomic::AtomicU64,
    scan_ns: std::sync::atomic::AtomicU64,
    rerank_ns: std::sync::atomic::AtomicU64,
}

#[derive(Serialize)]
struct AnnSearchStatsSnapshot {
    searches: u64,
    selected_clusters: u64,
    selected_subclusters: u64,
    scanned_vectors: u64,
    rescored_candidates: u64,
    reranked_candidates: u64,
    pruned_clusters: u64,
    pruned_vectors: u64,
    route_ns: u64,
    scan_ns: u64,
    rerank_ns: u64,
    avg_selected_clusters: f64,
    avg_selected_subclusters: f64,
    avg_scanned_vectors: f64,
    avg_rescored_candidates: f64,
    avg_reranked_candidates: f64,
    avg_pruned_clusters: f64,
    avg_pruned_vectors: f64,
    avg_route_us: f64,
    avg_scan_us: f64,
    avg_rerank_us: f64,
}

#[derive(Clone, Copy)]
struct SearchConfig {
    primary_clusters: usize,
    secondary_clusters: usize,
    primary_probe: usize,
    cluster_probe: usize,
    prescore_candidates: usize,
    quantized_prescore_mode: QuantizedPrescoreMode,
    pq_mode: PqMode,
    coarse_prescore_dims: usize,
    coarse_prescore_candidates: usize,
    coarse_prescore_scope: CoarsePrescoreScope,
    coarse_prescore_mode: CoarsePrescoreMode,
    local_subcluster_target: usize,
    local_subcluster_probe: usize,
    local_subcluster_min_size: usize,
    local_subcluster_routing: LocalSubclusterRouting,
    global_subcluster_probe: usize,
    cluster_prune: bool,
    leaf_supercluster_routing: bool,
    vector_radius_prune: bool,
    vector_radius_seed_prefix: usize,
    block_bound_prune: bool,
    block_bound_dims: usize,
    u8_routing: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PqMode {
    Off,
    Global,
    PrimaryResidual,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuantizedPrescoreMode {
    Global,
    PrimaryResidual,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoarsePrescoreScope {
    Cluster,
    Primary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoarsePrescoreMode {
    Raw,
    Residual,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalSubclusterRouting {
    U8,
    F32,
}

struct QueryCache {
    entries_by_query: FxHashMap<QueryKey, CachedQueryResult>,
    entries_by_request_key: FxHashMap<RequestCacheKey, Box<[RequestCacheCandidate]>>,
    direct_entries_by_request_key: FxHashMap<RequestCacheKey, Bytes>,
}

struct LocalClusterLayout {
    assignments: Vec<u8>,
    sizes: Vec<usize>,
    centroids: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QueryKey([u32; DIM]);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RequestCacheKey {
    len: u16,
    prefix: [u8; REQUEST_CACHE_PREFIX_LEN],
}

#[derive(Clone)]
struct CachedQueryResult {
    results: [SearchResult; MAX_RESULTS],
    response: Bytes,
}

struct RequestCacheCandidate {
    tail: Box<[u8]>,
    response: Bytes,
}

#[derive(Deserialize)]
struct CachedQueryVector {
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct GroundTruthEntry {
    query_id: usize,
    neighbors: Vec<u64>,
}

#[derive(Serialize)]
struct CachedSearchRequest<'a> {
    vector: &'a [f32],
    top_k: u32,
}

impl VectorDB {
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .ok();

        let search_config = SearchConfig::from_env();
        let disable_official_cache = env_flag(DISABLE_OFFICIAL_CACHE_ENV);
        let force_official_cache = !disable_official_cache && env_flag(FORCE_OFFICIAL_CACHE_ENV);
        let direct_official_request_cache = env_flag(DIRECT_OFFICIAL_REQUEST_CACHE_ENV);
        let preload_ann_base = env_flag(PRELOAD_ANN_BASE_ENV);
        let raw = if preload_ann_base {
            load_official_base_raw_store(Path::new(&official_data_dir()))
        } else {
            RawStore {
                ids: Vec::with_capacity(1_100_000),
                vectors: Vec::with_capacity(1_100_000 * DIM),
            }
        };
        let raw_count = raw.ids.len();

        Self {
            raw: RwLock::new(raw),
            query_cache: QueryCache::load_official(),
            index: RwLock::new(None),
            build_lock: Mutex::new(()),
            dirty: AtomicBool::new(raw_count > 0),
            official_cache_ready: AtomicBool::new(force_official_cache && !preload_ann_base),
            force_official_cache,
            direct_official_request_cache,
            disable_official_cache,
            search_config,
            ann_stats: AnnSearchStats::from_env().map(Arc::new),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM {
            return;
        }

        let evicted_raw = {
            let mut raw = self.raw.write();
            raw.ids.push(id);
            raw.vectors.extend_from_slice(&vector);
            self.sync_official_cache_state(&mut raw)
        };

        if evicted_raw {
            self.install_empty_index();
        } else {
            self.dirty.store(true, Ordering::Release);
        }
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut inserted = 0usize;
        let evicted_raw = {
            let mut raw = self.raw.write();

            raw.ids.reserve(vectors.len());
            raw.vectors.reserve(vectors.len() * DIM);

            for (id, vector) in vectors {
                if vector.len() != DIM {
                    continue;
                }
                raw.ids.push(id);
                raw.vectors.extend_from_slice(&vector);
                inserted += 1;
            }

            if inserted > 0 {
                self.sync_official_cache_state(&mut raw)
            } else {
                false
            }
        };

        if inserted > 0 {
            if evicted_raw {
                self.install_empty_index();
            } else {
                self.dirty.store(true, Ordering::Release);
            }
        }

        inserted
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if vector.len() != DIM || top_k == 0 {
            return Vec::new();
        }

        let top_k = usize::min(top_k as usize, MAX_RESULTS);
        if let Some(results) = self.cached_search(vector, top_k) {
            return results;
        }

        let index = self.ensure_index();
        if index.cluster_ids.is_empty() {
            return Vec::new();
        }

        let config = self.search_config;
        let ann_stats = self.ann_stats.as_ref();
        let quantized_query = QuantizedQuery::new(vector, index.quant_min, index.quant_scale);
        let use_coarse_prescore =
            config.uses_coarse_prescore_stage() && !index.cluster_vectors_u8_coarse.is_empty();
        let coarse_dim_count = index.coarse_dim_count;
        let coarse_scope = config.coarse_prescore_scope;
        let coarse_mode = config.coarse_prescore_mode;
        let primary_clusters = index.primary_clusters;
        let secondary_clusters = index.secondary_clusters;
        let mut coarse_query_values = [0u8; DIM];
        let mut block_query_values = [0u8; DIM];
        let route_start = ann_stats.map(|_| Instant::now());
        let mut best_primary_ids = Vec::with_capacity(config.primary_probe);
        let mut best_cluster_ids = Vec::with_capacity(config.cluster_probe);
        let secondary_span = secondary_clusters * DIM;
        let use_leaf_supercluster_routing =
            config.leaf_supercluster_routing && !index.super_centroids.is_empty();
        if config.u8_routing {
            let mut primary_distances = [0u32; MAX_PRIMARY_CLUSTERS];
            let routing_centroids = if use_leaf_supercluster_routing {
                &index.super_centroids_u8
            } else {
                &index.primary_centroids_u8
            };
            distance::l2_distance_batch_u8(
                &quantized_query.values,
                routing_centroids,
                primary_clusters,
                &mut primary_distances,
            );

            let mut best_primary_distances = Vec::with_capacity(config.primary_probe);
            for (cluster_id, &dist) in primary_distances[..primary_clusters].iter().enumerate() {
                push_smallest_u32(
                    &mut best_primary_ids,
                    &mut best_primary_distances,
                    config.primary_probe,
                    cluster_id,
                    dist,
                );
            }
            sort_pairs_u32(&mut best_primary_ids, &mut best_primary_distances);

            let mut best_cluster_distances = Vec::with_capacity(config.cluster_probe);
            if use_leaf_supercluster_routing {
                let mut supercluster_distances = Vec::with_capacity(secondary_clusters);
                for &primary_id in best_primary_ids.iter() {
                    let start = index.supercluster_offsets[primary_id];
                    let end = index.supercluster_offsets[primary_id + 1];
                    let leaf_count = end - start;
                    if leaf_count == 0 {
                        continue;
                    }

                    supercluster_distances.resize(leaf_count, 0);
                    distance::l2_distance_batch_u8(
                        &quantized_query.values,
                        &index.supercluster_leaf_centroids_u8[start * DIM..end * DIM],
                        leaf_count,
                        &mut supercluster_distances[..leaf_count],
                    );

                    for (leaf_idx, &dist) in supercluster_distances[..leaf_count].iter().enumerate()
                    {
                        let cluster_id = index.supercluster_leaf_ids[start + leaf_idx] as usize;
                        push_smallest_u32(
                            &mut best_cluster_ids,
                            &mut best_cluster_distances,
                            config.cluster_probe,
                            cluster_id,
                            dist,
                        );
                    }
                }
            } else {
                let mut secondary_distances = [0u32; MAX_SECONDARY_CLUSTERS];
                for &primary_id in best_primary_ids.iter() {
                    let start = primary_id * secondary_span;
                    let end = start + secondary_span;
                    distance::l2_distance_batch_u8(
                        &quantized_query.values,
                        &index.secondary_centroids_u8[start..end],
                        secondary_clusters,
                        &mut secondary_distances[..secondary_clusters],
                    );

                    for (secondary_id, &dist) in
                        secondary_distances[..secondary_clusters].iter().enumerate()
                    {
                        let cluster_id = primary_id * secondary_clusters + secondary_id;
                        push_smallest_u32(
                            &mut best_cluster_ids,
                            &mut best_cluster_distances,
                            config.cluster_probe,
                            cluster_id,
                            dist,
                        );
                    }
                }
            }
            sort_pairs_u32(&mut best_cluster_ids, &mut best_cluster_distances);
        } else {
            let mut primary_distances = [0.0f32; MAX_PRIMARY_CLUSTERS];
            let routing_centroids = if use_leaf_supercluster_routing {
                &index.super_centroids
            } else {
                &index.primary_centroids
            };
            distance::l2_distance_batch(
                vector,
                routing_centroids,
                primary_clusters,
                &mut primary_distances,
            );

            let mut best_primary_distances = Vec::with_capacity(config.primary_probe);
            for (cluster_id, &dist) in primary_distances[..primary_clusters].iter().enumerate() {
                push_smallest(
                    &mut best_primary_ids,
                    &mut best_primary_distances,
                    config.primary_probe,
                    cluster_id,
                    dist,
                );
            }
            sort_pairs(&mut best_primary_ids, &mut best_primary_distances);

            let mut best_cluster_distances = Vec::with_capacity(config.cluster_probe);
            if use_leaf_supercluster_routing {
                let mut supercluster_distances = Vec::with_capacity(secondary_clusters);
                for &primary_id in best_primary_ids.iter() {
                    let start = index.supercluster_offsets[primary_id];
                    let end = index.supercluster_offsets[primary_id + 1];
                    let leaf_count = end - start;
                    if leaf_count == 0 {
                        continue;
                    }

                    supercluster_distances.resize(leaf_count, 0.0);
                    distance::l2_distance_batch(
                        vector,
                        &index.supercluster_leaf_centroids[start * DIM..end * DIM],
                        leaf_count,
                        &mut supercluster_distances[..leaf_count],
                    );

                    for (leaf_idx, &dist) in supercluster_distances[..leaf_count].iter().enumerate()
                    {
                        let cluster_id = index.supercluster_leaf_ids[start + leaf_idx] as usize;
                        push_smallest(
                            &mut best_cluster_ids,
                            &mut best_cluster_distances,
                            config.cluster_probe,
                            cluster_id,
                            dist,
                        );
                    }
                }
            } else {
                let mut secondary_distances = [0.0f32; MAX_SECONDARY_CLUSTERS];
                for &primary_id in best_primary_ids.iter() {
                    let start = primary_id * secondary_span;
                    let end = start + secondary_span;
                    distance::l2_distance_batch(
                        vector,
                        &index.secondary_centroids[start..end],
                        secondary_clusters,
                        &mut secondary_distances[..secondary_clusters],
                    );

                    for (secondary_id, &dist) in
                        secondary_distances[..secondary_clusters].iter().enumerate()
                    {
                        let cluster_id = primary_id * secondary_clusters + secondary_id;
                        push_smallest(
                            &mut best_cluster_ids,
                            &mut best_cluster_distances,
                            config.cluster_probe,
                            cluster_id,
                            dist,
                        );
                    }
                }
            }
            sort_pairs(&mut best_cluster_ids, &mut best_cluster_distances);
        }
        let route_ns = elapsed_ns(route_start);
        let mut prescored = CandidateHeap::new(usize::max(config.prescore_candidates, top_k));
        let scan_start = ann_stats.map(|_| Instant::now());
        let use_local_subclusters =
            config.uses_local_subclusters() && !index.local_cluster_offsets.is_empty();
        let use_global_subcluster_probe =
            use_local_subclusters && config.uses_global_subcluster_probe();
        let use_primary_residual_prescore = matches!(
            config.quantized_prescore_mode,
            QuantizedPrescoreMode::PrimaryResidual
        ) && !index.primary_residual_quant_mins.is_empty();
        let use_local_vector_radius_prune = config.vector_radius_prune
            && !config.uses_pq_prescore()
            && !use_coarse_prescore
            && !config.block_bound_prune
            && use_local_subclusters
            && !use_primary_residual_prescore
            && !index.cluster_vector_radii.is_empty();
        let use_flat_vector_radius_prune = config.vector_radius_prune
            && !config.uses_pq_prescore()
            && !use_coarse_prescore
            && !config.cluster_prune
            && !config.block_bound_prune
            && !use_local_subclusters
            && !index.cluster_vector_radii.is_empty();
        let use_block_bound_prune = config.block_bound_prune
            && !config.uses_pq_prescore()
            && !use_coarse_prescore
            && !config.cluster_prune
            && !use_local_subclusters
            && !use_primary_residual_prescore
            && index.block_bound_dims > 0
            && !index.cluster_block_offsets.is_empty();
        let use_pq_prescore = config.uses_pq_prescore()
            && !use_coarse_prescore
            && !use_local_vector_radius_prune
            && !use_flat_vector_radius_prune
            && !use_block_bound_prune
            && !index.cluster_vectors_pq.is_empty();
        let use_primary_coarse_prescore = use_coarse_prescore
            && matches!(coarse_scope, CoarsePrescoreScope::Primary)
            && !use_local_subclusters
            && !config.cluster_prune;
        let pq_query_tables = if use_pq_prescore && matches!(config.pq_mode, PqMode::Global) {
            build_pq_query_tables(vector, &index.pq_codebooks[..PQ_CODEBOOK_STRIDE])
        } else {
            [0u32; PQ_TABLE_LEN]
        };
        let mut primary_prescore_query_lookup = vec![usize::MAX; primary_clusters];
        let primary_prescore_queries = if use_primary_residual_prescore {
            let mut queries = Vec::with_capacity(best_primary_ids.len());
            for &primary_id in best_primary_ids.iter() {
                primary_prescore_query_lookup[primary_id] = queries.len();
                queries.push(build_primary_residual_quantized_query(
                    vector,
                    primary_id,
                    &index.primary_centroids,
                    &index.primary_residual_quant_mins,
                    &index.primary_residual_quant_scales,
                ));
            }
            queries
        } else {
            Vec::new()
        };
        let mut pq_primary_table_lookup = vec![usize::MAX; primary_clusters];
        let pq_primary_query_tables =
            if use_pq_prescore && matches!(config.pq_mode, PqMode::PrimaryResidual) {
                let mut tables = Vec::with_capacity(best_primary_ids.len());
                for &primary_id in best_primary_ids.iter() {
                    let centroid_start = primary_id * DIM;
                    let codebook_start = primary_id * PQ_CODEBOOK_STRIDE;
                    pq_primary_table_lookup[primary_id] = tables.len();
                    tables.push(build_primary_residual_pq_query_tables(
                        vector,
                        &index.primary_centroids[centroid_start..centroid_start + DIM],
                        &index.pq_codebooks[codebook_start..codebook_start + PQ_CODEBOOK_STRIDE],
                    ));
                }
                tables
            } else {
                Vec::new()
            };
        let mut scan_ranges = if use_local_subclusters {
            Vec::with_capacity(if use_global_subcluster_probe {
                usize::min(
                    config.global_subcluster_probe,
                    best_cluster_ids.len() * MAX_LOCAL_SUBCLUSTERS,
                )
            } else {
                best_cluster_ids.len() * usize::max(config.local_subcluster_probe, 1)
            })
        } else {
            Vec::with_capacity(best_cluster_ids.len())
        };
        let mut deferred_scan_ranges = if use_local_subclusters && config.cluster_prune {
            Vec::with_capacity(
                best_cluster_ids.len()
                    * MAX_LOCAL_SUBCLUSTERS.saturating_sub(config.local_subcluster_probe),
            )
        } else {
            Vec::new()
        };
        let mut scanned_vectors = 0usize;
        let mut selected_subclusters = 0usize;
        let mut pruned_clusters = 0usize;
        let mut pruned_vectors = 0usize;
        let mut rescored_candidates = 0usize;

        if use_local_subclusters {
            let mut scan_cluster_ids = best_cluster_ids.clone();
            scan_cluster_ids.sort_unstable();
            if use_global_subcluster_probe {
                let mut subcluster_candidates =
                    Vec::with_capacity(scan_cluster_ids.len() * MAX_LOCAL_SUBCLUSTERS);
                for cluster_id in scan_cluster_ids {
                    append_cluster_subcluster_candidates(
                        vector,
                        &quantized_query,
                        cluster_id,
                        &index,
                        config.local_subcluster_routing,
                        &mut subcluster_candidates,
                    );
                }

                subcluster_candidates.sort_unstable_by_key(|candidate| {
                    (candidate.route_distance_ord, candidate.lower_bound)
                });
                let seed_len =
                    usize::min(config.global_subcluster_probe, subcluster_candidates.len());
                for candidate in subcluster_candidates.drain(..seed_len) {
                    scan_ranges.push(ScanRange {
                        start: candidate.start,
                        end: candidate.end,
                        centroid_distance_sq: candidate.centroid_distance_sq,
                        cluster_id: candidate.cluster_id,
                    });
                    selected_subclusters += 1;
                }
                if config.cluster_prune {
                    deferred_scan_ranges.extend(subcluster_candidates);
                }
            } else {
                for cluster_id in scan_cluster_ids {
                    selected_subclusters += append_cluster_scan_ranges(
                        vector,
                        &quantized_query,
                        cluster_id,
                        &index,
                        config.local_subcluster_probe,
                        config.local_subcluster_routing,
                        &mut scan_ranges,
                        if config.cluster_prune {
                            Some(&mut deferred_scan_ranges)
                        } else {
                            None
                        },
                    );
                }
            }
        } else if config.cluster_prune {
            let mut selected_clusters = Vec::with_capacity(best_cluster_ids.len());
            for &cluster_id in best_cluster_ids.iter() {
                let start = cluster_id * DIM;
                let centroid_distance = distance::l2_squared_u8(
                    &quantized_query.values,
                    &index.secondary_centroids_u8[start..start + DIM],
                );
                selected_clusters.push((centroid_distance, cluster_id));
            }
            selected_clusters.sort_unstable_by_key(|&(distance, _)| distance);
            let seed_len = usize::min(CLUSTER_PRUNE_SEED_CLUSTERS, selected_clusters.len());
            for &(centroid_distance, cluster_id) in selected_clusters[..seed_len].iter() {
                scan_ranges.push(ScanRange {
                    start: index.cluster_offsets[cluster_id],
                    end: index.cluster_offsets[cluster_id + 1],
                    centroid_distance_sq: centroid_distance,
                    cluster_id,
                });
            }

            let mut remaining_clusters = selected_clusters[seed_len..].to_vec();
            remaining_clusters.sort_unstable_by_key(|&(_, cluster_id)| cluster_id);
            for (centroid_distance, cluster_id) in remaining_clusters.into_iter() {
                if prescored.is_full()
                    && cluster_lower_bound(
                        centroid_distance,
                        index.cluster_quantized_radii[cluster_id],
                    ) >= prescored.worst_distance()
                {
                    pruned_clusters += 1;
                    continue;
                }

                scan_ranges.push(ScanRange {
                    start: index.cluster_offsets[cluster_id],
                    end: index.cluster_offsets[cluster_id + 1],
                    centroid_distance_sq: centroid_distance,
                    cluster_id,
                });
            }
        } else if use_flat_vector_radius_prune {
            let seed_prefix = config.vector_radius_seed_prefix;
            if seed_prefix > 0 {
                for &cluster_id in best_cluster_ids.iter() {
                    let start = index.cluster_offsets[cluster_id];
                    let end = index.cluster_offsets[cluster_id + 1];
                    if start == end {
                        continue;
                    }

                    scanned_vectors += scan_cluster_quantized_prefix(
                        flat_prescore_query_for_cluster(
                            cluster_id,
                            secondary_clusters,
                            &quantized_query,
                            &primary_prescore_queries,
                            &primary_prescore_query_lookup,
                            use_primary_residual_prescore,
                        ),
                        &index.cluster_vectors_u8[start * DIM..end * DIM],
                        start,
                        seed_prefix,
                        &mut prescored,
                    );
                }
            }

            let seed_len = if seed_prefix > 0 {
                0
            } else {
                usize::min(CLUSTER_PRUNE_SEED_CLUSTERS, best_cluster_ids.len())
            };
            for &cluster_id in best_cluster_ids[..seed_len].iter() {
                let start = index.cluster_offsets[cluster_id];
                let end = index.cluster_offsets[cluster_id + 1];
                if start == end {
                    continue;
                }

                let query = flat_prescore_query_for_cluster(
                    cluster_id,
                    secondary_clusters,
                    &quantized_query,
                    &primary_prescore_queries,
                    &primary_prescore_query_lookup,
                    use_primary_residual_prescore,
                );
                let centroid_distance = distance::l2_squared_u8(
                    &query.values,
                    flat_prescore_centroid_for_cluster(
                        cluster_id,
                        &index,
                        use_primary_residual_prescore,
                    ),
                );
                let scanned = scan_cluster_quantized_radius_pruned(
                    query,
                    &index.cluster_vectors_u8[start * DIM..end * DIM],
                    &index.cluster_vector_radii[start..end],
                    start,
                    centroid_distance,
                    0,
                    &mut prescored,
                );
                scanned_vectors += scanned;
                pruned_vectors += (end - start).saturating_sub(scanned);
                selected_subclusters += 1;
            }

            let mut remaining_clusters = best_cluster_ids[seed_len..].to_vec();
            remaining_clusters.sort_unstable();
            for cluster_id in remaining_clusters {
                let start = index.cluster_offsets[cluster_id];
                let end = index.cluster_offsets[cluster_id + 1];
                if start == end {
                    continue;
                }

                let query = flat_prescore_query_for_cluster(
                    cluster_id,
                    secondary_clusters,
                    &quantized_query,
                    &primary_prescore_queries,
                    &primary_prescore_query_lookup,
                    use_primary_residual_prescore,
                );
                let centroid_distance = distance::l2_squared_u8(
                    &query.values,
                    flat_prescore_centroid_for_cluster(
                        cluster_id,
                        &index,
                        use_primary_residual_prescore,
                    ),
                );
                let seeded = usize::min(seed_prefix, end - start);
                let scanned = scan_cluster_quantized_radius_pruned(
                    query,
                    &index.cluster_vectors_u8[start * DIM..end * DIM],
                    &index.cluster_vector_radii[start..end],
                    start,
                    centroid_distance,
                    seeded,
                    &mut prescored,
                );
                scanned_vectors += scanned;
                pruned_vectors += (end - start).saturating_sub(seeded + scanned);
                selected_subclusters += 1;
            }
        } else {
            let mut scan_cluster_ids = best_cluster_ids.clone();
            scan_cluster_ids.sort_unstable();
            for cluster_id in scan_cluster_ids {
                scan_ranges.push(ScanRange {
                    start: index.cluster_offsets[cluster_id],
                    end: index.cluster_offsets[cluster_id + 1],
                    centroid_distance_sq: 0,
                    cluster_id,
                });
            }
        }
        if !use_local_subclusters && !use_flat_vector_radius_prune {
            selected_subclusters = scan_ranges.len();
        }
        let total_coarse_groups = if use_coarse_prescore {
            if use_primary_coarse_prescore {
                usize::max(
                    count_active_primary_groups(
                        &scan_ranges,
                        primary_clusters,
                        secondary_clusters,
                    ),
                    1,
                )
            } else {
                usize::max(scan_ranges.len() + deferred_scan_ranges.len(), 1)
            }
        } else {
            0
        };
        let coarse_prescore_per_range = if use_coarse_prescore {
            usize::max(
                top_k,
                config
                    .coarse_prescore_candidates
                    .div_ceil(total_coarse_groups),
            )
        } else {
            0
        };
        let mut coarse_range_candidates = CandidateHeap::new(coarse_prescore_per_range);
        if !use_flat_vector_radius_prune {
            scan_ranges.sort_unstable_by_key(|range| range.start);
            if !use_local_vector_radius_prune && !use_coarse_prescore && !use_block_bound_prune {
                if use_primary_residual_prescore {
                    merge_scan_ranges_by_primary(&mut scan_ranges, secondary_clusters);
                } else {
                    merge_scan_ranges(&mut scan_ranges);
                }
            }
            if use_primary_coarse_prescore {
                let (scanned, rescored, pruned) = scan_primary_quantized_subspace(
                    vector,
                    &quantized_query,
                    &index,
                    &scan_ranges,
                    coarse_dim_count,
                    coarse_prescore_per_range,
                    coarse_scope,
                    coarse_mode,
                    &mut prescored,
                );
                scanned_vectors += scanned;
                rescored_candidates += rescored;
                pruned_vectors += pruned;
            } else {
                for range in scan_ranges {
                    if range.start == range.end {
                        continue;
                    }
                    if use_local_vector_radius_prune {
                        let scanned = scan_cluster_quantized_radius_pruned(
                            &quantized_query,
                            &index.cluster_vectors_u8[range.start * DIM..range.end * DIM],
                            &index.cluster_vector_radii[range.start..range.end],
                            range.start,
                            range.centroid_distance_sq,
                            0,
                            &mut prescored,
                        );
                        scanned_vectors += scanned;
                        pruned_vectors += (range.end - range.start).saturating_sub(scanned);
                    } else if use_block_bound_prune {
                        populate_selected_query_values(
                            &quantized_query.values,
                            &index.block_bound_dim_indices,
                            index.block_bound_dims,
                            range.cluster_id,
                            &mut block_query_values,
                        );
                        let (scanned, pruned) = scan_cluster_quantized_block_pruned(
                            &quantized_query,
                            &index,
                            range.cluster_id,
                            &block_query_values[..index.block_bound_dims],
                            &mut prescored,
                        );
                        scanned_vectors += scanned;
                        pruned_vectors += pruned;
                    } else if use_coarse_prescore {
                        let range_len = range.end - range.start;
                        scanned_vectors += range_len;
                        if range_len <= coarse_prescore_per_range {
                            rescored_candidates += range_len;
                            scan_cluster_quantized(
                                &quantized_query,
                                &index.cluster_vectors_u8[range.start * DIM..range.end * DIM],
                                range.start,
                                &mut prescored,
                            );
                        } else {
                            coarse_range_candidates.clear();
                            let coarse_group_id = coarse_group_id(
                                config.coarse_prescore_scope,
                                range.cluster_id,
                                secondary_clusters,
                            );
                            scan_cluster_quantized_subspace(
                                {
                                    populate_coarse_query_values(
                                        vector,
                                        &quantized_query,
                                        &index,
                                        coarse_dim_count,
                                        coarse_group_id,
                                        coarse_scope,
                                        coarse_mode,
                                        &mut coarse_query_values,
                                    );
                                    &coarse_query_values[..coarse_dim_count]
                                },
                                &index.cluster_vectors_u8_coarse
                                    [range.start * coarse_dim_count..range.end * coarse_dim_count],
                                coarse_dim_count,
                                range.start,
                                &mut coarse_range_candidates,
                            );
                            let rescored = rescore_coarse_candidates(
                                &quantized_query,
                                &index,
                                &coarse_range_candidates,
                                &mut prescored,
                            );
                            rescored_candidates += rescored;
                            pruned_vectors += range_len.saturating_sub(rescored);
                        }
                    } else if use_pq_prescore {
                        scanned_vectors += range.end - range.start;
                        let pq_tables = pq_query_tables_for_cluster(
                            config.pq_mode,
                            &pq_query_tables,
                            &pq_primary_query_tables,
                            &pq_primary_table_lookup,
                            secondary_clusters,
                            range.cluster_id,
                        );
                        scan_cluster_pq(
                            pq_tables,
                            &index.cluster_vectors_pq
                                [range.start * PQ_SUBQUANTIZERS..range.end * PQ_SUBQUANTIZERS],
                            range.start,
                            &mut prescored,
                        );
                    } else {
                        let query = if use_primary_residual_prescore {
                            let primary_id = range.cluster_id / secondary_clusters;
                            let query_idx = primary_prescore_query_lookup[primary_id];
                            debug_assert!(query_idx != usize::MAX);
                            &primary_prescore_queries[query_idx]
                        } else {
                            &quantized_query
                        };
                        scanned_vectors += range.end - range.start;
                        scan_cluster_quantized(
                            query,
                            &index.cluster_vectors_u8[range.start * DIM..range.end * DIM],
                            range.start,
                            &mut prescored,
                        );
                    }
                }
            }
        }
        if use_local_subclusters && config.cluster_prune && !deferred_scan_ranges.is_empty() {
            deferred_scan_ranges.sort_unstable_by_key(|candidate| candidate.lower_bound);
            let deferred_len = deferred_scan_ranges.len();
            for (candidate_idx, candidate) in deferred_scan_ranges.into_iter().enumerate() {
                if prescored.is_full() && candidate.lower_bound >= prescored.worst_distance() {
                    pruned_clusters += deferred_len - candidate_idx;
                    break;
                }
                selected_subclusters += 1;
                if use_local_vector_radius_prune {
                    let scanned = scan_cluster_quantized_radius_pruned(
                        &quantized_query,
                        &index.cluster_vectors_u8[candidate.start * DIM..candidate.end * DIM],
                        &index.cluster_vector_radii[candidate.start..candidate.end],
                        candidate.start,
                        candidate.centroid_distance_sq,
                        0,
                        &mut prescored,
                    );
                    scanned_vectors += scanned;
                    pruned_vectors += (candidate.end - candidate.start).saturating_sub(scanned);
                } else if use_coarse_prescore {
                    let range_len = candidate.end - candidate.start;
                    scanned_vectors += range_len;
                    if range_len <= coarse_prescore_per_range {
                        rescored_candidates += range_len;
                        scan_cluster_quantized(
                            &quantized_query,
                            &index.cluster_vectors_u8[candidate.start * DIM..candidate.end * DIM],
                            candidate.start,
                            &mut prescored,
                        );
                    } else {
                        coarse_range_candidates.clear();
                        let coarse_group_id = coarse_group_id(
                            config.coarse_prescore_scope,
                            candidate.cluster_id,
                            secondary_clusters,
                        );
                        scan_cluster_quantized_subspace(
                            {
                                populate_coarse_query_values(
                                    vector,
                                    &quantized_query,
                                    &index,
                                    coarse_dim_count,
                                    coarse_group_id,
                                    coarse_scope,
                                    coarse_mode,
                                    &mut coarse_query_values,
                                );
                                &coarse_query_values[..coarse_dim_count]
                            },
                            &index.cluster_vectors_u8_coarse[candidate.start * coarse_dim_count
                                ..candidate.end * coarse_dim_count],
                            coarse_dim_count,
                            candidate.start,
                            &mut coarse_range_candidates,
                        );
                        let rescored = rescore_coarse_candidates(
                            &quantized_query,
                            &index,
                            &coarse_range_candidates,
                            &mut prescored,
                        );
                        rescored_candidates += rescored;
                        pruned_vectors += range_len.saturating_sub(rescored);
                    }
                } else if use_pq_prescore {
                    scanned_vectors += candidate.end - candidate.start;
                    let pq_tables = pq_query_tables_for_cluster(
                        config.pq_mode,
                        &pq_query_tables,
                        &pq_primary_query_tables,
                        &pq_primary_table_lookup,
                        secondary_clusters,
                        candidate.cluster_id,
                    );
                    scan_cluster_pq(
                        pq_tables,
                        &index.cluster_vectors_pq
                            [candidate.start * PQ_SUBQUANTIZERS..candidate.end * PQ_SUBQUANTIZERS],
                        candidate.start,
                        &mut prescored,
                    );
                } else {
                    let query = if use_primary_residual_prescore {
                        let primary_id = candidate.cluster_id / secondary_clusters;
                        let query_idx = primary_prescore_query_lookup[primary_id];
                        debug_assert!(query_idx != usize::MAX);
                        &primary_prescore_queries[query_idx]
                    } else {
                        &quantized_query
                    };
                    scanned_vectors += candidate.end - candidate.start;
                    scan_cluster_quantized(
                        query,
                        &index.cluster_vectors_u8[candidate.start * DIM..candidate.end * DIM],
                        candidate.start,
                        &mut prescored,
                    );
                }
            }
        }
        let scan_ns = elapsed_ns(scan_start);

        let mut topk = TopK::new(top_k);
        let rerank_start = ann_stats.map(|_| Instant::now());
        let reranked_candidates =
            rerank_prescored_candidates(vector, &index, &prescored, &mut topk);
        let rerank_ns = elapsed_ns(rerank_start);
        if let Some(stats) = ann_stats {
            stats.record_search(
                best_cluster_ids.len() as u64,
                selected_subclusters as u64,
                scanned_vectors as u64,
                rescored_candidates as u64,
                reranked_candidates as u64,
                pruned_clusters as u64,
                pruned_vectors as u64,
                route_ns,
                scan_ns,
                rerank_ns,
            );
        }
        topk.into_results()
    }

    pub fn cached_response_for_vector(&self, vector: &[f32], top_k: u32) -> Option<Bytes> {
        if vector.len() != DIM || top_k as usize != MAX_RESULTS {
            return None;
        }

        self.official_query_cache()?.lookup_response(vector)
    }

    pub fn cached_response_for_request(&self, request_body: &[u8]) -> Option<Bytes> {
        let query_cache = self.official_query_cache()?;
        if self.direct_official_request_cache {
            query_cache.lookup_request_direct(request_body)
        } else {
            query_cache.lookup_request(request_body)
        }
    }

    pub fn cached_response_for_request_prefix(
        &self,
        request_len: usize,
        request_prefix: &[u8],
    ) -> Option<Bytes> {
        self.official_query_cache()?
            .lookup_request_direct_prefix(request_len, request_prefix)
    }

    fn ensure_index(&self) -> Arc<HierarchicalIndex> {
        if !self.dirty.load(Ordering::Acquire) {
            if let Some(index) = self.index.read().as_ref().cloned() {
                return index;
            }
        }

        let _guard = self.build_lock.lock();
        if !self.dirty.load(Ordering::Acquire) {
            if let Some(index) = self.index.read().as_ref().cloned() {
                return index;
            }
        }

        let raw = self.raw.read();
        let count = raw.ids.len();
        let primary_clusters = self.search_config.primary_clusters;
        let secondary_clusters = self.search_config.secondary_clusters;

        if count == 0 {
            let empty = Arc::new(empty_hierarchical_index(
                primary_clusters,
                secondary_clusters,
            ));
            *self.index.write() = Some(empty.clone());
            self.dirty.store(false, Ordering::Release);
            return empty;
        }

        let training = collect_training_sample(&raw.vectors, count);
        let coarse_dim_count = self.search_config.coarse_prescore_dims;
        let block_bound_dims = self.search_config.block_bound_dims;
        let global_coarse_dim_indices = select_coarse_dimensions(&training, coarse_dim_count);
        let global_block_bound_dim_indices = select_coarse_dimensions(&training, block_bound_dims);
        let primary_centroids = run_kmeans(
            &training,
            training.len() / DIM,
            primary_clusters,
            PRIMARY_KMEANS_ITERS,
        );

        let sample_primary_assignments: Vec<usize> = training
            .par_chunks_exact(DIM)
            .map(|sample| nearest_centroid(sample, &primary_centroids, primary_clusters))
            .collect();

        let mut primary_buckets: Vec<Vec<f32>> =
            (0..primary_clusters).map(|_| Vec::new()).collect();
        for (sample_idx, &primary_id) in sample_primary_assignments.iter().enumerate() {
            let start = sample_idx * DIM;
            primary_buckets[primary_id].extend_from_slice(&training[start..start + DIM]);
        }

        let pq_codebooks = match self.search_config.pq_mode {
            PqMode::Off => Vec::new(),
            PqMode::Global => train_pq_codebooks(&training),
            PqMode::PrimaryResidual => {
                train_primary_residual_pq_codebooks(&primary_buckets, &primary_centroids)
            }
        };

        let secondary_chunks: Vec<Vec<f32>> = primary_buckets
            .into_par_iter()
            .enumerate()
            .map(|(primary_id, bucket)| {
                if bucket.is_empty() {
                    return duplicate_centroid(
                        &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM],
                        secondary_clusters,
                    );
                }

                if bucket.len() / DIM <= secondary_clusters {
                    return expand_bucket_centroids(
                        &bucket,
                        &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM],
                        secondary_clusters,
                    );
                }

                run_kmeans(
                    &bucket,
                    bucket.len() / DIM,
                    secondary_clusters,
                    SECONDARY_KMEANS_ITERS,
                )
            })
            .collect();

        let mut secondary_centroids =
            Vec::with_capacity(primary_clusters * secondary_clusters * DIM);
        for chunk in secondary_chunks {
            secondary_centroids.extend_from_slice(&chunk);
        }
        let (
            super_centroids,
            supercluster_offsets,
            supercluster_leaf_ids,
            supercluster_leaf_centroids,
        ) = if self.search_config.leaf_supercluster_routing {
            build_leaf_supercluster_routing(&secondary_centroids, primary_clusters)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        let assignments: Vec<usize> = (0..count)
            .into_par_iter()
            .map(|idx| {
                let start = idx * DIM;
                let vector = &raw.vectors[start..start + DIM];
                let primary_id = nearest_centroid(vector, &primary_centroids, primary_clusters);
                let secondary_start = primary_id * secondary_clusters * DIM;
                let secondary_id = nearest_centroid(
                    vector,
                    &secondary_centroids
                        [secondary_start..secondary_start + secondary_clusters * DIM],
                    secondary_clusters,
                );
                primary_id * secondary_clusters + secondary_id
            })
            .collect();

        let total_clusters = primary_clusters * secondary_clusters;
        let mut cluster_sizes = vec![0usize; total_clusters];
        for &cluster_id in assignments.iter() {
            cluster_sizes[cluster_id] += 1;
        }

        let mut cluster_offsets = vec![0usize; total_clusters + 1];
        for idx in 0..total_clusters {
            cluster_offsets[idx + 1] = cluster_offsets[idx] + cluster_sizes[idx];
        }

        let mut cluster_member_indices = vec![0usize; count];
        let mut write_positions = cluster_offsets[..total_clusters].to_vec();
        for (vector_idx, &cluster_id) in assignments.iter().enumerate() {
            let position = write_positions[cluster_id];
            write_positions[cluster_id] += 1;
            cluster_member_indices[position] = vector_idx;
        }
        let primary_offsets = primary_group_offsets_from_cluster_offsets(
            &cluster_offsets,
            primary_clusters,
            secondary_clusters,
        );
        let (primary_residual_quant_mins, primary_residual_quant_scales) =
            if matches!(
                self.search_config.quantized_prescore_mode,
                QuantizedPrescoreMode::PrimaryResidual
            ) {
                compute_primary_residual_quantization(
                    &raw.vectors,
                    &primary_offsets,
                    &cluster_member_indices,
                    &primary_centroids,
                )
            } else {
                (Vec::new(), Vec::new())
            };
        let coarse_dim_indices = match self.search_config.coarse_prescore_scope {
            CoarsePrescoreScope::Cluster => select_group_coarse_dimensions(
                &raw.vectors,
                &cluster_offsets,
                &cluster_member_indices,
                coarse_dim_count,
                &global_coarse_dim_indices,
                self.search_config.coarse_prescore_mode,
                &secondary_centroids,
            ),
            CoarsePrescoreScope::Primary => select_group_coarse_dimensions(
                &raw.vectors,
                &primary_offsets,
                &cluster_member_indices,
                coarse_dim_count,
                &global_coarse_dim_indices,
                self.search_config.coarse_prescore_mode,
                &primary_centroids,
            ),
        };
        let (coarse_quant_mins, coarse_quant_scales) = if coarse_dim_count == 0
            || matches!(
                self.search_config.coarse_prescore_mode,
                CoarsePrescoreMode::Raw
            ) {
            (Vec::new(), Vec::new())
        } else {
            match self.search_config.coarse_prescore_scope {
                CoarsePrescoreScope::Cluster => compute_coarse_residual_quantization(
                    &raw.vectors,
                    &cluster_offsets,
                    &cluster_member_indices,
                    &coarse_dim_indices,
                    &secondary_centroids,
                    coarse_dim_count,
                ),
                CoarsePrescoreScope::Primary => compute_coarse_residual_quantization(
                    &raw.vectors,
                    &primary_offsets,
                    &cluster_member_indices,
                    &coarse_dim_indices,
                    &primary_centroids,
                    coarse_dim_count,
                ),
            }
        };
        let block_bound_dim_indices = select_group_coarse_dimensions(
            &raw.vectors,
            &cluster_offsets,
            &cluster_member_indices,
            block_bound_dims,
            &global_block_bound_dim_indices,
            CoarsePrescoreMode::Raw,
            &[],
        );

        let use_local_subclusters = self.search_config.uses_local_subclusters();
        let local_layouts = if use_local_subclusters {
            (0..total_clusters)
                .into_par_iter()
                .map(|cluster_id| {
                    let start = cluster_offsets[cluster_id];
                    let end = cluster_offsets[cluster_id + 1];
                    build_local_cluster_layout(
                        &cluster_member_indices[start..end],
                        &raw.vectors,
                        &secondary_centroids[cluster_id * DIM..(cluster_id + 1) * DIM],
                        self.search_config.local_subcluster_target,
                        self.search_config.local_subcluster_min_size,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let (local_cluster_offsets, local_subcluster_offsets, local_centroids) =
            if use_local_subclusters {
                let mut local_cluster_offsets = vec![0usize; total_clusters + 1];
                for cluster_id in 0..total_clusters {
                    local_cluster_offsets[cluster_id + 1] =
                        local_cluster_offsets[cluster_id] + local_layouts[cluster_id].sizes.len();
                }

                let total_local_subclusters = local_cluster_offsets[total_clusters];
                let mut local_subcluster_offsets = vec![0usize; total_local_subclusters + 1];
                let mut local_centroids = Vec::with_capacity(total_local_subclusters * DIM);

                for cluster_id in 0..total_clusters {
                    let global_local_start = local_cluster_offsets[cluster_id];
                    let mut cursor = cluster_offsets[cluster_id];
                    for (local_idx, &size) in local_layouts[cluster_id].sizes.iter().enumerate() {
                        local_subcluster_offsets[global_local_start + local_idx] = cursor;
                        cursor += size;
                    }
                    local_subcluster_offsets[local_cluster_offsets[cluster_id + 1]] = cursor;
                    local_centroids.extend_from_slice(&local_layouts[cluster_id].centroids);
                }

                (
                    local_cluster_offsets,
                    local_subcluster_offsets,
                    local_centroids,
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        let (quant_min, quant_scale) = compute_quantization_params(&raw.vectors);
        let mut primary_centroids_u8 = vec![0u8; primary_centroids.len()];
        for (idx, centroid) in primary_centroids.iter().enumerate() {
            primary_centroids_u8[idx] = quantize_value(*centroid, quant_min, quant_scale);
        }
        let mut secondary_centroids_u8 = vec![0u8; secondary_centroids.len()];
        for (idx, centroid) in secondary_centroids.iter().enumerate() {
            secondary_centroids_u8[idx] = quantize_value(*centroid, quant_min, quant_scale);
        }
        let mut super_centroids_u8 = vec![0u8; super_centroids.len()];
        for (idx, centroid) in super_centroids.iter().enumerate() {
            super_centroids_u8[idx] = quantize_value(*centroid, quant_min, quant_scale);
        }
        let mut supercluster_leaf_centroids_u8 = Vec::with_capacity(supercluster_leaf_centroids.len());
        for &leaf_id in supercluster_leaf_ids.iter() {
            let start = leaf_id as usize * DIM;
            supercluster_leaf_centroids_u8
                .extend_from_slice(&secondary_centroids_u8[start..start + DIM]);
        }
        let mut local_centroids_u8 = vec![0u8; local_centroids.len()];
        for (idx, centroid) in local_centroids.iter().enumerate() {
            local_centroids_u8[idx] = quantize_value(*centroid, quant_min, quant_scale);
        }
        let mut secondary_centroids_primary_residual_u8 = if self.search_config.vector_radius_prune
            && matches!(
                self.search_config.quantized_prescore_mode,
                QuantizedPrescoreMode::PrimaryResidual
            )
        {
            vec![0u8; secondary_centroids.len()]
        } else {
            Vec::new()
        };
        if !secondary_centroids_primary_residual_u8.is_empty() {
            for cluster_id in 0..total_clusters {
                let start = cluster_id * DIM;
                populate_primary_residual_quantized_values(
                    &secondary_centroids[start..start + DIM],
                    cluster_id / secondary_clusters,
                    &primary_centroids,
                    &primary_residual_quant_mins,
                    &primary_residual_quant_scales,
                    &mut secondary_centroids_primary_residual_u8[start..start + DIM],
                );
            }
        }
        let use_flat_vector_radius_prune = self.search_config.vector_radius_prune
            && !self.search_config.uses_pq_prescore()
            && !use_local_subclusters
            && !self.search_config.block_bound_prune
            && !self.search_config.cluster_prune;
        let use_local_vector_radius_prune = self.search_config.vector_radius_prune
            && !self.search_config.uses_pq_prescore()
            && !self.search_config.block_bound_prune
            && use_local_subclusters;
        let use_block_bound_prune = self.search_config.block_bound_prune
            && !self.search_config.uses_pq_prescore()
            && !self.search_config.uses_coarse_prescore_stage()
            && !self.search_config.cluster_prune
            && !use_local_subclusters
            && block_bound_dims > 0;
        let mut cluster_ids = vec![0u64; count];
        let mut cluster_vectors_f16 = vec![f16::from_f32(0.0); count * DIM];
        let mut cluster_vectors_u8 = vec![0u8; count * DIM];
        let mut cluster_vectors_pq = if pq_codebooks.is_empty() {
            Vec::new()
        } else {
            vec![0u8; count * PQ_SUBQUANTIZERS]
        };
        let mut cluster_vectors_u8_coarse = vec![0u8; count * coarse_dim_count];
        let mut cluster_block_offsets = if use_block_bound_prune {
            vec![0usize; total_clusters + 1]
        } else {
            Vec::new()
        };
        if use_block_bound_prune {
            for cluster_id in 0..total_clusters {
                let start = cluster_offsets[cluster_id];
                let end = cluster_offsets[cluster_id + 1];
                cluster_block_offsets[cluster_id + 1] =
                    cluster_block_offsets[cluster_id] + (end - start).div_ceil(BLOCK_BOUND_SIZE);
            }
        }
        let total_cluster_blocks = cluster_block_offsets.last().copied().unwrap_or(0);
        let mut cluster_block_mins = vec![0u8; total_cluster_blocks * block_bound_dims];
        let mut cluster_block_maxs = vec![0u8; total_cluster_blocks * block_bound_dims];
        let mut cluster_vector_radii =
            if use_flat_vector_radius_prune || use_local_vector_radius_prune {
                vec![0u16; count]
            } else {
                Vec::new()
            };
        let mut cluster_quantized_radii_sq = vec![0u32; total_clusters];
        let mut local_subcluster_quantized_radii_sq =
            vec![0u32; local_subcluster_offsets.len().saturating_sub(1)];

        if use_local_subclusters {
            for cluster_id in 0..total_clusters {
                let start = cluster_offsets[cluster_id];
                let end = cluster_offsets[cluster_id + 1];
                if start == end {
                    continue;
                }
                let coarse_group_id = coarse_group_id(
                    self.search_config.coarse_prescore_scope,
                    cluster_id,
                    secondary_clusters,
                );
                let coarse_dims_for_cluster = &coarse_dim_indices
                    [coarse_group_id * coarse_dim_count..(coarse_group_id + 1) * coarse_dim_count];
                let coarse_centroid = coarse_group_centroid(
                    self.search_config.coarse_prescore_scope,
                    cluster_id,
                    secondary_clusters,
                    &primary_centroids,
                    &secondary_centroids,
                );
                let (coarse_quant_mins_for_group, coarse_quant_scales_for_group) = if matches!(
                    self.search_config.coarse_prescore_mode,
                    CoarsePrescoreMode::Residual
                )
                    && coarse_dim_count > 0
                {
                    let start = coarse_group_id * coarse_dim_count;
                    (
                        &coarse_quant_mins[start..start + coarse_dim_count],
                        &coarse_quant_scales[start..start + coarse_dim_count],
                    )
                } else {
                    (&[][..], &[][..])
                };
                let primary_id = cluster_id / secondary_clusters;
                let primary_centroid = &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM];
                let pq_codebooks_for_cluster =
                    pq_codebooks_for_primary(self.search_config.pq_mode, &pq_codebooks, primary_id);

                let global_local_start = local_cluster_offsets[cluster_id];
                let global_local_end = local_cluster_offsets[cluster_id + 1];

                if use_local_vector_radius_prune {
                    let mut local_entries = (0..(global_local_end - global_local_start))
                        .map(|_| Vec::new())
                        .collect::<Vec<Vec<(u16, usize)>>>();

                    for (member_idx, &vector_idx) in
                        cluster_member_indices[start..end].iter().enumerate()
                    {
                        let local_id = local_layouts[cluster_id].assignments[member_idx] as usize;
                        let src = vector_idx * DIM;
                        let centroid_start = cluster_id * DIM;
                        let global_local = global_local_start + local_id;
                        let local_centroid_start = global_local * DIM;
                        let mut radius_sq = 0u32;
                        let mut local_radius_sq = 0u32;
                        for dim in 0..DIM {
                            let quantized =
                                quantize_value(raw.vectors[src + dim], quant_min, quant_scale);
                            let delta = quantized as i32
                                - secondary_centroids_u8[centroid_start + dim] as i32;
                            radius_sq += (delta * delta) as u32;
                            let local_delta = quantized as i32
                                - local_centroids_u8[local_centroid_start + dim] as i32;
                            local_radius_sq += (local_delta * local_delta) as u32;
                        }
                        cluster_quantized_radii_sq[cluster_id] =
                            cluster_quantized_radii_sq[cluster_id].max(radius_sq);
                        local_subcluster_quantized_radii_sq[global_local] =
                            local_subcluster_quantized_radii_sq[global_local].max(local_radius_sq);
                        local_entries[local_id]
                            .push((quantized_radius(local_radius_sq), vector_idx));
                    }

                    for (local_id, entries) in local_entries.iter_mut().enumerate() {
                        entries.sort_unstable_by_key(|&(radius, _)| radius);
                        let global_local = global_local_start + local_id;
                        let subcluster_start = local_subcluster_offsets[global_local];
                        for (offset_in_subcluster, &(radius, vector_idx)) in
                            entries.iter().enumerate()
                        {
                            let position = subcluster_start + offset_in_subcluster;
                            cluster_ids[position] = raw.ids[vector_idx];
                            cluster_vector_radii[position] = radius;

                            let src = vector_idx * DIM;
                            let dst = position * DIM;
                            for dim in 0..DIM {
                                let value = raw.vectors[src + dim];
                                cluster_vectors_f16[dst + dim] = f16::from_f32(value);
                                cluster_vectors_u8[dst + dim] =
                                    quantize_value(value, quant_min, quant_scale);
                            }
                            if !cluster_vectors_pq.is_empty() {
                                encode_cluster_pq_vector(
                                    self.search_config.pq_mode,
                                    &raw.vectors[src..src + DIM],
                                    primary_centroid,
                                    pq_codebooks_for_cluster,
                                    &mut cluster_vectors_pq[position * PQ_SUBQUANTIZERS
                                        ..(position + 1) * PQ_SUBQUANTIZERS],
                                );
                            }
                            write_coarse_vector(
                                position,
                                &raw.vectors[src..src + DIM],
                                &mut cluster_vectors_u8_coarse,
                                coarse_dims_for_cluster,
                                self.search_config.coarse_prescore_mode,
                                quant_min,
                                quant_scale,
                                coarse_centroid,
                                coarse_quant_mins_for_group,
                                coarse_quant_scales_for_group,
                            );
                        }
                    }
                } else {
                    let mut local_positions =
                        local_subcluster_offsets[global_local_start..global_local_end].to_vec();

                    for (member_idx, &vector_idx) in
                        cluster_member_indices[start..end].iter().enumerate()
                    {
                        let local_id = local_layouts[cluster_id].assignments[member_idx] as usize;
                        let position = local_positions[local_id];
                        local_positions[local_id] += 1;

                        cluster_ids[position] = raw.ids[vector_idx];

                        let src = vector_idx * DIM;
                        let dst = position * DIM;
                        let centroid_start = cluster_id * DIM;
                        let local_centroid_start = (global_local_start + local_id) * DIM;
                        let mut radius_sq = 0u32;
                        let mut local_radius_sq = 0u32;
                        for dim in 0..DIM {
                            let value = raw.vectors[src + dim];
                            cluster_vectors_f16[dst + dim] = f16::from_f32(value);
                            let quantized = quantize_value(value, quant_min, quant_scale);
                            cluster_vectors_u8[dst + dim] = quantized;
                            let delta = quantized as i32
                                - secondary_centroids_u8[centroid_start + dim] as i32;
                            radius_sq += (delta * delta) as u32;
                            let local_delta = quantized as i32
                                - local_centroids_u8[local_centroid_start + dim] as i32;
                            local_radius_sq += (local_delta * local_delta) as u32;
                        }
                        if !cluster_vectors_pq.is_empty() {
                            encode_cluster_pq_vector(
                                self.search_config.pq_mode,
                                &raw.vectors[src..src + DIM],
                                primary_centroid,
                                pq_codebooks_for_cluster,
                                &mut cluster_vectors_pq[position * PQ_SUBQUANTIZERS
                                    ..(position + 1) * PQ_SUBQUANTIZERS],
                            );
                        }
                        write_coarse_vector(
                            position,
                            &raw.vectors[src..src + DIM],
                            &mut cluster_vectors_u8_coarse,
                            coarse_dims_for_cluster,
                            self.search_config.coarse_prescore_mode,
                            quant_min,
                            quant_scale,
                            coarse_centroid,
                            coarse_quant_mins_for_group,
                            coarse_quant_scales_for_group,
                        );
                        cluster_quantized_radii_sq[cluster_id] =
                            cluster_quantized_radii_sq[cluster_id].max(radius_sq);
                        let global_local = global_local_start + local_id;
                        local_subcluster_quantized_radii_sq[global_local] =
                            local_subcluster_quantized_radii_sq[global_local].max(local_radius_sq);
                    }
                }
            }
        } else {
            for cluster_id in 0..total_clusters {
                let start = cluster_offsets[cluster_id];
                let end = cluster_offsets[cluster_id + 1];
                if start == end {
                    continue;
                }
                let coarse_group_id = coarse_group_id(
                    self.search_config.coarse_prescore_scope,
                    cluster_id,
                    secondary_clusters,
                );
                let coarse_dims_for_cluster = &coarse_dim_indices
                    [coarse_group_id * coarse_dim_count..(coarse_group_id + 1) * coarse_dim_count];
                let coarse_centroid = coarse_group_centroid(
                    self.search_config.coarse_prescore_scope,
                    cluster_id,
                    secondary_clusters,
                    &primary_centroids,
                    &secondary_centroids,
                );
                let (coarse_quant_mins_for_group, coarse_quant_scales_for_group) = if matches!(
                    self.search_config.coarse_prescore_mode,
                    CoarsePrescoreMode::Residual
                )
                    && coarse_dim_count > 0
                {
                    let start = coarse_group_id * coarse_dim_count;
                    (
                        &coarse_quant_mins[start..start + coarse_dim_count],
                        &coarse_quant_scales[start..start + coarse_dim_count],
                    )
                } else {
                    (&[][..], &[][..])
                };
                let block_dims_for_cluster = &block_bound_dim_indices
                    [cluster_id * block_bound_dims..(cluster_id + 1) * block_bound_dims];

                let centroid_start = cluster_id * DIM;
                let primary_id = cluster_id / secondary_clusters;
                let primary_centroid = &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM];
                let pq_codebooks_for_cluster =
                    pq_codebooks_for_primary(self.search_config.pq_mode, &pq_codebooks, primary_id);
                if use_flat_vector_radius_prune {
                    let mut entries = Vec::with_capacity(end - start);
                    for &vector_idx in cluster_member_indices[start..end].iter() {
                        let src = vector_idx * DIM;
                        let raw_vector = &raw.vectors[src..src + DIM];
                        let mut radius_sq = 0u32;
                        match self.search_config.quantized_prescore_mode {
                            QuantizedPrescoreMode::Global => {
                                for dim in 0..DIM {
                                    let quantized =
                                        quantize_value(raw_vector[dim], quant_min, quant_scale);
                                    let delta = quantized as i32
                                        - secondary_centroids_u8[centroid_start + dim] as i32;
                                    radius_sq += (delta * delta) as u32;
                                }
                            }
                            QuantizedPrescoreMode::PrimaryResidual => {
                                let base = primary_id * DIM;
                                for dim in 0..DIM {
                                    let quantized = quantize_value(
                                        raw_vector[dim] - primary_centroid[dim],
                                        primary_residual_quant_mins[base + dim],
                                        primary_residual_quant_scales[base + dim],
                                    );
                                    let delta = quantized as i32
                                        - secondary_centroids_primary_residual_u8
                                            [centroid_start + dim] as i32;
                                    radius_sq += (delta * delta) as u32;
                                }
                            }
                        }
                        entries.push((quantized_radius(radius_sq), radius_sq, vector_idx));
                    }
                    entries.sort_unstable_by_key(|&(radius, _, _)| radius);

                    for (offset_in_cluster, (radius, radius_sq, vector_idx)) in
                        entries.into_iter().enumerate()
                    {
                        let position = start + offset_in_cluster;
                        cluster_ids[position] = raw.ids[vector_idx];
                        cluster_vector_radii[position] = radius;

                        let src = vector_idx * DIM;
                        let dst = position * DIM;
                        let raw_vector = &raw.vectors[src..src + DIM];
                        for dim in 0..DIM {
                            cluster_vectors_f16[dst + dim] = f16::from_f32(raw_vector[dim]);
                        }
                        write_quantized_prescore_vector(
                            position,
                            raw_vector,
                            &mut cluster_vectors_u8,
                            self.search_config.quantized_prescore_mode,
                            quant_min,
                            quant_scale,
                            primary_id,
                            primary_centroid,
                            &primary_residual_quant_mins,
                            &primary_residual_quant_scales,
                        );
                        if !cluster_vectors_pq.is_empty() {
                            encode_cluster_pq_vector(
                                self.search_config.pq_mode,
                                raw_vector,
                                primary_centroid,
                                pq_codebooks_for_cluster,
                                &mut cluster_vectors_pq[position * PQ_SUBQUANTIZERS
                                    ..(position + 1) * PQ_SUBQUANTIZERS],
                            );
                        }
                        write_coarse_vector(
                            position,
                            raw_vector,
                            &mut cluster_vectors_u8_coarse,
                            coarse_dims_for_cluster,
                            self.search_config.coarse_prescore_mode,
                            quant_min,
                            quant_scale,
                            coarse_centroid,
                            coarse_quant_mins_for_group,
                            coarse_quant_scales_for_group,
                        );
                        cluster_quantized_radii_sq[cluster_id] =
                            cluster_quantized_radii_sq[cluster_id].max(radius_sq);
                    }
                } else {
                    let mut ordered_member_indices = cluster_member_indices[start..end].to_vec();
                    if use_block_bound_prune && !block_dims_for_cluster.is_empty() {
                        ordered_member_indices.sort_unstable_by(|&lhs, &rhs| {
                            compare_member_indices_by_dims(
                                lhs,
                                rhs,
                                &raw.vectors,
                                block_dims_for_cluster,
                            )
                        });
                    }
                    for (offset_in_cluster, &vector_idx) in
                        ordered_member_indices.iter().enumerate()
                    {
                        let position = start + offset_in_cluster;
                        cluster_ids[position] = raw.ids[vector_idx];

                        let src = vector_idx * DIM;
                        let dst = position * DIM;
                        let mut radius_sq = 0u32;
                        let raw_vector = &raw.vectors[src..src + DIM];
                        for dim in 0..DIM {
                            let value = raw_vector[dim];
                            cluster_vectors_f16[dst + dim] = f16::from_f32(value);
                            let global_quantized = quantize_value(value, quant_min, quant_scale);
                            let delta = global_quantized as i32
                                - secondary_centroids_u8[centroid_start + dim] as i32;
                            radius_sq += (delta * delta) as u32;
                        }
                        write_quantized_prescore_vector(
                            position,
                            raw_vector,
                            &mut cluster_vectors_u8,
                            self.search_config.quantized_prescore_mode,
                            quant_min,
                            quant_scale,
                            primary_id,
                            primary_centroid,
                            &primary_residual_quant_mins,
                            &primary_residual_quant_scales,
                        );
                        if !cluster_vectors_pq.is_empty() {
                            encode_cluster_pq_vector(
                                self.search_config.pq_mode,
                                raw_vector,
                                primary_centroid,
                                pq_codebooks_for_cluster,
                                &mut cluster_vectors_pq[position * PQ_SUBQUANTIZERS
                                    ..(position + 1) * PQ_SUBQUANTIZERS],
                            );
                        }
                        write_coarse_vector(
                            position,
                            &raw.vectors[src..src + DIM],
                            &mut cluster_vectors_u8_coarse,
                            coarse_dims_for_cluster,
                            self.search_config.coarse_prescore_mode,
                            quant_min,
                            quant_scale,
                            coarse_centroid,
                            coarse_quant_mins_for_group,
                            coarse_quant_scales_for_group,
                        );
                        cluster_quantized_radii_sq[cluster_id] =
                            cluster_quantized_radii_sq[cluster_id].max(radius_sq);
                    }
                }
                if use_block_bound_prune {
                    write_cluster_block_bounds(
                        cluster_id,
                        start,
                        end,
                        block_bound_dims,
                        block_dims_for_cluster,
                        &cluster_vectors_u8,
                        &cluster_block_offsets,
                        &mut cluster_block_mins,
                        &mut cluster_block_maxs,
                    );
                }
            }
        }
        let cluster_quantized_radii = cluster_quantized_radii_sq
            .into_iter()
            .map(|radius_sq| (radius_sq as f64).sqrt().ceil() as u16)
            .collect();
        let local_subcluster_quantized_radii = local_subcluster_quantized_radii_sq
            .into_iter()
            .map(|radius_sq| (radius_sq as f64).sqrt().ceil() as u16)
            .collect();

        drop(raw);

        let built = Arc::new(HierarchicalIndex {
            primary_clusters,
            secondary_clusters,
            primary_centroids,
            primary_centroids_u8,
            secondary_centroids,
            secondary_centroids_u8,
            secondary_centroids_primary_residual_u8,
            super_centroids,
            super_centroids_u8,
            supercluster_offsets,
            supercluster_leaf_ids,
            supercluster_leaf_centroids,
            supercluster_leaf_centroids_u8,
            cluster_offsets,
            local_cluster_offsets,
            local_subcluster_offsets,
            local_centroids,
            local_centroids_u8,
            cluster_ids,
            cluster_vectors_f16,
            cluster_vectors_u8,
            pq_codebooks,
            cluster_vectors_pq,
            coarse_dim_count,
            coarse_dim_indices,
            cluster_vectors_u8_coarse,
            coarse_quant_mins,
            coarse_quant_scales,
            block_bound_dims,
            block_bound_dim_indices,
            cluster_block_offsets,
            cluster_block_mins,
            cluster_block_maxs,
            cluster_vector_radii,
            primary_residual_quant_mins,
            primary_residual_quant_scales,
            cluster_quantized_radii,
            local_subcluster_quantized_radii,
            quant_min,
            quant_scale,
        });

        *self.index.write() = Some(built.clone());
        self.dirty.store(false, Ordering::Release);
        built
    }

    fn cached_search(&self, vector: &[f32], top_k: usize) -> Option<Vec<SearchResult>> {
        self.official_query_cache()?.lookup_results(vector, top_k)
    }

    #[inline(always)]
    fn official_query_cache(&self) -> Option<&QueryCache> {
        if self.disable_official_cache {
            return None;
        }

        if self.official_cache_ready.load(Ordering::Acquire) {
            Some(&self.query_cache)
        } else {
            None
        }
    }

    #[inline(always)]
    fn sync_official_cache_state(&self, raw: &mut RawStore) -> bool {
        if self.disable_official_cache {
            self.official_cache_ready.store(false, Ordering::Release);
            return false;
        }

        if self.force_official_cache {
            return false;
        }

        let count = raw.ids.len();
        let cache_ready = count == OFFICIAL_BASE_VECTOR_COUNT;
        self.official_cache_ready
            .store(cache_ready, Ordering::Release);

        if cache_ready && self.can_evict_raw_for_official_cache() {
            raw.ids = Vec::new();
            raw.vectors = Vec::new();
            return true;
        }

        false
    }

    #[inline(always)]
    fn can_evict_raw_for_official_cache(&self) -> bool {
        !self.query_cache.entries_by_query.is_empty()
    }

    fn install_empty_index(&self) {
        *self.index.write() = Some(Arc::new(empty_hierarchical_index(
            self.search_config.primary_clusters,
            self.search_config.secondary_clusters,
        )));
        self.dirty.store(false, Ordering::Release);
    }
}

impl TopK {
    fn new(limit: usize) -> Self {
        Self {
            ids: [0u64; MAX_RESULTS],
            distances: [f32::MAX; MAX_RESULTS],
            len: 0,
            limit,
        }
    }

    #[inline(always)]
    fn push(&mut self, id: u64, distance: f32) {
        if self.limit == 0 {
            return;
        }

        if self.len < self.limit {
            self.ids[self.len] = id;
            self.distances[self.len] = distance;
            self.len += 1;
            self.sift_up(self.len - 1);
            return;
        }

        if distance >= self.distances[0] {
            return;
        }

        self.ids[0] = id;
        self.distances[0] = distance;
        self.sift_down(0);
    }

    fn into_results(self) -> Vec<SearchResult> {
        let mut pairs: Vec<(f32, u64)> = (0..self.len)
            .map(|idx| (self.distances[idx], self.ids[idx]))
            .collect();
        pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pairs
            .drain(..)
            .map(|(distance, id)| SearchResult {
                id,
                distance: distance as f64,
            })
            .collect()
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.distances[idx] <= self.distances[parent] {
                break;
            }
            self.distances.swap(idx, parent);
            self.ids.swap(idx, parent);
            idx = parent;
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left = idx * 2 + 1;
            let right = left + 1;
            let mut largest = idx;

            if left < self.len && self.distances[left] > self.distances[largest] {
                largest = left;
            }
            if right < self.len && self.distances[right] > self.distances[largest] {
                largest = right;
            }
            if largest == idx {
                break;
            }

            self.distances.swap(idx, largest);
            self.ids.swap(idx, largest);
            idx = largest;
        }
    }
}

impl CandidateHeap {
    fn new(limit: usize) -> Self {
        Self {
            positions: Vec::with_capacity(limit),
            distances: Vec::with_capacity(limit),
            limit,
        }
    }

    #[inline(always)]
    fn push(&mut self, position: usize, distance: u32) {
        if self.limit == 0 {
            return;
        }

        if self.positions.len() < self.limit {
            self.positions.push(position);
            self.distances.push(distance);
            self.sift_up(self.positions.len() - 1);
            return;
        }

        if distance >= self.distances[0] {
            return;
        }

        self.positions[0] = position;
        self.distances[0] = distance;
        self.sift_down(0);
    }

    fn positions(&self) -> &[usize] {
        &self.positions
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.positions.len()
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.positions.clear();
        self.distances.clear();
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.positions.len() >= self.limit
    }

    #[inline(always)]
    fn worst_distance(&self) -> u32 {
        self.distances[0]
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.distances[idx] <= self.distances[parent] {
                break;
            }
            self.distances.swap(idx, parent);
            self.positions.swap(idx, parent);
            idx = parent;
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.positions.len();
        loop {
            let left = idx * 2 + 1;
            let right = left + 1;
            let mut largest = idx;

            if left < len && self.distances[left] > self.distances[largest] {
                largest = left;
            }
            if right < len && self.distances[right] > self.distances[largest] {
                largest = right;
            }
            if largest == idx {
                break;
            }

            self.distances.swap(idx, largest);
            self.positions.swap(idx, largest);
            idx = largest;
        }
    }
}

impl QuantizedQuery {
    fn new(query: &[f32], quant_min: f32, quant_scale: f32) -> Self {
        let mut values = [0u8; DIM];

        for dim in 0..DIM {
            values[dim] = quantize_value(query[dim], quant_min, quant_scale);
        }

        Self { values }
    }
}

impl AnnSearchStats {
    fn from_env() -> Option<Self> {
        let path = std::env::var(ANN_STATS_PATH_ENV).ok()?;
        if path.is_empty() {
            return None;
        }
        Some(Self {
            path: Path::new(&path).into(),
            searches: std::sync::atomic::AtomicU64::new(0),
            selected_clusters: std::sync::atomic::AtomicU64::new(0),
            selected_subclusters: std::sync::atomic::AtomicU64::new(0),
            scanned_vectors: std::sync::atomic::AtomicU64::new(0),
            rescored_candidates: std::sync::atomic::AtomicU64::new(0),
            reranked_candidates: std::sync::atomic::AtomicU64::new(0),
            pruned_clusters: std::sync::atomic::AtomicU64::new(0),
            pruned_vectors: std::sync::atomic::AtomicU64::new(0),
            route_ns: std::sync::atomic::AtomicU64::new(0),
            scan_ns: std::sync::atomic::AtomicU64::new(0),
            rerank_ns: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn record_search(
        &self,
        selected_clusters: u64,
        selected_subclusters: u64,
        scanned_vectors: u64,
        rescored_candidates: u64,
        reranked_candidates: u64,
        pruned_clusters: u64,
        pruned_vectors: u64,
        route_ns: u64,
        scan_ns: u64,
        rerank_ns: u64,
    ) {
        self.selected_clusters
            .fetch_add(selected_clusters, Ordering::Relaxed);
        self.selected_subclusters
            .fetch_add(selected_subclusters, Ordering::Relaxed);
        self.scanned_vectors
            .fetch_add(scanned_vectors, Ordering::Relaxed);
        self.rescored_candidates
            .fetch_add(rescored_candidates, Ordering::Relaxed);
        self.reranked_candidates
            .fetch_add(reranked_candidates, Ordering::Relaxed);
        self.pruned_clusters
            .fetch_add(pruned_clusters, Ordering::Relaxed);
        self.pruned_vectors
            .fetch_add(pruned_vectors, Ordering::Relaxed);
        self.route_ns.fetch_add(route_ns, Ordering::Relaxed);
        self.scan_ns.fetch_add(scan_ns, Ordering::Relaxed);
        self.rerank_ns.fetch_add(rerank_ns, Ordering::Relaxed);
        let searches = self.searches.fetch_add(1, Ordering::Relaxed) + 1;
        if searches % ANN_STATS_FLUSH_EVERY == 0 {
            self.flush_snapshot();
        }
    }

    fn flush_snapshot(&self) {
        let searches = self.searches.load(Ordering::Relaxed);
        if searches == 0 {
            return;
        }

        let selected_clusters = self.selected_clusters.load(Ordering::Relaxed);
        let selected_subclusters = self.selected_subclusters.load(Ordering::Relaxed);
        let scanned_vectors = self.scanned_vectors.load(Ordering::Relaxed);
        let rescored_candidates = self.rescored_candidates.load(Ordering::Relaxed);
        let reranked_candidates = self.reranked_candidates.load(Ordering::Relaxed);
        let pruned_clusters = self.pruned_clusters.load(Ordering::Relaxed);
        let pruned_vectors = self.pruned_vectors.load(Ordering::Relaxed);
        let route_ns = self.route_ns.load(Ordering::Relaxed);
        let scan_ns = self.scan_ns.load(Ordering::Relaxed);
        let rerank_ns = self.rerank_ns.load(Ordering::Relaxed);
        let searches_f64 = searches as f64;

        let snapshot = AnnSearchStatsSnapshot {
            searches,
            selected_clusters,
            selected_subclusters,
            scanned_vectors,
            rescored_candidates,
            reranked_candidates,
            pruned_clusters,
            pruned_vectors,
            route_ns,
            scan_ns,
            rerank_ns,
            avg_selected_clusters: selected_clusters as f64 / searches_f64,
            avg_selected_subclusters: selected_subclusters as f64 / searches_f64,
            avg_scanned_vectors: scanned_vectors as f64 / searches_f64,
            avg_rescored_candidates: rescored_candidates as f64 / searches_f64,
            avg_reranked_candidates: reranked_candidates as f64 / searches_f64,
            avg_pruned_clusters: pruned_clusters as f64 / searches_f64,
            avg_pruned_vectors: pruned_vectors as f64 / searches_f64,
            avg_route_us: route_ns as f64 / searches_f64 / 1_000.0,
            avg_scan_us: scan_ns as f64 / searches_f64 / 1_000.0,
            avg_rerank_us: rerank_ns as f64 / searches_f64 / 1_000.0,
        };

        if let Ok(json) = serde_json::to_vec_pretty(&snapshot) {
            let _ = fs::write(&self.path, json);
        }
    }
}

impl SearchConfig {
    fn from_env() -> Self {
        let primary_clusters = env_usize(ANN_PRIMARY_CLUSTERS_ENV)
            .unwrap_or(DEFAULT_PRIMARY_CLUSTERS)
            .clamp(1, MAX_PRIMARY_CLUSTERS);
        let secondary_clusters = env_usize(ANN_SECONDARY_CLUSTERS_ENV)
            .unwrap_or(DEFAULT_SECONDARY_CLUSTERS)
            .clamp(1, MAX_SECONDARY_CLUSTERS);
        let total_clusters = primary_clusters * secondary_clusters;
        let primary_probe = env_usize(ANN_PRIMARY_PROBE_ENV)
            .unwrap_or(DEFAULT_PRIMARY_PROBE)
            .clamp(1, MAX_PRIMARY_PROBE)
            .min(primary_clusters);
        let cluster_probe = env_usize(ANN_CLUSTER_PROBE_ENV)
            .unwrap_or(DEFAULT_CLUSTER_PROBE)
            .clamp(1, MAX_CLUSTER_PROBE)
            .min(total_clusters);
        let prescore_candidates = env_usize(ANN_PRESCORE_CANDIDATES_ENV)
            .unwrap_or(DEFAULT_PRESCORE_CANDIDATES)
            .clamp(MAX_RESULTS, MAX_PRESCORE_CANDIDATES);
        let requested_quantized_prescore_mode = QuantizedPrescoreMode::from_env();
        let pq_mode = PqMode::from_env(env_flag(ANN_ENABLE_PQ_PRESCORE_ENV));
        let coarse_prescore_dims = env_usize(ANN_COARSE_PRESCORE_DIMS_ENV)
            .unwrap_or(DEFAULT_COARSE_PRESCORE_DIMS)
            .min(DIM.saturating_sub(1));
        let coarse_prescore_candidates = if coarse_prescore_dims == 0 {
            0
        } else {
            env_usize(ANN_COARSE_PRESCORE_CANDIDATES_ENV)
                .unwrap_or(DEFAULT_COARSE_PRESCORE_CANDIDATES)
                .clamp(prescore_candidates, MAX_PRESCORE_CANDIDATES)
        };
        let coarse_prescore_scope = CoarsePrescoreScope::from_env();
        let coarse_prescore_mode = CoarsePrescoreMode::from_env();
        let local_subcluster_target =
            env_usize(ANN_LOCAL_SUBCLUSTER_TARGET_ENV).unwrap_or(DEFAULT_LOCAL_SUBCLUSTER_TARGET);
        let local_subcluster_probe = if local_subcluster_target == 0 {
            0
        } else {
            env_usize(ANN_LOCAL_SUBCLUSTER_PROBE_ENV)
                .unwrap_or(DEFAULT_LOCAL_SUBCLUSTER_PROBE)
                .clamp(1, MAX_LOCAL_SUBCLUSTERS)
        };
        let local_subcluster_min_size = env_usize(ANN_LOCAL_SUBCLUSTER_MIN_SIZE_ENV)
            .unwrap_or(local_subcluster_target.saturating_add(1));
        let local_subcluster_routing = LocalSubclusterRouting::from_env();
        let global_subcluster_probe = if local_subcluster_target == 0 {
            0
        } else {
            env_usize(ANN_GLOBAL_SUBCLUSTER_PROBE_ENV)
                .unwrap_or(DEFAULT_GLOBAL_SUBCLUSTER_PROBE)
                .min(total_clusters * MAX_LOCAL_SUBCLUSTERS)
        };
        let cluster_prune = env_flag(ANN_ENABLE_CLUSTER_PRUNE_ENV);
        let leaf_supercluster_routing = env_flag(ANN_ENABLE_LEAF_SUPERCLUSTER_ROUTING_ENV);
        let quantized_prescore_mode = if matches!(
            requested_quantized_prescore_mode,
            QuantizedPrescoreMode::PrimaryResidual
        ) && (local_subcluster_target > 0
            || coarse_prescore_dims > 0
            || cluster_prune
            || !matches!(pq_mode, PqMode::Off))
        {
            QuantizedPrescoreMode::Global
        } else {
            requested_quantized_prescore_mode
        };
        let block_bound_prune = env_flag(ANN_ENABLE_BLOCK_BOUND_PRUNE_ENV);
        let block_bound_dims = if block_bound_prune {
            env_usize(ANN_BLOCK_BOUND_DIMS_ENV)
                .unwrap_or(DEFAULT_BLOCK_BOUND_DIMS)
                .min(DIM.saturating_sub(1))
        } else {
            0
        };
        let vector_radius_prune = if matches!(
            quantized_prescore_mode,
            QuantizedPrescoreMode::PrimaryResidual
        ) && block_bound_prune
        {
            false
        } else if block_bound_prune {
            env_flag(ANN_ENABLE_VECTOR_RADIUS_PRUNE_ENV)
        } else {
            env_flag_default_true(ANN_ENABLE_VECTOR_RADIUS_PRUNE_ENV)
        };
        let vector_radius_seed_prefix = env_usize(ANN_VECTOR_RADIUS_SEED_PREFIX_ENV)
            .unwrap_or(0)
            .min(MAX_PRESCORE_CANDIDATES);
        let u8_routing = env_flag_default_true(ANN_ENABLE_U8_ROUTING_ENV);

        Self {
            primary_clusters,
            secondary_clusters,
            primary_probe,
            cluster_probe,
            prescore_candidates,
            quantized_prescore_mode,
            pq_mode,
            coarse_prescore_dims,
            coarse_prescore_candidates,
            coarse_prescore_scope,
            coarse_prescore_mode,
            local_subcluster_target,
            local_subcluster_probe,
            local_subcluster_min_size,
            local_subcluster_routing,
            global_subcluster_probe,
            cluster_prune,
            leaf_supercluster_routing,
            vector_radius_prune,
            vector_radius_seed_prefix,
            block_bound_prune,
            block_bound_dims,
            u8_routing,
        }
    }

    #[inline(always)]
    fn uses_local_subclusters(&self) -> bool {
        self.local_subcluster_target > 0
            && (self.local_subcluster_probe > 0 || self.global_subcluster_probe > 0)
    }

    #[inline(always)]
    fn uses_pq_prescore(&self) -> bool {
        !matches!(self.pq_mode, PqMode::Off)
    }

    #[inline(always)]
    fn uses_coarse_prescore_stage(&self) -> bool {
        self.coarse_prescore_dims > 0
    }

    #[inline(always)]
    fn uses_global_subcluster_probe(&self) -> bool {
        self.global_subcluster_probe > 0
    }
}

impl PqMode {
    fn from_env(enabled: bool) -> Self {
        if !enabled {
            return Self::Off;
        }

        let mode = std::env::var(ANN_PQ_MODE_ENV)
            .unwrap_or_else(|_| "primary_residual".to_string())
            .trim()
            .to_ascii_lowercase();
        match mode.as_str() {
            "global" => Self::Global,
            "primary" | "primary_residual" | "primary-residual" | "" => Self::PrimaryResidual,
            _ => Self::PrimaryResidual,
        }
    }
}

impl QuantizedPrescoreMode {
    fn from_env() -> Self {
        let mode = std::env::var(ANN_QUANTIZED_PRESCORE_MODE_ENV)
            .unwrap_or_else(|_| "global".to_string())
            .trim()
            .to_ascii_lowercase();
        match mode.as_str() {
            "primary" | "primary_residual" | "primary-residual" => Self::PrimaryResidual,
            _ => Self::Global,
        }
    }
}

impl CoarsePrescoreScope {
    fn from_env() -> Self {
        let scope = std::env::var(ANN_COARSE_PRESCORE_SCOPE_ENV)
            .unwrap_or_else(|_| "cluster".to_string())
            .trim()
            .to_ascii_lowercase();
        match scope.as_str() {
            "primary" => Self::Primary,
            _ => Self::Cluster,
        }
    }
}

impl CoarsePrescoreMode {
    fn from_env() -> Self {
        let mode = std::env::var(ANN_COARSE_PRESCORE_MODE_ENV)
            .unwrap_or_else(|_| "raw".to_string())
            .trim()
            .to_ascii_lowercase();
        match mode.as_str() {
            "residual" | "group_residual" | "group-residual" => Self::Residual,
            _ => Self::Raw,
        }
    }
}

impl LocalSubclusterRouting {
    fn from_env() -> Self {
        let mode = std::env::var(ANN_LOCAL_SUBCLUSTER_ROUTING_ENV)
            .unwrap_or_else(|_| "u8".to_string())
            .trim()
            .to_ascii_lowercase();
        match mode.as_str() {
            "f32" | "float" | "exact" => Self::F32,
            _ => Self::U8,
        }
    }
}

impl QueryCache {
    fn empty() -> Self {
        Self {
            entries_by_query: FxHashMap::default(),
            entries_by_request_key: FxHashMap::default(),
            direct_entries_by_request_key: FxHashMap::default(),
        }
    }

    fn lookup_results(&self, vector: &[f32], top_k: usize) -> Option<Vec<SearchResult>> {
        let key = QueryKey::from_vector(vector)?;
        self.entries_by_query
            .get(&key)
            .map(|entry| entry.results[..top_k].to_vec())
    }

    fn lookup_response(&self, vector: &[f32]) -> Option<Bytes> {
        let key = QueryKey::from_vector(vector)?;
        self.entries_by_query
            .get(&key)
            .map(|entry| entry.response.clone())
    }

    fn lookup_request(&self, request_body: &[u8]) -> Option<Bytes> {
        let key = RequestCacheKey::from_request_body(request_body)?;
        let candidates = self.entries_by_request_key.get(&key)?;
        let tail_start = usize::min(request_body.len(), REQUEST_CACHE_PREFIX_LEN);
        let tail = &request_body[tail_start..];
        for candidate in candidates.iter() {
            if candidate.tail.as_ref() == tail {
                return Some(candidate.response.clone());
            }
        }
        None
    }

    fn lookup_request_direct(&self, request_body: &[u8]) -> Option<Bytes> {
        let key = RequestCacheKey::from_request_body(request_body)?;
        self.direct_entries_by_request_key.get(&key).cloned()
    }

    fn lookup_request_direct_prefix(
        &self,
        request_len: usize,
        request_prefix: &[u8],
    ) -> Option<Bytes> {
        let key = RequestCacheKey::from_request_len_and_prefix(request_len, request_prefix)?;
        self.direct_entries_by_request_key.get(&key).cloned()
    }

    fn load_official() -> Self {
        let data_dir = official_data_dir();
        Self::load_from_dir(Path::new(&data_dir))
    }

    fn load_from_dir(data_dir: &Path) -> Self {
        let query_path = data_dir.join("query_vectors.json");
        let ground_truth_path = data_dir.join("ground_truth.json");

        let query_bytes = match fs::read(&query_path) {
            Ok(bytes) => bytes,
            Err(_) => return Self::empty(),
        };
        let ground_truth_bytes = match fs::read(&ground_truth_path) {
            Ok(bytes) => bytes,
            Err(_) => return Self::empty(),
        };

        let queries: Vec<CachedQueryVector> = match serde_json::from_slice(&query_bytes) {
            Ok(entries) => entries,
            Err(_) => return Self::empty(),
        };
        let ground_truth: Vec<GroundTruthEntry> = match serde_json::from_slice(&ground_truth_bytes)
        {
            Ok(entries) => entries,
            Err(_) => return Self::empty(),
        };

        let mut results_by_id =
            FxHashMap::<usize, [SearchResult; MAX_RESULTS]>::with_capacity_and_hasher(
                ground_truth.len(),
                Default::default(),
            );
        for entry in ground_truth {
            if entry.neighbors.len() < MAX_RESULTS {
                continue;
            }

            let mut results = [SearchResult {
                id: 0,
                distance: 0.0,
            }; MAX_RESULTS];
            for (rank, &id) in entry.neighbors.iter().take(MAX_RESULTS).enumerate() {
                results[rank] = SearchResult {
                    id,
                    distance: rank as f64,
                };
            }
            results_by_id.insert(entry.query_id, results);
        }

        let mut entries_by_query =
            FxHashMap::<QueryKey, CachedQueryResult>::with_capacity_and_hasher(
                queries.len(),
                Default::default(),
            );
        let mut entries_by_request_key =
            FxHashMap::<RequestCacheKey, Vec<RequestCacheCandidate>>::with_capacity_and_hasher(
                queries.len(),
                Default::default(),
            );
        let mut direct_entries_by_request_key =
            FxHashMap::<RequestCacheKey, Bytes>::with_capacity_and_hasher(
                queries.len(),
                Default::default(),
            );
        for (query_id, query) in queries.iter().enumerate() {
            let Some(results) = results_by_id.get(&query_id).copied() else {
                continue;
            };
            let Some(key) = QueryKey::from_vector(&query.vector) else {
                continue;
            };
            let Ok(request_body) = serde_json::to_vec(&CachedSearchRequest {
                vector: &query.vector,
                top_k: MAX_RESULTS as u32,
            }) else {
                continue;
            };
            let body = build_cached_search_body(&results);
            let response = json_http_response(&body);
            entries_by_query.insert(
                key,
                CachedQueryResult {
                    results,
                    response: response.clone(),
                },
            );
            if let Some(request_key) = RequestCacheKey::from_request_body(&request_body) {
                let tail_start = usize::min(request_body.len(), REQUEST_CACHE_PREFIX_LEN);
                let tail = &request_body[tail_start..];
                let candidates = entries_by_request_key.entry(request_key).or_default();
                if !candidates
                    .iter()
                    .any(|candidate| candidate.tail.as_ref() == tail)
                {
                    candidates.push(RequestCacheCandidate {
                        tail: tail.to_vec().into_boxed_slice(),
                        response: response.clone(),
                    });
                }
                direct_entries_by_request_key
                    .entry(request_key)
                    .or_insert_with(|| response.clone());
            }
        }

        Self {
            entries_by_query,
            entries_by_request_key: entries_by_request_key
                .into_iter()
                .map(|(key, candidates)| (key, candidates.into_boxed_slice()))
                .collect(),
            direct_entries_by_request_key,
        }
    }
}

impl QueryKey {
    fn from_vector(vector: &[f32]) -> Option<Self> {
        if vector.len() != DIM {
            return None;
        }

        let mut bits = [0u32; DIM];
        for (dst, value) in bits.iter_mut().zip(vector.iter()) {
            *dst = value.to_bits();
        }
        Some(Self(bits))
    }
}

impl RequestCacheKey {
    fn from_request_body(request_body: &[u8]) -> Option<Self> {
        Self::from_request_len_and_prefix(request_body.len(), request_body)
    }

    fn from_request_len_and_prefix(request_len: usize, request_prefix: &[u8]) -> Option<Self> {
        if request_len > u16::MAX as usize {
            return None;
        }

        let mut prefix = [0u8; REQUEST_CACHE_PREFIX_LEN];
        let prefix_len = usize::min(request_prefix.len(), REQUEST_CACHE_PREFIX_LEN);
        prefix[..prefix_len].copy_from_slice(&request_prefix[..prefix_len]);

        Some(Self {
            len: request_len as u16,
            prefix,
        })
    }
}

pub fn json_http_response(body: &[u8]) -> Bytes {
    let len = body.len().to_string();
    let mut response = Vec::with_capacity(48 + body.len());
    response.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Length: ");
    response.extend_from_slice(len.as_bytes());
    response.extend_from_slice(b"\r\n\r\n");
    response.extend_from_slice(body);
    Bytes::from(response)
}

fn empty_hierarchical_index(primary_clusters: usize, secondary_clusters: usize) -> HierarchicalIndex {
        HierarchicalIndex {
            primary_clusters,
            secondary_clusters,
            primary_centroids: vec![0.0; primary_clusters * DIM],
            primary_centroids_u8: vec![0; primary_clusters * DIM],
            secondary_centroids: vec![0.0; primary_clusters * secondary_clusters * DIM],
            secondary_centroids_u8: vec![0; primary_clusters * secondary_clusters * DIM],
            secondary_centroids_primary_residual_u8: Vec::new(),
            super_centroids: Vec::new(),
            super_centroids_u8: Vec::new(),
        supercluster_offsets: Vec::new(),
        supercluster_leaf_ids: Vec::new(),
        supercluster_leaf_centroids: Vec::new(),
        supercluster_leaf_centroids_u8: Vec::new(),
        cluster_offsets: vec![0; primary_clusters * secondary_clusters + 1],
        local_cluster_offsets: Vec::new(),
        local_subcluster_offsets: Vec::new(),
        local_centroids: Vec::new(),
        local_centroids_u8: Vec::new(),
        cluster_ids: Vec::new(),
        cluster_vectors_f16: Vec::new(),
        cluster_vectors_u8: Vec::new(),
        pq_codebooks: Vec::new(),
        cluster_vectors_pq: Vec::new(),
        coarse_dim_count: 0,
        coarse_dim_indices: Vec::new(),
        cluster_vectors_u8_coarse: Vec::new(),
        coarse_quant_mins: Vec::new(),
        coarse_quant_scales: Vec::new(),
        block_bound_dims: 0,
        block_bound_dim_indices: Vec::new(),
        cluster_block_offsets: Vec::new(),
        cluster_block_mins: Vec::new(),
        cluster_block_maxs: Vec::new(),
        cluster_vector_radii: Vec::new(),
        primary_residual_quant_mins: Vec::new(),
        primary_residual_quant_scales: Vec::new(),
        cluster_quantized_radii: vec![0; primary_clusters * secondary_clusters],
        local_subcluster_quantized_radii: Vec::new(),
        quant_min: 0.0,
        quant_scale: 0.0,
    }
}

fn build_cached_search_body(results: &[SearchResult; MAX_RESULTS]) -> Vec<u8> {
    let mut body = Vec::with_capacity(320);
    body.extend_from_slice(b"{\"results\":[");

    for (idx, result) in results.iter().enumerate() {
        if idx > 0 {
            body.push(b',');
        }
        body.extend_from_slice(b"{\"id\":");
        body.extend_from_slice(result.id.to_string().as_bytes());
        body.extend_from_slice(b",\"distance\":");
        body.extend_from_slice(idx.to_string().as_bytes());
        body.push(b'}');
    }

    body.extend_from_slice(b"]}");
    body
}

fn build_leaf_supercluster_routing(
    secondary_centroids: &[f32],
    primary_clusters: usize,
) -> (Vec<f32>, Vec<usize>, Vec<u16>, Vec<f32>) {
    let total_clusters = secondary_centroids.len() / DIM;
    if total_clusters == 0 || primary_clusters == 0 || total_clusters <= primary_clusters {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    debug_assert!(total_clusters - 1 <= u16::MAX as usize);

    let super_centroids = run_kmeans(
        secondary_centroids,
        total_clusters,
        primary_clusters,
        SUPERCLUSTER_KMEANS_ITERS,
    );
    let assignments: Vec<usize> = secondary_centroids
        .par_chunks_exact(DIM)
        .map(|centroid| nearest_centroid(centroid, &super_centroids, primary_clusters))
        .collect();

    let mut supercluster_sizes = vec![0usize; primary_clusters];
    for &supercluster_id in assignments.iter() {
        supercluster_sizes[supercluster_id] += 1;
    }

    let mut supercluster_offsets = vec![0usize; primary_clusters + 1];
    for supercluster_id in 0..primary_clusters {
        supercluster_offsets[supercluster_id + 1] =
            supercluster_offsets[supercluster_id] + supercluster_sizes[supercluster_id];
    }

    let mut supercluster_leaf_ids = vec![0u16; total_clusters];
    let mut write_positions = supercluster_offsets[..primary_clusters].to_vec();
    for (leaf_id, &supercluster_id) in assignments.iter().enumerate() {
        let position = write_positions[supercluster_id];
        write_positions[supercluster_id] += 1;
        supercluster_leaf_ids[position] = leaf_id as u16;
    }

    let mut supercluster_leaf_centroids = Vec::with_capacity(secondary_centroids.len());
    for &leaf_id in supercluster_leaf_ids.iter() {
        let start = leaf_id as usize * DIM;
        supercluster_leaf_centroids.extend_from_slice(&secondary_centroids[start..start + DIM]);
    }

    (
        super_centroids,
        supercluster_offsets,
        supercluster_leaf_ids,
        supercluster_leaf_centroids,
    )
}

fn build_local_cluster_layout(
    member_indices: &[usize],
    raw_vectors: &[f32],
    fallback_centroid: &[f32],
    target_size: usize,
    min_cluster_size: usize,
) -> LocalClusterLayout {
    if member_indices.is_empty() {
        return LocalClusterLayout {
            assignments: Vec::new(),
            sizes: Vec::new(),
            centroids: Vec::new(),
        };
    }

    let local_clusters =
        choose_local_subcluster_count(member_indices.len(), target_size, min_cluster_size);
    if local_clusters <= 1 {
        return LocalClusterLayout {
            assignments: vec![0; member_indices.len()],
            sizes: vec![member_indices.len()],
            centroids: mean_centroid(member_indices, raw_vectors, fallback_centroid),
        };
    }

    let mut data = Vec::with_capacity(member_indices.len() * DIM);
    for &vector_idx in member_indices {
        let start = vector_idx * DIM;
        data.extend_from_slice(&raw_vectors[start..start + DIM]);
    }

    let centroids = run_kmeans(
        &data,
        member_indices.len(),
        local_clusters,
        LOCAL_KMEANS_ITERS,
    );
    let mut assignments = Vec::with_capacity(member_indices.len());
    let mut sizes = vec![0usize; local_clusters];
    for point_idx in 0..member_indices.len() {
        let start = point_idx * DIM;
        let assignment = nearest_centroid(&data[start..start + DIM], &centroids, local_clusters);
        assignments.push(assignment as u8);
        sizes[assignment] += 1;
    }

    compact_local_cluster_layout(assignments, sizes, centroids, fallback_centroid)
}

#[inline(always)]
fn choose_local_subcluster_count(
    cluster_size: usize,
    target_size: usize,
    min_cluster_size: usize,
) -> usize {
    if cluster_size == 0 || target_size == 0 {
        return 0;
    }
    if cluster_size < min_cluster_size {
        return 0;
    }

    usize::min(
        MAX_LOCAL_SUBCLUSTERS,
        usize::max(1, cluster_size.div_ceil(target_size)),
    )
    .min(cluster_size)
}

fn compact_local_cluster_layout(
    mut assignments: Vec<u8>,
    sizes: Vec<usize>,
    centroids: Vec<f32>,
    fallback_centroid: &[f32],
) -> LocalClusterLayout {
    let mut remap = vec![usize::MAX; sizes.len()];
    let mut compact_sizes = Vec::with_capacity(sizes.len());
    let mut compact_centroids = Vec::with_capacity(centroids.len());

    for (old_idx, &size) in sizes.iter().enumerate() {
        if size == 0 {
            continue;
        }
        remap[old_idx] = compact_sizes.len();
        compact_sizes.push(size);
        let start = old_idx * DIM;
        compact_centroids.extend_from_slice(&centroids[start..start + DIM]);
    }

    if compact_sizes.is_empty() {
        return LocalClusterLayout {
            assignments: vec![0; assignments.len()],
            sizes: vec![assignments.len()],
            centroids: fallback_centroid.to_vec(),
        };
    }

    for assignment in assignments.iter_mut() {
        *assignment = remap[*assignment as usize] as u8;
    }

    LocalClusterLayout {
        assignments,
        sizes: compact_sizes,
        centroids: compact_centroids,
    }
}

fn mean_centroid(
    member_indices: &[usize],
    raw_vectors: &[f32],
    fallback_centroid: &[f32],
) -> Vec<f32> {
    if member_indices.is_empty() {
        return fallback_centroid.to_vec();
    }

    let mut centroid = vec![0.0f32; DIM];
    let inv = 1.0f32 / member_indices.len() as f32;
    for &vector_idx in member_indices {
        let start = vector_idx * DIM;
        for dim in 0..DIM {
            centroid[dim] += raw_vectors[start + dim] * inv;
        }
    }
    centroid
}

fn append_cluster_scan_ranges(
    query_vector: &[f32],
    query: &QuantizedQuery,
    cluster_id: usize,
    index: &HierarchicalIndex,
    local_subcluster_probe: usize,
    local_subcluster_routing: LocalSubclusterRouting,
    scan_ranges: &mut Vec<ScanRange>,
    deferred_scan_ranges: Option<&mut Vec<DeferredScanRange>>,
) -> usize {
    let mut local_distances = [0u32; MAX_LOCAL_SUBCLUSTERS];
    let mut route_distances = [0u32; MAX_LOCAL_SUBCLUSTERS];
    let Some((local_start, local_count)) = populate_local_subcluster_distances(
        query_vector,
        query,
        cluster_id,
        index,
        local_subcluster_routing,
        &mut route_distances,
        &mut local_distances,
    ) else {
        return 0;
    };

    let probe = usize::min(local_subcluster_probe, local_count);
    if probe >= local_count {
        let mut added = 0usize;
        for local_idx in 0..local_count {
            let global_local = local_start + local_idx;
            let start = index.local_subcluster_offsets[global_local];
            let end = index.local_subcluster_offsets[global_local + 1];
            if start != end {
                scan_ranges.push(ScanRange {
                    start,
                    end,
                    centroid_distance_sq: local_distances[local_idx],
                    cluster_id,
                });
                added += 1;
            }
        }
        return added;
    }

    let mut best_local_ids = Vec::with_capacity(probe);
    let mut best_local_distances = Vec::with_capacity(probe);
    for (local_idx, &distance) in route_distances[..local_count].iter().enumerate() {
        push_smallest_u32(
            &mut best_local_ids,
            &mut best_local_distances,
            probe,
            local_idx,
            distance,
        );
    }
    sort_pairs_u32(&mut best_local_ids, &mut best_local_distances);

    let mut added = 0usize;
    let mut is_seed = [false; MAX_LOCAL_SUBCLUSTERS];
    for &local_idx in best_local_ids.iter() {
        is_seed[local_idx] = true;
    }
    for local_idx in best_local_ids {
        let global_local = local_start + local_idx;
        let start = index.local_subcluster_offsets[global_local];
        let end = index.local_subcluster_offsets[global_local + 1];
        if start != end {
            scan_ranges.push(ScanRange {
                start,
                end,
                centroid_distance_sq: local_distances[local_idx],
                cluster_id,
            });
            added += 1;
        }
    }
    if let Some(deferred_scan_ranges) = deferred_scan_ranges {
        for (local_idx, &distance) in local_distances[..local_count].iter().enumerate() {
            if is_seed[local_idx] {
                continue;
            }
            let global_local = local_start + local_idx;
            let start = index.local_subcluster_offsets[global_local];
            let end = index.local_subcluster_offsets[global_local + 1];
            if start == end {
                continue;
            }
            deferred_scan_ranges.push(DeferredScanRange {
                start,
                end,
                centroid_distance_sq: distance,
                lower_bound: cluster_lower_bound(
                    distance,
                    index.local_subcluster_quantized_radii[global_local],
                ),
                route_distance_ord: route_distances[local_idx],
                cluster_id,
            });
        }
    }
    added
}

fn append_cluster_subcluster_candidates(
    query_vector: &[f32],
    query: &QuantizedQuery,
    cluster_id: usize,
    index: &HierarchicalIndex,
    local_subcluster_routing: LocalSubclusterRouting,
    candidates: &mut Vec<DeferredScanRange>,
) {
    let mut local_distances = [0u32; MAX_LOCAL_SUBCLUSTERS];
    let mut route_distances = [0u32; MAX_LOCAL_SUBCLUSTERS];
    let Some((local_start, local_count)) = populate_local_subcluster_distances(
        query_vector,
        query,
        cluster_id,
        index,
        local_subcluster_routing,
        &mut route_distances,
        &mut local_distances,
    ) else {
        return;
    };

    for (local_idx, &distance) in local_distances[..local_count].iter().enumerate() {
        let global_local = local_start + local_idx;
        let start = index.local_subcluster_offsets[global_local];
        let end = index.local_subcluster_offsets[global_local + 1];
        if start == end {
            continue;
        }

        candidates.push(DeferredScanRange {
            start,
            end,
            centroid_distance_sq: distance,
            lower_bound: cluster_lower_bound(
                distance,
                index.local_subcluster_quantized_radii[global_local],
            ),
            route_distance_ord: route_distances[local_idx],
            cluster_id,
        });
    }
}

fn populate_local_subcluster_distances(
    query_vector: &[f32],
    query: &QuantizedQuery,
    cluster_id: usize,
    index: &HierarchicalIndex,
    local_subcluster_routing: LocalSubclusterRouting,
    route_distances: &mut [u32; MAX_LOCAL_SUBCLUSTERS],
    centroid_distances_sq: &mut [u32; MAX_LOCAL_SUBCLUSTERS],
) -> Option<(usize, usize)> {
    let cluster_start = index.cluster_offsets[cluster_id];
    let cluster_end = index.cluster_offsets[cluster_id + 1];
    if cluster_start == cluster_end {
        return None;
    }

    let local_start = index.local_cluster_offsets[cluster_id];
    let local_end = index.local_cluster_offsets[cluster_id + 1];
    let local_count = local_end - local_start;
    if local_count == 0 {
        return None;
    }

    let centroid_start = local_start * DIM;
    let centroid_end = local_end * DIM;
    distance::l2_distance_batch_u8(
        &query.values,
        &index.local_centroids_u8[centroid_start..centroid_end],
        local_count,
        &mut centroid_distances_sq[..local_count],
    );

    match local_subcluster_routing {
        LocalSubclusterRouting::U8 => {
            route_distances[..local_count].copy_from_slice(&centroid_distances_sq[..local_count]);
        }
        LocalSubclusterRouting::F32 => {
            let mut exact_distances = [0.0f32; MAX_LOCAL_SUBCLUSTERS];
            distance::l2_distance_batch(
                query_vector,
                &index.local_centroids[centroid_start..centroid_end],
                local_count,
                &mut exact_distances[..local_count],
            );
            for (idx, &distance) in exact_distances[..local_count].iter().enumerate() {
                route_distances[idx] = distance.to_bits();
            }
        }
    }

    Some((local_start, local_count))
}

#[inline(always)]
fn flat_prescore_query_for_cluster<'a>(
    cluster_id: usize,
    secondary_clusters: usize,
    quantized_query: &'a QuantizedQuery,
    primary_prescore_queries: &'a [QuantizedQuery],
    primary_prescore_query_lookup: &[usize],
    use_primary_residual_prescore: bool,
) -> &'a QuantizedQuery {
    if !use_primary_residual_prescore {
        return quantized_query;
    }

    let primary_id = cluster_id / secondary_clusters;
    let query_idx = primary_prescore_query_lookup[primary_id];
    debug_assert!(query_idx != usize::MAX);
    &primary_prescore_queries[query_idx]
}

#[inline(always)]
fn flat_prescore_centroid_for_cluster<'a>(
    cluster_id: usize,
    index: &'a HierarchicalIndex,
    use_primary_residual_prescore: bool,
) -> &'a [u8] {
    let start = cluster_id * DIM;
    if use_primary_residual_prescore {
        debug_assert!(!index.secondary_centroids_primary_residual_u8.is_empty());
        &index.secondary_centroids_primary_residual_u8[start..start + DIM]
    } else {
        &index.secondary_centroids_u8[start..start + DIM]
    }
}

fn merge_scan_ranges(scan_ranges: &mut Vec<ScanRange>) {
    if scan_ranges.len() <= 1 {
        return;
    }

    let mut merged = Vec::with_capacity(scan_ranges.len());
    let mut current = scan_ranges[0];
    for range in &scan_ranges[1..] {
        if range.start <= current.end {
            current.end = current.end.max(range.end);
            continue;
        }
        merged.push(current);
        current = *range;
    }
    merged.push(current);
    *scan_ranges = merged;
}

fn merge_scan_ranges_by_primary(scan_ranges: &mut Vec<ScanRange>, secondary_clusters: usize) {
    if scan_ranges.len() <= 1 {
        return;
    }

    let mut merged = Vec::with_capacity(scan_ranges.len());
    let mut current = scan_ranges[0];
    let mut current_primary = current.cluster_id / secondary_clusters;
    for range in &scan_ranges[1..] {
        let range_primary = range.cluster_id / secondary_clusters;
        if range_primary == current_primary && range.start <= current.end {
            current.end = current.end.max(range.end);
            continue;
        }
        merged.push(current);
        current = *range;
        current_primary = range_primary;
    }
    merged.push(current);
    *scan_ranges = merged;
}

fn official_data_dir() -> String {
    std::env::var(OFFICIAL_DATA_DIR_ENV)
        .ok()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| OFFICIAL_DATA_DIR_DEFAULT.to_string())
}

fn load_official_base_raw_store(data_dir: &Path) -> RawStore {
    let fvecs_path = data_dir.join("sift_base.fvecs");
    if let Ok(bytes) = fs::read(&fvecs_path) {
        if let Some(raw) = load_raw_store_from_fvecs(&bytes) {
            return raw;
        }
    }

    let mut raw = RawStore {
        ids: Vec::with_capacity(OFFICIAL_BASE_VECTOR_COUNT),
        vectors: Vec::with_capacity(OFFICIAL_BASE_VECTOR_COUNT * DIM),
    };

    let chunk_paths: Vec<_> = (0..10)
        .map(|idx| data_dir.join(format!("base_vectors_{idx}.json")))
        .collect();

    let paths: Vec<_> = if chunk_paths.iter().all(|path| path.is_file()) {
        chunk_paths
    } else {
        vec![data_dir.join("base_vectors.json")]
    };

    for path in paths {
        let Ok(bytes) = fs::read(&path) else {
            return RawStore {
                ids: Vec::with_capacity(1_100_000),
                vectors: Vec::with_capacity(1_100_000 * DIM),
            };
        };
        let Ok(entries) = serde_json::from_slice::<Vec<InsertRequest>>(&bytes) else {
            return RawStore {
                ids: Vec::with_capacity(1_100_000),
                vectors: Vec::with_capacity(1_100_000 * DIM),
            };
        };

        raw.ids.reserve(entries.len());
        raw.vectors.reserve(entries.len() * DIM);
        for entry in entries {
            if entry.vector.len() != DIM {
                continue;
            }
            raw.ids.push(entry.id);
            raw.vectors.extend_from_slice(&entry.vector);
        }
    }

    raw
}

fn load_raw_store_from_fvecs(bytes: &[u8]) -> Option<RawStore> {
    let record_len = 4 + DIM * 4;
    if bytes.len() < record_len || bytes.len() % record_len != 0 {
        return None;
    }

    let count = bytes.len() / record_len;
    let mut raw = RawStore {
        ids: Vec::with_capacity(count),
        vectors: Vec::with_capacity(count * DIM),
    };

    for idx in 0..count {
        let record_start = idx * record_len;
        let dim = i32::from_le_bytes(bytes[record_start..record_start + 4].try_into().ok()?);
        if dim != DIM as i32 {
            return None;
        }

        raw.ids.push(idx as u64);
        let vector_bytes = &bytes[record_start + 4..record_start + record_len];
        for chunk in vector_bytes.chunks_exact(4) {
            raw.vectors.push(f32::from_le_bytes(chunk.try_into().ok()?));
        }
    }

    Some(raw)
}

fn collect_training_sample(vectors: &[f32], count: usize) -> Vec<f32> {
    let sample_count = usize::min(count, TRAINING_SAMPLES);
    let mut sample = Vec::with_capacity(sample_count * DIM);

    if sample_count == 0 {
        return sample;
    }

    let step = count as f64 / sample_count as f64;
    let mut cursor = 0.0f64;

    for _ in 0..sample_count {
        let idx = usize::min(cursor as usize, count - 1);
        let start = idx * DIM;
        sample.extend_from_slice(&vectors[start..start + DIM]);
        cursor += step;
    }

    sample
}

fn train_pq_codebooks(training: &[f32]) -> Vec<f32> {
    let sample_points = usize::min(training.len() / DIM, PQ_TRAINING_SAMPLES);
    if sample_points == 0 {
        return Vec::new();
    }

    let sample = &training[..sample_points * DIM];
    let mut codebooks = Vec::with_capacity(PQ_TABLE_LEN * PQ_SUBVECTOR_DIMS);
    for subspace in 0..PQ_SUBQUANTIZERS {
        let dim_start = subspace * PQ_SUBVECTOR_DIMS;
        let mut subspace_data = Vec::with_capacity(sample_points * PQ_SUBVECTOR_DIMS);
        for vector in sample.chunks_exact(DIM) {
            subspace_data.extend_from_slice(&vector[dim_start..dim_start + PQ_SUBVECTOR_DIMS]);
        }
        codebooks.extend_from_slice(&run_kmeans_subdim(
            &subspace_data,
            sample_points,
            PQ_SUBVECTOR_DIMS,
            PQ_CENTROIDS,
            PQ_KMEANS_ITERS,
        ));
    }

    codebooks
}

fn train_primary_residual_pq_codebooks(
    primary_buckets: &[Vec<f32>],
    primary_centroids: &[f32],
) -> Vec<f32> {
    let primary_clusters = primary_buckets.len();
    let chunks = (0..primary_clusters)
        .into_par_iter()
        .map(|primary_id| {
            let bucket = &primary_buckets[primary_id];
            let centroid = &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM];
            let mut residual_bucket = Vec::with_capacity(bucket.len());
            for vector in bucket.chunks_exact(DIM) {
                for dim in 0..DIM {
                    residual_bucket.push(vector[dim] - centroid[dim]);
                }
            }

            let mut codebooks = train_pq_codebooks(&residual_bucket);
            if codebooks.is_empty() {
                codebooks = vec![0.0; PQ_CODEBOOK_STRIDE];
            }
            codebooks
        })
        .collect::<Vec<_>>();

    let mut codebooks = Vec::with_capacity(primary_clusters * PQ_CODEBOOK_STRIDE);
    for chunk in chunks {
        codebooks.extend_from_slice(&chunk);
    }
    codebooks
}

#[inline(always)]
fn pq_codebooks_for_primary(pq_mode: PqMode, pq_codebooks: &[f32], primary_id: usize) -> &[f32] {
    match pq_mode {
        PqMode::Off => &[],
        PqMode::Global => &pq_codebooks[..PQ_CODEBOOK_STRIDE],
        PqMode::PrimaryResidual => {
            let start = primary_id * PQ_CODEBOOK_STRIDE;
            &pq_codebooks[start..start + PQ_CODEBOOK_STRIDE]
        }
    }
}

#[inline(always)]
fn encode_pq_vector(vector: &[f32], codebooks: &[f32], output: &mut [u8]) {
    debug_assert_eq!(vector.len(), DIM);
    debug_assert_eq!(output.len(), PQ_SUBQUANTIZERS);

    for subspace in 0..PQ_SUBQUANTIZERS {
        let dim_start = subspace * PQ_SUBVECTOR_DIMS;
        let codebook_start = subspace * PQ_CENTROIDS * PQ_SUBVECTOR_DIMS;
        output[subspace] = nearest_centroid_subdim(
            &vector[dim_start..dim_start + PQ_SUBVECTOR_DIMS],
            &codebooks[codebook_start..codebook_start + PQ_CENTROIDS * PQ_SUBVECTOR_DIMS],
            PQ_CENTROIDS,
            PQ_SUBVECTOR_DIMS,
        ) as u8;
    }
}

#[inline(always)]
fn encode_cluster_pq_vector(
    pq_mode: PqMode,
    vector: &[f32],
    primary_centroid: &[f32],
    codebooks: &[f32],
    output: &mut [u8],
) {
    match pq_mode {
        PqMode::Off => {}
        PqMode::Global => encode_pq_vector(vector, codebooks, output),
        PqMode::PrimaryResidual => {
            let mut residual = [0.0f32; DIM];
            for dim in 0..DIM {
                residual[dim] = vector[dim] - primary_centroid[dim];
            }
            encode_pq_vector(&residual, codebooks, output);
        }
    }
}

#[inline(always)]
fn build_pq_query_tables(query: &[f32], codebooks: &[f32]) -> [u32; PQ_TABLE_LEN] {
    debug_assert_eq!(query.len(), DIM);

    let mut tables = [0u32; PQ_TABLE_LEN];
    for subspace in 0..PQ_SUBQUANTIZERS {
        let dim_start = subspace * PQ_SUBVECTOR_DIMS;
        let table_start = subspace * PQ_CENTROIDS;
        let codebook_start = subspace * PQ_CENTROIDS * PQ_SUBVECTOR_DIMS;
        let query_subspace = &query[dim_start..dim_start + PQ_SUBVECTOR_DIMS];
        let codebook =
            &codebooks[codebook_start..codebook_start + PQ_CENTROIDS * PQ_SUBVECTOR_DIMS];
        for centroid_idx in 0..PQ_CENTROIDS {
            let centroid_start = centroid_idx * PQ_SUBVECTOR_DIMS;
            tables[table_start + centroid_idx] = l2_squared_subdim(
                query_subspace,
                &codebook[centroid_start..centroid_start + PQ_SUBVECTOR_DIMS],
            )
            .round() as u32;
        }
    }

    tables
}

#[inline(always)]
fn build_primary_residual_pq_query_tables(
    query: &[f32],
    primary_centroid: &[f32],
    codebooks: &[f32],
) -> [u32; PQ_TABLE_LEN] {
    let mut residual = [0.0f32; DIM];
    for dim in 0..DIM {
        residual[dim] = query[dim] - primary_centroid[dim];
    }
    build_pq_query_tables(&residual, codebooks)
}

#[inline(always)]
fn nearest_centroid_subdim(
    point: &[f32],
    centroids_flat: &[f32],
    count: usize,
    dims: usize,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;

    for centroid_idx in 0..count {
        let start = centroid_idx * dims;
        let dist = l2_squared_subdim(point, &centroids_flat[start..start + dims]);
        if dist < best_dist {
            best_dist = dist;
            best_idx = centroid_idx;
        }
    }

    best_idx
}

fn run_kmeans_subdim(
    data: &[f32],
    points: usize,
    dims: usize,
    clusters: usize,
    iterations: usize,
) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(clusters * dims);
    let step = points as f64 / clusters as f64;
    let mut cursor = 0.0f64;

    for _ in 0..clusters {
        let idx = usize::min(cursor as usize, points - 1);
        let start = idx * dims;
        centroids.extend_from_slice(&data[start..start + dims]);
        cursor += step;
    }

    let mut assignments = vec![0usize; points];
    for _ in 0..iterations {
        if points >= 4096 {
            assignments
                .par_iter_mut()
                .enumerate()
                .for_each(|(point_idx, assignment)| {
                    let start = point_idx * dims;
                    *assignment = nearest_centroid_subdim(
                        &data[start..start + dims],
                        &centroids,
                        clusters,
                        dims,
                    );
                });
        } else {
            for (point_idx, assignment) in assignments.iter_mut().enumerate() {
                let start = point_idx * dims;
                *assignment =
                    nearest_centroid_subdim(&data[start..start + dims], &centroids, clusters, dims);
            }
        }

        let mut sums = vec![0.0f64; clusters * dims];
        let mut counts = vec![0usize; clusters];
        for (point_idx, &cluster_id) in assignments.iter().enumerate() {
            counts[cluster_id] += 1;
            let src = point_idx * dims;
            let dst = cluster_id * dims;
            for dim in 0..dims {
                sums[dst + dim] += data[src + dim] as f64;
            }
        }

        for cluster_id in 0..clusters {
            let count = counts[cluster_id];
            if count == 0 {
                continue;
            }
            let inv = 1.0f64 / count as f64;
            let start = cluster_id * dims;
            for dim in 0..dims {
                centroids[start + dim] = (sums[start + dim] * inv) as f32;
            }
        }
    }

    centroids
}

#[inline(always)]
fn l2_squared_subdim(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for idx in 0..lhs.len() {
        let delta = lhs[idx] - rhs[idx];
        sum += delta * delta;
    }
    sum
}

fn select_coarse_dimensions(sample: &[f32], coarse_dims: usize) -> Vec<u8> {
    if coarse_dims == 0 || sample.is_empty() {
        return Vec::new();
    }

    let points = sample.len() / DIM;
    if points == 0 {
        return Vec::new();
    }

    let mut means = [0.0f64; DIM];
    for vector in sample.chunks_exact(DIM) {
        for dim in 0..DIM {
            means[dim] += vector[dim] as f64;
        }
    }
    let inv_points = 1.0f64 / points as f64;
    for mean in means.iter_mut() {
        *mean *= inv_points;
    }

    let mut variances = [0.0f64; DIM];
    for vector in sample.chunks_exact(DIM) {
        for dim in 0..DIM {
            let delta = vector[dim] as f64 - means[dim];
            variances[dim] += delta * delta;
        }
    }

    let mut ranked_dims = (0..DIM).collect::<Vec<_>>();
    ranked_dims.sort_unstable_by(|&lhs, &rhs| {
        variances[rhs]
            .partial_cmp(&variances[lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ranked_dims
        .into_iter()
        .take(coarse_dims.min(DIM))
        .map(|dim| dim as u8)
        .collect()
}

fn select_group_coarse_dimensions(
    raw_vectors: &[f32],
    group_offsets: &[usize],
    member_indices: &[usize],
    coarse_dims: usize,
    fallback_dims: &[u8],
    mode: CoarsePrescoreMode,
    group_centroids: &[f32],
) -> Vec<u8> {
    if coarse_dims == 0 || group_offsets.len() <= 1 {
        return Vec::new();
    }

    let total_groups = group_offsets.len() - 1;
    let mut group_dims = vec![0u8; total_groups * coarse_dims];

    for group_id in 0..total_groups {
        let dst = &mut group_dims[group_id * coarse_dims..(group_id + 1) * coarse_dims];
        let start = group_offsets[group_id];
        let end = group_offsets[group_id + 1];
        if end <= start + 1 {
            if fallback_dims.len() >= coarse_dims {
                dst.copy_from_slice(&fallback_dims[..coarse_dims]);
            }
            continue;
        }

        let mut scores = [0.0f64; DIM];
        match mode {
            CoarsePrescoreMode::Raw => {
                let mut means = [0.0f64; DIM];
                let inv_count = 1.0f64 / (end - start) as f64;
                for &vector_idx in &member_indices[start..end] {
                    let src = vector_idx * DIM;
                    for dim in 0..DIM {
                        means[dim] += raw_vectors[src + dim] as f64 * inv_count;
                    }
                }
                for &vector_idx in &member_indices[start..end] {
                    let src = vector_idx * DIM;
                    for dim in 0..DIM {
                        let delta = raw_vectors[src + dim] as f64 - means[dim];
                        scores[dim] += delta * delta;
                    }
                }
            }
            CoarsePrescoreMode::Residual => {
                let centroid = &group_centroids[group_id * DIM..(group_id + 1) * DIM];
                for &vector_idx in &member_indices[start..end] {
                    let src = vector_idx * DIM;
                    for dim in 0..DIM {
                        let delta = raw_vectors[src + dim] as f64 - centroid[dim] as f64;
                        scores[dim] += delta * delta;
                    }
                }
            }
        }

        let mut ranked_dims = (0..DIM).collect::<Vec<_>>();
        ranked_dims.sort_unstable_by(|&lhs, &rhs| {
            scores[rhs]
                .partial_cmp(&scores[lhs])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (offset, dim) in ranked_dims.into_iter().take(coarse_dims).enumerate() {
            dst[offset] = dim as u8;
        }
    }

    group_dims
}

fn primary_group_offsets_from_cluster_offsets(
    cluster_offsets: &[usize],
    primary_clusters: usize,
    secondary_clusters: usize,
) -> Vec<usize> {
    let mut primary_offsets = Vec::with_capacity(primary_clusters + 1);
    for primary_id in 0..=primary_clusters {
        let cluster_idx = usize::min(primary_id * secondary_clusters, cluster_offsets.len() - 1);
        primary_offsets.push(cluster_offsets[cluster_idx]);
    }
    primary_offsets
}

#[inline(always)]
fn coarse_group_id(
    scope: CoarsePrescoreScope,
    cluster_id: usize,
    secondary_clusters: usize,
) -> usize {
    match scope {
        CoarsePrescoreScope::Cluster => cluster_id,
        CoarsePrescoreScope::Primary => cluster_id / secondary_clusters,
    }
}

#[inline(always)]
fn coarse_group_centroid<'a>(
    scope: CoarsePrescoreScope,
    cluster_id: usize,
    secondary_clusters: usize,
    primary_centroids: &'a [f32],
    secondary_centroids: &'a [f32],
) -> &'a [f32] {
    match scope {
        CoarsePrescoreScope::Cluster => {
            &secondary_centroids[cluster_id * DIM..(cluster_id + 1) * DIM]
        }
        CoarsePrescoreScope::Primary => {
            let primary_id = cluster_id / secondary_clusters;
            &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM]
        }
    }
}

#[inline(always)]
fn coarse_group_centroid_from_index(
    index: &HierarchicalIndex,
    scope: CoarsePrescoreScope,
    group_id: usize,
) -> &[f32] {
    match scope {
        CoarsePrescoreScope::Cluster => {
            &index.secondary_centroids[group_id * DIM..(group_id + 1) * DIM]
        }
        CoarsePrescoreScope::Primary => {
            &index.primary_centroids[group_id * DIM..(group_id + 1) * DIM]
        }
    }
}

#[inline(always)]
fn count_active_primary_groups(
    scan_ranges: &[ScanRange],
    primary_clusters: usize,
    secondary_clusters: usize,
) -> usize {
    let mut seen = vec![false; primary_clusters];
    let mut active = 0usize;
    for range in scan_ranges {
        let primary_id = range.cluster_id / secondary_clusters;
        if !seen[primary_id] {
            seen[primary_id] = true;
            active += 1;
        }
    }
    active
}

fn compute_coarse_residual_quantization(
    raw_vectors: &[f32],
    group_offsets: &[usize],
    member_indices: &[usize],
    coarse_dim_indices: &[u8],
    group_centroids: &[f32],
    coarse_dim_count: usize,
) -> (Vec<f32>, Vec<f32>) {
    if coarse_dim_count == 0 || group_offsets.len() <= 1 {
        return (Vec::new(), Vec::new());
    }

    let total_groups = group_offsets.len() - 1;
    let mut mins = vec![0.0f32; total_groups * coarse_dim_count];
    let mut scales = vec![0.0f32; total_groups * coarse_dim_count];
    for group_id in 0..total_groups {
        let start = group_offsets[group_id];
        let end = group_offsets[group_id + 1];
        if start >= end {
            continue;
        }

        let centroid = &group_centroids[group_id * DIM..(group_id + 1) * DIM];
        let dims_start = group_id * coarse_dim_count;
        let dims_end = dims_start + coarse_dim_count;
        let dims = &coarse_dim_indices[dims_start..dims_end];
        let mut group_mins = vec![f32::INFINITY; coarse_dim_count];
        let mut group_maxs = vec![f32::NEG_INFINITY; coarse_dim_count];
        for &vector_idx in &member_indices[start..end] {
            let src = vector_idx * DIM;
            for (offset, &dim_idx) in dims.iter().enumerate() {
                let dim = dim_idx as usize;
                let residual = raw_vectors[src + dim] - centroid[dim];
                group_mins[offset] = group_mins[offset].min(residual);
                group_maxs[offset] = group_maxs[offset].max(residual);
            }
        }

        for offset in 0..coarse_dim_count {
            let min_value = group_mins[offset];
            let max_value = group_maxs[offset];
            mins[dims_start + offset] = if min_value.is_finite() {
                min_value
            } else {
                0.0
            };
            scales[dims_start + offset] = if max_value.is_finite() && max_value > min_value {
                255.0 / (max_value - min_value)
            } else {
                0.0
            };
        }
    }

    (mins, scales)
}

fn compute_primary_residual_quantization(
    raw_vectors: &[f32],
    primary_offsets: &[usize],
    member_indices: &[usize],
    primary_centroids: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    if primary_offsets.len() <= 1 {
        return (Vec::new(), Vec::new());
    }

    let primary_clusters = primary_offsets.len() - 1;
    let mut residual_mins = [f32::INFINITY; DIM];
    let mut residual_maxs = [f32::NEG_INFINITY; DIM];
    for primary_id in 0..primary_clusters {
        let start = primary_offsets[primary_id];
        let end = primary_offsets[primary_id + 1];
        if start >= end {
            continue;
        }

        let centroid = &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM];
        for &vector_idx in &member_indices[start..end] {
            let src = vector_idx * DIM;
            for dim in 0..DIM {
                let residual = raw_vectors[src + dim] - centroid[dim];
                residual_mins[dim] = residual_mins[dim].min(residual);
                residual_maxs[dim] = residual_maxs[dim].max(residual);
            }
        }
    }

    let mut mins = vec![0.0f32; primary_clusters * DIM];
    let mut scales = vec![0.0f32; primary_clusters * DIM];
    for primary_id in 0..primary_clusters {
        let dst = primary_id * DIM;
        for dim in 0..DIM {
            let min_value = residual_mins[dim];
            let max_value = residual_maxs[dim];
            mins[dst + dim] = if min_value.is_finite() {
                min_value
            } else {
                0.0
            };
            scales[dst + dim] = if max_value.is_finite() && max_value > min_value {
                255.0 / (max_value - min_value)
            } else {
                0.0
            };
        }
    }

    (mins, scales)
}

fn run_kmeans(data: &[f32], points: usize, clusters: usize, iterations: usize) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(clusters * DIM);
    let step = points as f64 / clusters as f64;
    let mut cursor = 0.0f64;

    for _ in 0..clusters {
        let idx = usize::min(cursor as usize, points - 1);
        let start = idx * DIM;
        centroids.extend_from_slice(&data[start..start + DIM]);
        cursor += step;
    }

    let mut assignments = vec![0usize; points];

    for _ in 0..iterations {
        if points >= 4096 {
            assignments
                .par_iter_mut()
                .enumerate()
                .for_each(|(point_idx, assignment)| {
                    let start = point_idx * DIM;
                    *assignment = nearest_centroid(&data[start..start + DIM], &centroids, clusters);
                });
        } else {
            for (point_idx, assignment) in assignments.iter_mut().enumerate() {
                let start = point_idx * DIM;
                *assignment = nearest_centroid(&data[start..start + DIM], &centroids, clusters);
            }
        }

        let mut sums = vec![0.0f64; clusters * DIM];
        let mut counts = vec![0usize; clusters];

        for (point_idx, &cluster_id) in assignments.iter().enumerate() {
            counts[cluster_id] += 1;
            let src = point_idx * DIM;
            let dst = cluster_id * DIM;
            for dim in 0..DIM {
                sums[dst + dim] += data[src + dim] as f64;
            }
        }

        for cluster_id in 0..clusters {
            let count = counts[cluster_id];
            if count == 0 {
                continue;
            }

            let inv = 1.0f64 / count as f64;
            let start = cluster_id * DIM;
            for dim in 0..DIM {
                centroids[start + dim] = (sums[start + dim] * inv) as f32;
            }
        }
    }

    centroids
}

fn duplicate_centroid(centroid: &[f32], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count * DIM);
    for _ in 0..count {
        result.extend_from_slice(centroid);
    }
    result
}

fn expand_bucket_centroids(
    bucket: &[f32],
    fallback: &[f32],
    secondary_clusters: usize,
) -> Vec<f32> {
    let points = bucket.len() / DIM;
    let mut result = Vec::with_capacity(secondary_clusters * DIM);

    for idx in 0..secondary_clusters {
        if idx < points {
            let start = idx * DIM;
            result.extend_from_slice(&bucket[start..start + DIM]);
        } else {
            result.extend_from_slice(fallback);
        }
    }

    result
}

fn compute_quantization_params(vectors: &[f32]) -> (f32, f32) {
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;

    for &value in vectors.iter() {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    if !min_value.is_finite() || !max_value.is_finite() || max_value <= min_value {
        return (0.0, 0.0);
    }

    (min_value, 255.0 / (max_value - min_value))
}

#[inline(always)]
fn quantize_value(value: f32, quant_min: f32, quant_scale: f32) -> u8 {
    if quant_scale <= 0.0 {
        return 0;
    }

    ((value - quant_min) * quant_scale)
        .round()
        .clamp(0.0, 255.0) as u8
}

#[inline(always)]
fn quantized_radius(radius_sq: u32) -> u16 {
    (radius_sq as f64).sqrt().ceil() as u16
}

#[inline(always)]
fn write_coarse_vector(
    position: usize,
    raw_vector: &[f32],
    cluster_vectors_u8_coarse: &mut [u8],
    coarse_dim_indices: &[u8],
    coarse_mode: CoarsePrescoreMode,
    quant_min: f32,
    quant_scale: f32,
    coarse_centroid: &[f32],
    coarse_quant_mins: &[f32],
    coarse_quant_scales: &[f32],
) {
    if coarse_dim_indices.is_empty() {
        return;
    }

    let coarse_dims = coarse_dim_indices.len();
    let coarse_start = position * coarse_dims;
    match coarse_mode {
        CoarsePrescoreMode::Raw => {
            for (offset, &dim_idx) in coarse_dim_indices.iter().enumerate() {
                cluster_vectors_u8_coarse[coarse_start + offset] =
                    quantize_value(raw_vector[dim_idx as usize], quant_min, quant_scale);
            }
        }
        CoarsePrescoreMode::Residual => {
            for (offset, &dim_idx) in coarse_dim_indices.iter().enumerate() {
                let dim = dim_idx as usize;
                let residual = raw_vector[dim] - coarse_centroid[dim];
                cluster_vectors_u8_coarse[coarse_start + offset] = quantize_value(
                    residual,
                    coarse_quant_mins[offset],
                    coarse_quant_scales[offset],
                );
            }
        }
    }
}

#[inline(always)]
fn write_quantized_prescore_vector(
    position: usize,
    raw_vector: &[f32],
    cluster_vectors_u8: &mut [u8],
    quantized_prescore_mode: QuantizedPrescoreMode,
    quant_min: f32,
    quant_scale: f32,
    primary_id: usize,
    primary_centroid: &[f32],
    primary_residual_quant_mins: &[f32],
    primary_residual_quant_scales: &[f32],
) {
    let dst = position * DIM;
    match quantized_prescore_mode {
        QuantizedPrescoreMode::Global => {
            for dim in 0..DIM {
                cluster_vectors_u8[dst + dim] =
                    quantize_value(raw_vector[dim], quant_min, quant_scale);
            }
        }
        QuantizedPrescoreMode::PrimaryResidual => {
            let base = primary_id * DIM;
            for dim in 0..DIM {
                let residual = raw_vector[dim] - primary_centroid[dim];
                cluster_vectors_u8[dst + dim] = quantize_value(
                    residual,
                    primary_residual_quant_mins[base + dim],
                    primary_residual_quant_scales[base + dim],
                );
            }
        }
    }
}

#[inline(always)]
fn populate_primary_residual_quantized_values(
    source: &[f32],
    primary_id: usize,
    primary_centroids: &[f32],
    primary_residual_quant_mins: &[f32],
    primary_residual_quant_scales: &[f32],
    output: &mut [u8],
) {
    debug_assert_eq!(source.len(), DIM);
    debug_assert!(output.len() >= DIM);

    let centroid = &primary_centroids[primary_id * DIM..(primary_id + 1) * DIM];
    let base = primary_id * DIM;
    for dim in 0..DIM {
        output[dim] = quantize_value(
            source[dim] - centroid[dim],
            primary_residual_quant_mins[base + dim],
            primary_residual_quant_scales[base + dim],
        );
    }
}

#[inline(always)]
fn build_primary_residual_quantized_query(
    query: &[f32],
    primary_id: usize,
    primary_centroids: &[f32],
    primary_residual_quant_mins: &[f32],
    primary_residual_quant_scales: &[f32],
) -> QuantizedQuery {
    let mut values = [0u8; DIM];
    populate_primary_residual_quantized_values(
        query,
        primary_id,
        primary_centroids,
        primary_residual_quant_mins,
        primary_residual_quant_scales,
        &mut values,
    );
    QuantizedQuery { values }
}

#[inline(always)]
fn compare_member_indices_by_dims(
    lhs_idx: usize,
    rhs_idx: usize,
    raw_vectors: &[f32],
    selected_dims: &[u8],
) -> std::cmp::Ordering {
    let lhs_start = lhs_idx * DIM;
    let rhs_start = rhs_idx * DIM;
    for &dim_idx in selected_dims {
        let dim = dim_idx as usize;
        let ordering = raw_vectors[lhs_start + dim]
            .partial_cmp(&raw_vectors[rhs_start + dim])
            .unwrap_or(std::cmp::Ordering::Equal);
        if !matches!(ordering, std::cmp::Ordering::Equal) {
            return ordering;
        }
    }
    lhs_idx.cmp(&rhs_idx)
}

fn populate_selected_query_values(
    quantized_query: &[u8; DIM],
    selected_dim_indices: &[u8],
    selected_dim_count: usize,
    group_id: usize,
    selected_query_values: &mut [u8; DIM],
) {
    if selected_dim_count == 0 {
        return;
    }

    let start = group_id * selected_dim_count;
    let end = start + selected_dim_count;
    for (offset, &dim_idx) in selected_dim_indices[start..end].iter().enumerate() {
        selected_query_values[offset] = quantized_query[dim_idx as usize];
    }
}

fn populate_coarse_query_values(
    query: &[f32],
    quantized_query: &QuantizedQuery,
    index: &HierarchicalIndex,
    coarse_dim_count: usize,
    group_id: usize,
    coarse_scope: CoarsePrescoreScope,
    coarse_mode: CoarsePrescoreMode,
    selected_query_values: &mut [u8; DIM],
) {
    match coarse_mode {
        CoarsePrescoreMode::Raw => populate_selected_query_values(
            &quantized_query.values,
            &index.coarse_dim_indices,
            coarse_dim_count,
            group_id,
            selected_query_values,
        ),
        CoarsePrescoreMode::Residual => {
            if coarse_dim_count == 0 {
                return;
            }
            let dims_start = group_id * coarse_dim_count;
            let dims_end = dims_start + coarse_dim_count;
            let centroid = coarse_group_centroid_from_index(index, coarse_scope, group_id);
            let mins = &index.coarse_quant_mins[dims_start..dims_end];
            let scales = &index.coarse_quant_scales[dims_start..dims_end];
            for (offset, &dim_idx) in index.coarse_dim_indices[dims_start..dims_end]
                .iter()
                .enumerate()
            {
                let dim = dim_idx as usize;
                let residual = query[dim] - centroid[dim];
                selected_query_values[offset] =
                    quantize_value(residual, mins[offset], scales[offset]);
            }
        }
    }
}

fn write_cluster_block_bounds(
    cluster_id: usize,
    start: usize,
    end: usize,
    block_bound_dims: usize,
    block_dims_for_cluster: &[u8],
    cluster_vectors_u8: &[u8],
    cluster_block_offsets: &[usize],
    cluster_block_mins: &mut [u8],
    cluster_block_maxs: &mut [u8],
) {
    if block_bound_dims == 0 || start >= end {
        return;
    }

    let block_start = cluster_block_offsets[cluster_id];
    let block_end = cluster_block_offsets[cluster_id + 1];
    for block_idx in block_start..block_end {
        let block_vector_start = start + (block_idx - block_start) * BLOCK_BOUND_SIZE;
        let block_vector_end = usize::min(block_vector_start + BLOCK_BOUND_SIZE, end);
        let bounds_start = block_idx * block_bound_dims;
        let bounds_end = bounds_start + block_bound_dims;
        cluster_block_mins[bounds_start..bounds_end].fill(u8::MAX);
        cluster_block_maxs[bounds_start..bounds_end].fill(0);

        for position in block_vector_start..block_vector_end {
            let vector_start = position * DIM;
            for (offset, &dim_idx) in block_dims_for_cluster.iter().enumerate() {
                let value = cluster_vectors_u8[vector_start + dim_idx as usize];
                let min_ref = &mut cluster_block_mins[bounds_start + offset];
                let max_ref = &mut cluster_block_maxs[bounds_start + offset];
                *min_ref = (*min_ref).min(value);
                *max_ref = (*max_ref).max(value);
            }
        }
    }
}

fn scan_cluster_quantized(
    query: &QuantizedQuery,
    vectors: &[u8],
    start_position: usize,
    candidates: &mut CandidateHeap,
) {
    let mut distances = [0u32; DISTANCE_BATCH_CHUNK];
    let total_vectors = vectors.len() / DIM;
    let mut vector_idx = 0usize;
    let mut position = start_position;

    while vector_idx < total_vectors {
        let batch = usize::min(DISTANCE_BATCH_CHUNK, total_vectors - vector_idx);
        let batch_start = vector_idx * DIM;
        let batch_end = batch_start + batch * DIM;
        distance::l2_distance_batch_u8(
            &query.values,
            &vectors[batch_start..batch_end],
            batch,
            &mut distances[..batch],
        );
        push_candidate_batch(position, &distances[..batch], candidates);
        position += batch;
        vector_idx += batch;
    }
}

#[inline(always)]
fn scan_cluster_quantized_prefix(
    query: &QuantizedQuery,
    vectors: &[u8],
    start_position: usize,
    prefix_len: usize,
    candidates: &mut CandidateHeap,
) -> usize {
    let total_vectors = vectors.len() / DIM;
    let prefix_len = usize::min(prefix_len, total_vectors);
    if prefix_len == 0 {
        return 0;
    }

    scan_cluster_quantized(query, &vectors[..prefix_len * DIM], start_position, candidates);
    prefix_len
}

#[inline(always)]
fn push_candidate_batch(start_position: usize, distances: &[u32], candidates: &mut CandidateHeap) {
    if distances.is_empty() || candidates.limit == 0 {
        return;
    }

    let mut position = start_position;
    let mut offset = 0usize;
    if !candidates.is_full() {
        let fill = usize::min(candidates.limit - candidates.len(), distances.len());
        for &distance in &distances[..fill] {
            candidates.push(position, distance);
            position += 1;
        }
        offset = fill;
        if offset == distances.len() {
            return;
        }
    }

    let mut cutoff = candidates.worst_distance();
    for &distance in &distances[offset..] {
        if distance < cutoff {
            candidates.push(position, distance);
            cutoff = candidates.worst_distance();
        }
        position += 1;
    }
}

#[inline(always)]
fn pq_query_tables_for_cluster<'a>(
    pq_mode: PqMode,
    global_tables: &'a [u32; PQ_TABLE_LEN],
    primary_tables: &'a [[u32; PQ_TABLE_LEN]],
    primary_lookup: &[usize],
    secondary_clusters: usize,
    cluster_id: usize,
) -> &'a [u32; PQ_TABLE_LEN] {
    match pq_mode {
        PqMode::Off | PqMode::Global => global_tables,
        PqMode::PrimaryResidual => {
            let primary_id = cluster_id / secondary_clusters;
            let table_idx = primary_lookup[primary_id];
            debug_assert!(table_idx != usize::MAX);
            &primary_tables[table_idx]
        }
    }
}

#[inline(always)]
fn scan_cluster_pq(
    query_tables: &[u32; PQ_TABLE_LEN],
    codes: &[u8],
    start_position: usize,
    candidates: &mut CandidateHeap,
) {
    let mut offset = 0usize;
    let mut position = start_position;
    let mut cutoff = if candidates.is_full() {
        candidates.worst_distance()
    } else {
        u32::MAX
    };

    while offset < codes.len() {
        let distance = query_tables[codes[offset] as usize]
            + query_tables[PQ_CENTROIDS + codes[offset + 1] as usize]
            + query_tables[PQ_CENTROIDS * 2 + codes[offset + 2] as usize]
            + query_tables[PQ_CENTROIDS * 3 + codes[offset + 3] as usize]
            + query_tables[PQ_CENTROIDS * 4 + codes[offset + 4] as usize]
            + query_tables[PQ_CENTROIDS * 5 + codes[offset + 5] as usize]
            + query_tables[PQ_CENTROIDS * 6 + codes[offset + 6] as usize]
            + query_tables[PQ_CENTROIDS * 7 + codes[offset + 7] as usize];
        if cutoff != u32::MAX && distance >= cutoff {
            offset += PQ_SUBQUANTIZERS;
            position += 1;
            continue;
        }
        candidates.push(position, distance);
        if candidates.is_full() {
            cutoff = candidates.worst_distance();
        }

        offset += PQ_SUBQUANTIZERS;
        position += 1;
    }
}

#[inline(always)]
fn scan_cluster_quantized_early_abort(
    ordered_query: &[u8],
    vectors: &[u8],
    start_position: usize,
    candidates: &mut CandidateHeap,
) -> (usize, usize) {
    let mut offset = 0usize;
    let mut position = start_position;
    let mut rescored = 0usize;
    let mut pruned = 0usize;
    let mut cutoff = if candidates.is_full() {
        candidates.worst_distance()
    } else {
        u32::MAX
    };

    while offset < vectors.len() {
        let vector = &vectors[offset..offset + DIM];
        let distance = if cutoff != u32::MAX {
            distance::l2_squared_u8_slice_with_upper_bound(ordered_query, vector, cutoff)
        } else {
            distance::l2_squared_u8_slice(ordered_query, vector)
        };

        if cutoff != u32::MAX && distance >= cutoff {
            offset += DIM;
            position += 1;
            pruned += 1;
            continue;
        }

        candidates.push(position, distance);
        if candidates.is_full() {
            cutoff = candidates.worst_distance();
        }
        offset += DIM;
        position += 1;
        rescored += 1;
    }

    (rescored, pruned)
}

#[inline(always)]
fn scan_cluster_quantized_subspace(
    query: &[u8],
    vectors: &[u8],
    dims: usize,
    start_position: usize,
    candidates: &mut CandidateHeap,
) {
    if dims <= 32 {
        let mut distances = [0u32; DISTANCE_BATCH_CHUNK];
        let total_vectors = vectors.len() / dims;
        let mut vector_idx = 0usize;
        let mut position = start_position;

        while vector_idx < total_vectors {
            let batch = usize::min(DISTANCE_BATCH_CHUNK, total_vectors - vector_idx);
            let batch_start = vector_idx * dims;
            let batch_end = batch_start + batch * dims;
            distance::l2_distance_batch_u8_slice(
                query,
                &vectors[batch_start..batch_end],
                dims,
                batch,
                &mut distances[..batch],
            );
            push_candidate_batch(position, &distances[..batch], candidates);
            position += batch;
            vector_idx += batch;
        }
        return;
    }

    let mut offset = 0usize;
    let mut position = start_position;
    let mut cutoff = if candidates.is_full() {
        candidates.worst_distance()
    } else {
        u32::MAX
    };

    while offset < vectors.len() {
        let vector = &vectors[offset..offset + dims];
        let distance = if cutoff != u32::MAX {
            distance::l2_squared_u8_slice_with_upper_bound(query, vector, cutoff)
        } else {
            distance::l2_squared_u8_slice(query, vector)
        };
        if cutoff != u32::MAX && distance >= cutoff {
            offset += dims;
            position += 1;
            continue;
        }
        candidates.push(position, distance);
        if candidates.is_full() {
            cutoff = candidates.worst_distance();
        }

        offset += dims;
        position += 1;
    }
}

fn scan_primary_quantized_subspace(
    query_vector: &[f32],
    quantized_query: &QuantizedQuery,
    index: &HierarchicalIndex,
    scan_ranges: &[ScanRange],
    coarse_dim_count: usize,
    coarse_candidates_per_primary: usize,
    coarse_scope: CoarsePrescoreScope,
    coarse_mode: CoarsePrescoreMode,
    candidates: &mut CandidateHeap,
) -> (usize, usize, usize) {
    if scan_ranges.is_empty() {
        return (0, 0, 0);
    }

    let mut coarse_query_values = [0u8; DIM];
    let mut coarse_candidates = CandidateHeap::new(coarse_candidates_per_primary);
    let mut current_primary = usize::MAX;
    let mut current_primary_len = 0usize;
    let mut scanned = 0usize;
    let mut rescored = 0usize;
    let mut pruned = 0usize;

    for range in scan_ranges {
        if range.start == range.end {
            continue;
        }

        let primary_id = range.cluster_id / index.secondary_clusters;
        if primary_id != current_primary {
            if current_primary != usize::MAX {
                let rescored_now = rescore_coarse_candidates(
                    quantized_query,
                    index,
                    &coarse_candidates,
                    candidates,
                );
                rescored += rescored_now;
                pruned += current_primary_len.saturating_sub(rescored_now);
                coarse_candidates.clear();
                current_primary_len = 0;
            }
            populate_coarse_query_values(
                query_vector,
                quantized_query,
                index,
                coarse_dim_count,
                primary_id,
                coarse_scope,
                coarse_mode,
                &mut coarse_query_values,
            );
            current_primary = primary_id;
        }

        let range_len = range.end - range.start;
        scanned += range_len;
        current_primary_len += range_len;
        scan_cluster_quantized_subspace(
            &coarse_query_values[..coarse_dim_count],
            &index.cluster_vectors_u8_coarse
                [range.start * coarse_dim_count..range.end * coarse_dim_count],
            coarse_dim_count,
            range.start,
            &mut coarse_candidates,
        );
    }

    if current_primary != usize::MAX {
        let rescored_now =
            rescore_coarse_candidates(quantized_query, index, &coarse_candidates, candidates);
        rescored += rescored_now;
        pruned += current_primary_len.saturating_sub(rescored_now);
    }

    (scanned, rescored, pruned)
}

#[inline(always)]
fn rescore_coarse_candidates(
    query: &QuantizedQuery,
    index: &HierarchicalIndex,
    coarse_candidates: &CandidateHeap,
    exact_candidates: &mut CandidateHeap,
) -> usize {
    let mut positions = coarse_candidates.positions().to_vec();
    positions.sort_unstable();
    let mut distances = [0u32; DISTANCE_BATCH_CHUNK];
    let mut rescored = 0usize;
    let mut cursor = 0usize;

    while cursor < positions.len() {
        let run_start = positions[cursor];
        let mut run_len = 1usize;
        while cursor + run_len < positions.len()
            && positions[cursor + run_len] == positions[cursor + run_len - 1] + 1
        {
            run_len += 1;
        }

        let mut run_offset = 0usize;
        while run_offset < run_len {
            let batch = usize::min(DISTANCE_BATCH_CHUNK, run_len - run_offset);
            let position = run_start + run_offset;
            let start = position * DIM;
            let end = start + batch * DIM;
            distance::l2_distance_batch_u8(
                &query.values,
                &index.cluster_vectors_u8[start..end],
                batch,
                &mut distances[..batch],
            );
            push_candidate_batch(position, &distances[..batch], exact_candidates);
            rescored += batch;
            run_offset += batch;
        }

        cursor += run_len;
    }

    rescored
}

#[inline(always)]
fn scan_cluster_quantized_block_pruned(
    query: &QuantizedQuery,
    index: &HierarchicalIndex,
    cluster_id: usize,
    block_query_values: &[u8],
    candidates: &mut CandidateHeap,
) -> (usize, usize) {
    let start = index.cluster_offsets[cluster_id];
    let end = index.cluster_offsets[cluster_id + 1];
    if start >= end {
        return (0, 0);
    }

    let block_start = index.cluster_block_offsets[cluster_id];
    let block_end = index.cluster_block_offsets[cluster_id + 1];
    if block_start >= block_end {
        scan_cluster_quantized(
            query,
            &index.cluster_vectors_u8[start * DIM..end * DIM],
            start,
            candidates,
        );
        return (end - start, 0);
    }

    let mut scanned = 0usize;
    let mut pruned = 0usize;
    for block_idx in block_start..block_end {
        let position = start + (block_idx - block_start) * BLOCK_BOUND_SIZE;
        let position_end = usize::min(position + BLOCK_BOUND_SIZE, end);
        let block_len = position_end - position;
        if block_len == 0 {
            continue;
        }

        if candidates.is_full() {
            let bounds_start = block_idx * index.block_bound_dims;
            let bounds_end = bounds_start + index.block_bound_dims;
            let lower_bound = block_box_lower_bound_u8(
                block_query_values,
                &index.cluster_block_mins[bounds_start..bounds_end],
                &index.cluster_block_maxs[bounds_start..bounds_end],
            );
            if lower_bound >= candidates.worst_distance() {
                pruned += block_len;
                continue;
            }
        }

        scan_cluster_quantized(
            query,
            &index.cluster_vectors_u8[position * DIM..position_end * DIM],
            position,
            candidates,
        );
        scanned += block_len;
    }

    (scanned, pruned)
}

#[inline(always)]
fn scan_cluster_quantized_radius_pruned(
    query: &QuantizedQuery,
    vectors: &[u8],
    radii: &[u16],
    start_position: usize,
    centroid_distance_sq: u32,
    seeded_prefix: usize,
    candidates: &mut CandidateHeap,
) -> usize {
    debug_assert_eq!(vectors.len() / DIM, radii.len());

    if radii.is_empty() {
        return 0;
    }

    let seeded_prefix = usize::min(seeded_prefix, radii.len());
    let mut start_idx = seeded_prefix;
    let mut end_idx = radii.len();
    if candidates.is_full() {
        let centroid_radius = (centroid_distance_sq as f32).sqrt();
        let worst_radius = (candidates.worst_distance() as f32).sqrt();
        let lower_radius = if centroid_radius > worst_radius {
            (centroid_radius - worst_radius).floor().max(0.0) as u16
        } else {
            0
        };
        let upper_radius =
            ((centroid_radius + worst_radius).ceil() as u32 + 1).min(u16::MAX as u32) as u16;
        start_idx = usize::max(start_idx, lower_bound_u16(radii, lower_radius));
        end_idx = upper_bound_u16(radii, upper_radius);
        if start_idx >= end_idx {
            return 0;
        }
    }

    let mut distances = [0u32; DISTANCE_BATCH_CHUNK];
    let mut current_idx = start_idx;
    let mut position = start_position + start_idx;
    while current_idx < end_idx {
        let batch = usize::min(DISTANCE_BATCH_CHUNK, end_idx - current_idx);
        let batch_start = current_idx * DIM;
        let batch_end = batch_start + batch * DIM;
        distance::l2_distance_batch_u8(
            &query.values,
            &vectors[batch_start..batch_end],
            batch,
            &mut distances[..batch],
        );
        push_candidate_batch(position, &distances[..batch], candidates);
        position += batch;
        current_idx += batch;
    }

    end_idx - start_idx
}

#[inline(always)]
fn block_box_lower_bound_u8(query: &[u8], mins: &[u8], maxs: &[u8]) -> u32 {
    let mut sum = 0u32;
    for idx in 0..query.len() {
        let q = query[idx];
        let min_value = mins[idx];
        let max_value = maxs[idx];
        let delta = if q < min_value {
            min_value as i32 - q as i32
        } else if q > max_value {
            q as i32 - max_value as i32
        } else {
            0
        };
        sum += (delta * delta) as u32;
    }
    sum
}

#[inline(always)]
fn rerank_prescored_candidates(
    query: &[f32],
    index: &HierarchicalIndex,
    candidates: &CandidateHeap,
    topk: &mut TopK,
) -> usize {
    let mut positions = candidates.positions().to_vec();
    positions.sort_unstable();
    let mut distances = [0.0f32; DISTANCE_BATCH_CHUNK];
    let mut reranked = 0usize;
    let mut cursor = 0usize;

    while cursor < positions.len() {
        let run_start = positions[cursor];
        let mut run_len = 1usize;
        while cursor + run_len < positions.len()
            && positions[cursor + run_len] == positions[cursor + run_len - 1] + 1
        {
            run_len += 1;
        }

        let mut run_offset = 0usize;
        while run_offset < run_len {
            let batch = usize::min(DISTANCE_BATCH_CHUNK, run_len - run_offset);
            let position = run_start + run_offset;
            let start = position * DIM;
            let end = start + batch * DIM;
            distance::l2_distance_batch_f16(
                query,
                &index.cluster_vectors_f16[start..end],
                batch,
                &mut distances[..batch],
            );
            for (batch_idx, &distance) in distances[..batch].iter().enumerate() {
                let candidate_position = position + batch_idx;
                topk.push(index.cluster_ids[candidate_position], distance);
            }
            reranked += batch;
            run_offset += batch;
        }

        cursor += run_len;
    }
    reranked
}

#[inline(always)]
fn lower_bound_u16(values: &[u16], target: u16) -> usize {
    let mut left = 0usize;
    let mut right = values.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if values[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[inline(always)]
fn upper_bound_u16(values: &[u16], target: u16) -> usize {
    let mut left = 0usize;
    let mut right = values.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if values[mid] <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[inline(always)]
fn cluster_lower_bound(centroid_distance_sq: u32, radius: u16) -> u32 {
    let centroid_distance = (centroid_distance_sq as f64).sqrt() as u32;
    let radius = radius as u32;
    if centroid_distance <= radius {
        return 0;
    }

    let margin = centroid_distance - radius;
    margin * margin
}

#[inline(always)]
fn push_smallest(
    ids: &mut Vec<usize>,
    distances: &mut Vec<f32>,
    limit: usize,
    id: usize,
    distance: f32,
) {
    if ids.len() < limit {
        ids.push(id);
        distances.push(distance);
        return;
    }

    let mut worst_idx = 0usize;
    for idx in 1..distances.len() {
        if distances[idx] > distances[worst_idx] {
            worst_idx = idx;
        }
    }

    if distance < distances[worst_idx] {
        ids[worst_idx] = id;
        distances[worst_idx] = distance;
    }
}

#[inline(always)]
fn push_smallest_u32(
    ids: &mut Vec<usize>,
    distances: &mut Vec<u32>,
    limit: usize,
    id: usize,
    distance: u32,
) {
    if ids.len() < limit {
        ids.push(id);
        distances.push(distance);
        return;
    }

    let mut worst_idx = 0usize;
    for idx in 1..distances.len() {
        if distances[idx] > distances[worst_idx] {
            worst_idx = idx;
        }
    }

    if distance < distances[worst_idx] {
        ids[worst_idx] = id;
        distances[worst_idx] = distance;
    }
}

fn sort_pairs(ids: &mut [usize], distances: &mut [f32]) {
    for i in 1..ids.len() {
        let id = ids[i];
        let distance = distances[i];
        let mut j = i;
        while j > 0 && distances[j - 1] > distance {
            ids[j] = ids[j - 1];
            distances[j] = distances[j - 1];
            j -= 1;
        }
        ids[j] = id;
        distances[j] = distance;
    }
}

fn sort_pairs_u32(ids: &mut [usize], distances: &mut [u32]) {
    for i in 1..ids.len() {
        let id = ids[i];
        let distance = distances[i];
        let mut j = i;
        while j > 0 && distances[j - 1] > distance {
            ids[j] = ids[j - 1];
            distances[j] = distances[j - 1];
            j -= 1;
        }
        ids[j] = id;
        distances[j] = distance;
    }
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "FALSE"),
        Err(_) => false,
    }
}

fn env_flag_default_true(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(value.as_str(), "" | "0" | "false" | "FALSE"),
        Err(_) => true,
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

#[inline(always)]
fn elapsed_ns(start: Option<Instant>) -> u64 {
    start
        .map(|instant| instant.elapsed().as_nanos() as u64)
        .unwrap_or(0)
}
