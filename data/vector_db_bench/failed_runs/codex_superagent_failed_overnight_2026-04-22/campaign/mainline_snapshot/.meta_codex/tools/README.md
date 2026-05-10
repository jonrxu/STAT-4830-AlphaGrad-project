# Workspace Tools

These scripts provide official-style benchmark actions and local measurement helpers inside the Codex workspace:

- `build_project`
- `run_correctness_test`
- `run_benchmark`
- `run_profiling`
- `get_status`
- `bench_stability`
- `analyze_query_prefixes`
- `analyze_request_cache_keys`
- `capture_request_shapes`
- `capture_benchmark_lanes`
- `replay_search_bodies`
- `replay_search_bodies_fast`
- `measure_dummy_cache_ceiling`
- `measure_cache_path_ceiling`
- `measure_official_client_ceiling`
- `measure_official_client_breakdown`
- `measure_official_benchmark_breakdown`
- `measure_ann_benchmark`
- `measure_ann_benchmark_stability`
- `measure_ann_search_only`
- `compare_ann_coarse_shortlist`
- `compare_ann_u8_routing`
- `sweep_coarse_prescore_dims`
- `sweep_ann_benchmark`
- `sweep_ann_search_only`
- `sweep_official_breakdown`
- `measure_reqwest_search_phases`

They are wrappers around the same benchmark helper logic used by the driver.

`measure_official_benchmark_breakdown` now captures both request stats and ANN stats when the
server writes them, so `measure_ann_benchmark` can expose exact-wrapper ANN internals from the
same run.

The scored cache path now keeps `VDB_SPLIT_CACHED_SEARCH_RESPONSE` default-on because cycle 54
found repeated wins on the exact breakdown wrapper. Use `VDB_SPLIT_CACHED_SEARCH_RESPONSE=0`
or `sweep_official_breakdown VDB_SPLIT_CACHED_SEARCH_RESPONSE off 1` when you need an explicit
default-vs-split A/B.

`measure_ann_benchmark --fast` and `sweep_ann_benchmark --fast` skip that extra stats wrapper and
run the official benchmark directly, which is the cheaper decision gate for ANN QPS sweeps. The
ANN helpers also now expose explicit `--enable-u8-routing` / `--disable-u8-routing` flags for the
opt-in quantized routing branch. The ANN sweep helpers now also expose
`--enable-vector-radius-prune` / `--disable-vector-radius-prune` so radius-band A/B runs do not
need one-off env wrappers. Cycle-36 also added experimental
`--coarse-prescore-dims` / `--coarse-prescore-candidates` passthrough on the ANN wrappers for
ordered-dimension shortlist experiments; keep those knobs default-off unless a branch has already
shown promise locally.

Cycle-51 adds `--primary-clusters` to the ANN measure/sweep wrappers. It rebuilds the top-level
IVF fanout as `P x N` instead of being locked to `256 x N`; use it for geometry studies only, and
do not trust local wins without the exact ANN gate because the first `384 x 64` winner did not
transfer.

Cycle-52 adds `--enable-leaf-supercluster-routing` / `--disable-leaf-supercluster-routing` to the
ANN measure/sweep wrappers. The branch keeps the existing leaf clusters and scan path, but it
builds a second k-means routing layer over those leaf centroids so the search can expand
super-clusters of leaves instead of only the original per-primary buckets. Keep it experimental
until an exact ANN check beats the same-point flat route.

Cycle-53 adds `--quantized-prescore-mode global|primary_residual` to the ANN measure/sweep
wrappers. It switches the flat shortlisted-vector prescore between the old global `u8` space and
an experimental primary-residual `u8` space with shared residual scales across all primaries.
Keep it experimental: the first local-valid `14 / 144 / 416` no-radius point reached about
`5896.86 QPS`, `0.9507` on the primed local gate, but the fast exact `1000`-query check reached
only about `4877.22`, `0.9557`.

Cycle-50 adds `--local-subcluster-routing u8|f32` to the ANN measure/sweep wrappers. Use it only
for the parked local-subcluster branch; it switches that selector between the old quantized local
centroid routing path and a new exact `f32` local-centroid routing path without touching the flat
default ANN branch.

Cycle-48 adds `--secondary-clusters` to the ANN measure/sweep wrappers. It rebuilds the `256 x 64`
hierarchy as `256 x N` for local or exact ANN tests without touching the default scored path; use
it for geometry experiments only, not as a promotion signal by itself.

Cycle-43 extends that coarse branch with `--coarse-prescore-scope cluster|primary`, so the same
local/exact ANN wrappers can compare per-cluster and per-primary compact shortlist groupings
without ad hoc env wrappers.

Cycle-49 adds `--coarse-prescore-mode raw|residual`, which lets the same compact shortlist branch
scan either raw selected dimensions or centroid-relative residual dimensions for the chosen
cluster/primary scope.

`compare_ann_u8_routing` is a narrow serial A/B wrapper for routing rechecks. It runs the same
ANN point with `u8` routing off and on, then emits both payloads plus the QPS/recall delta in one
JSON summary so future cycles do not need hand-run paired measurements for that branch.

`measure_ann_benchmark_stability` repeats one ANN exact benchmark point and summarizes the QPS /
recall spread. Use it when the fast ANN gate looks noisy and a narrow point or routing A/B needs
more than one sample before you trust it.

`compare_ann_coarse_shortlist` is a narrow local A/B wrapper for the parked coarse-shortlist
branch. It repeats the same search-only point with and without coarse prescore enabled, then emits
paired QPS / recall / ANN-scan summaries so future cycles do not need hand-run baseline-vs-coarse
comparisons.

Cycle-40 also added experimental `--enable-pq-prescore` to the ANN wrappers for the new global PQ
shortlist branch, plus `measure_ann_search_only --prime-build` so local search-only runs can force
first-search index build before the timed replay on heavier ANN branches. As of cycle 41, enabling
PQ also forces `VDB_ENABLE_VECTOR_RADIUS_PRUNE=0` unless you explicitly request an invalid
combination, so the PQ branch now actually runs instead of silently falling back to the default
radius-pruned flat path. Cycle 42 extends that with `--pq-mode`; the default PQ mode is now the
new `primary_residual` branch, and `--pq-mode global` re-runs the older global-codebook variant.

Examples:

```bash
.meta_codex/tools/build_project
.meta_codex/tools/run_correctness_test
.meta_codex/tools/run_benchmark
.meta_codex/tools/run_benchmark --full
.meta_codex/tools/run_profiling --duration 30
.meta_codex/tools/get_status
.meta_codex/tools/bench_stability 3
.meta_codex/tools/analyze_query_prefixes
.meta_codex/tools/analyze_request_cache_keys
.meta_codex/tools/capture_request_shapes --warmup 1 --queries 1
.meta_codex/tools/capture_benchmark_lanes --warmup 100 --queries 200 --concurrency 4
.meta_codex/tools/replay_search_bodies --server-url http://127.0.0.1:8080 --queries 2000
.meta_codex/tools/replay_search_bodies_fast --server-url http://127.0.0.1:8080 --queries 2000
.meta_codex/tools/measure_dummy_cache_ceiling --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_cache_path_ceiling --queries 5000 --warmup 1000 --concurrency 4
.meta_codex/tools/measure_official_client_ceiling --backend real --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_ceiling --backend dummy --distance-mode zero --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_ceiling --backend real --split-search-response --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_ceiling --backend real --close-after-cached-search --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_breakdown --backend real -- --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_breakdown --backend dummy --split-search-response -- --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_breakdown --backend real --post-write-spin-us 25 -- --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_client_breakdown --backend real --close-after-cached-search -- --max-queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_official_benchmark_breakdown
.meta_codex/tools/measure_official_benchmark_breakdown --full
.meta_codex/tools/measure_ann_benchmark -- --max-queries 200
.meta_codex/tools/measure_ann_benchmark --fast -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --primary-probe 12 --cluster-probe 128 --prescore-candidates 1024 -- --max-queries 200
.meta_codex/tools/measure_ann_benchmark --fast --secondary-clusters 80 --cluster-probe 152 --prescore-candidates 384 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --cluster-probe 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark_stability --count 3 --fast --cluster-probe 138 --prescore-candidates 384 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --fast --enable-pq-prescore --cluster-probe 136 --prescore-candidates 352 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --fast --enable-pq-prescore --pq-mode global --cluster-probe 136 --prescore-candidates 352 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --fast --disable-vector-radius-prune --quantized-prescore-mode primary_residual --cluster-probe 144 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --coarse-prescore-dims 32 --cluster-probe 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --coarse-prescore-dims 32 --coarse-prescore-candidates 2048 --coarse-prescore-scope primary --cluster-probe 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --coarse-prescore-dims 32 --coarse-prescore-candidates 2048 --coarse-prescore-scope cluster --coarse-prescore-mode residual --cluster-probe 138 --prescore-candidates 384 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --fast --enable-u8-routing --cluster-probe 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/compare_ann_u8_routing --mode exact --fast --cluster-probe 144 --prescore-candidates 416
.meta_codex/tools/measure_ann_benchmark --disable-vector-radius-prune --cluster-probe 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --enable-vector-radius-prune --cluster-probe 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/measure_ann_benchmark --enable-block-bound-prune --block-bound-dims 8 --cluster-probe 138 --prescore-candidates 384 -- --max-queries 200
.meta_codex/tools/measure_ann_benchmark --cluster-probe 138 --prescore-candidates 416 --local-subcluster-target 24 --local-subcluster-probe 1 --local-subcluster-min-size 32 --enable-cluster-prune -- --max-queries 200
.meta_codex/tools/measure_ann_benchmark --enable-cluster-prune -- --max-queries 200
.meta_codex/tools/measure_ann_search_only -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --primary-clusters 384 --primary-probe 20 --secondary-clusters 64 --cluster-probe 192 --prescore-candidates 384 --prime-build --include-ann-stats -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/measure_ann_search_only --enable-leaf-supercluster-routing --primary-probe 24 --cluster-probe 144 --prescore-candidates 416 --prime-build -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/measure_ann_search_only --secondary-clusters 80 --cluster-probe 152 --prescore-candidates 384 -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/measure_ann_search_only --enable-u8-routing -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --disable-vector-radius-prune --quantized-prescore-mode primary_residual --cluster-probe 144 --prescore-candidates 416 --prime-build -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/compare_ann_u8_routing --mode local --prime-build --cluster-probe 144 --prescore-candidates 416 -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/compare_ann_coarse_shortlist --count 2 --cluster-probe 138 --prescore-candidates 416 --coarse-prescore-dims 32 --coarse-prescore-candidates 2048 -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/measure_ann_search_only --disable-vector-radius-prune -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --enable-vector-radius-prune --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --enable-block-bound-prune --block-bound-dims 8 --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --include-request-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --include-request-stats --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --primary-probe 14 --cluster-probe 132 --prescore-candidates 640 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --cluster-probe 138 --prescore-candidates 416 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --cluster-probe 136 --prescore-candidates 352 --enable-pq-prescore --prime-build --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --cluster-probe 136 --prescore-candidates 352 --enable-pq-prescore --pq-mode global --prime-build --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --coarse-prescore-dims 32 --cluster-probe 138 --prescore-candidates 416 --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --coarse-prescore-dims 32 --coarse-prescore-candidates 2048 --coarse-prescore-scope primary --cluster-probe 138 --prescore-candidates 416 --include-ann-stats --prime-build -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --coarse-prescore-dims 32 --coarse-prescore-candidates 2048 --coarse-prescore-scope cluster --coarse-prescore-mode residual --cluster-probe 138 --prescore-candidates 384 --include-ann-stats --prime-build -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_coarse_prescore_dims --runner search-only --dims 8 12 16 24 32 -- --cluster-probe 138 --prescore-candidates 416 --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_coarse_prescore_dims --runner search-only --dims 16 24 32 --coarse-candidates 1536 2048 -- --cluster-probe 138 --prescore-candidates 416 --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --cluster-probe 138 --prescore-candidates 416 --local-subcluster-target 24 --local-subcluster-probe 1 --local-subcluster-min-size 32 --enable-cluster-prune --include-ann-stats -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/measure_ann_search_only --cluster-probe 138 --prescore-candidates 384 --local-subcluster-target 24 --local-subcluster-probe 1 --local-subcluster-min-size 32 --local-subcluster-routing f32 --include-ann-stats --prime-build -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_ann_benchmark --cluster-probes 140 136 134 --prescore-candidates 1536 1280 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --secondary-clusters 64 80 96 --cluster-probes 138 152 --prescore-candidates 384 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --fast --enable-pq-prescore --cluster-probes 136 138 --prescore-candidates 1024 1536 2048 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --primary-clusters 256 384 --primary-probes 14 20 --secondary-clusters 64 --cluster-probes 192 --prescore-candidates 384 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --fast --enable-u8-routing --cluster-probes 136 138 --prescore-candidates 416 448 480 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --coarse-prescore-dims 32 --coarse-prescore-candidates 1536 2048 --coarse-prescore-scopes cluster primary --cluster-probes 138 --prescore-candidates 416 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --cluster-probes 136 138 --prescore-candidates 416 448 480 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --fast --disable-vector-radius-prune --quantized-prescore-modes global primary_residual --cluster-probes 144 --prescore-candidates 384 416 -- --max-queries 200
.meta_codex/tools/sweep_ann_benchmark --enable-vector-radius-prune --cluster-probes 136 138 --prescore-candidates 416 448 480 -- --max-queries 1000
.meta_codex/tools/sweep_ann_benchmark --enable-block-bound-prune --block-bound-dims 8 --cluster-probes 138 --prescore-candidates 384 -- --max-queries 200
.meta_codex/tools/sweep_ann_benchmark --cluster-probes 138 --prescore-candidates 416 --local-subcluster-targets 24 --local-subcluster-probes 1 2 --local-subcluster-min-sizes 32 --enable-cluster-prune -- --max-queries 200
.meta_codex/tools/sweep_ann_benchmark --enable-cluster-prune --cluster-probes 140 136 134 --prescore-candidates 1536 1280 -- --max-queries 1000
.meta_codex/tools/sweep_ann_search_only --cluster-probes 140 136 132 --prescore-candidates 896 768 640 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --secondary-clusters 64 80 96 --cluster-probes 138 152 --prescore-candidates 384 -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --enable-pq-prescore --cluster-probes 136 138 --prescore-candidates 1024 1536 2048 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --primary-clusters 256 384 --primary-probes 14 20 --secondary-clusters 64 --cluster-probes 176 192 --prescore-candidates 384 --prime-build --include-ann-stats -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --enable-leaf-supercluster-routing --primary-probes 14 20 24 --cluster-probes 144 152 --prescore-candidates 416 --prime-build -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --enable-u8-routing --cluster-probes 136 138 --prescore-candidates 416 448 480 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --coarse-prescore-dims 32 --coarse-prescore-candidates 1536 2048 --coarse-prescore-scopes cluster primary --cluster-probes 138 --prescore-candidates 416 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --cluster-probes 136 138 --prescore-candidates 416 448 480 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --disable-vector-radius-prune --quantized-prescore-modes global primary_residual --cluster-probes 144 --prescore-candidates 384 416 --prime-build -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --disable-vector-radius-prune --cluster-probes 136 138 --prescore-candidates 416 448 480 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --enable-block-bound-prune --block-bound-dims 8 --cluster-probes 138 --prescore-candidates 384 -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --cluster-probes 138 --prescore-candidates 416 --local-subcluster-targets 24 --local-subcluster-probes 1 2 --local-subcluster-min-sizes 32 --enable-cluster-prune -- --queries 1000 --warmup 100 --concurrency 4
.meta_codex/tools/sweep_ann_search_only --cluster-probes 138 --prescore-candidates 384 --local-subcluster-targets 24 --local-subcluster-probes 1 2 --local-subcluster-min-sizes 32 --local-subcluster-routings u8 f32 --prime-build --include-ann-stats -- --queries 300 --warmup 50 --concurrency 4
.meta_codex/tools/sweep_official_breakdown VDB_EARLY_CACHED_SEARCH_PARTIAL_READ_BYTES off 256 512
.meta_codex/tools/measure_reqwest_search_phases --backend real -- --max-queries 1000 --warmup 100 --concurrency 4 --completion-mode json
.meta_codex/tools/measure_reqwest_search_phases --backend dummy --split-search-response -- --max-queries 1000 --warmup 100 --concurrency 4 --completion-mode bytes
.meta_codex/tools/measure_reqwest_search_phases --backend real --early-cached-search-headers -- --max-queries 1000 --warmup 100 --concurrency 4 --completion-mode json
.meta_codex/tools/measure_reqwest_search_phases --backend real --preload-official-base --disable-force-official-cache -- --max-queries 1000 --warmup 100 --concurrency 4 --completion-mode json
.meta_codex/tools/measure_reqwest_search_phases --backend real --benchmark-warmup-semantics --preload-official-base --disable-force-official-cache -- --max-queries 1000 --warmup 100 --concurrency 4 --completion-mode json
.meta_codex/tools/measure_reqwest_search_phases --backend real --preload-official-base --disable-force-official-cache --early-cached-search-headers -- --max-queries 1000 --warmup 100 --concurrency 4 --completion-mode json
```

`finish` is intentionally absent because the outer driver owns cycle boundaries and promotion.
