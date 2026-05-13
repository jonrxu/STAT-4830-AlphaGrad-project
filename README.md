# AlphaGrad: AI-Driven Optimization on Strictly Verifiable Problems

This repository contains the final code, report, slides, and experiment artifacts for the STAT 4830 semester project.

The project studies a common pattern across several domains:

- use an LLM to generate candidate solutions
- evaluate those candidates with a deterministic external checker
- keep only verified improvements
- repeat until the system reaches a strong local optimum or a new state of the art

The work spans four main research arcs:

1. test-time RL for CP26 circle packing and AC1 autocorrelation minimization
2. GEPA-style prompt/program optimization for CP26 and AirBench
3. parameter golf for small language model design under strict size and training-time constraints
4. persistent agentic optimization of a Rust vector-search server on SIFT1M

## Repository Deliverables

### 1. Final written report

- Final PDF: [STAT4830_AlphaGrad_Final_Report.pdf](STAT4830_AlphaGrad_Final_Report.pdf)
- Source draft: [report.md](report.md)

### 2. All code

Primary code locations:

- Circle packing + verifiers:
  - [scripts/circle_packing/env_cp.py](scripts/circle_packing/env_cp.py)
  - [scripts/circle_packing/verifier.py](scripts/circle_packing/verifier.py)
- AirBench GEPA:
  - [scripts/airbench_gepa/](scripts/airbench_gepa)
- AirBench autoresearch:
  - [scripts/airbench_autoresearch/](scripts/airbench_autoresearch)
- VectorDBBench:
  - [scripts/vector_db_bench/](scripts/vector_db_bench)
- Parameter golf:
  - [parameter_golf/](parameter_golf)
- Notebooks and analysis:
  - [notebooks/](notebooks)

### 3. Presentation slides

- Final presentation PDF: [Presentation Documents/STAT 4830 Final Presentation.pdf](Presentation%20Documents/STAT%204830%20Final%20Presentation.pdf)
- Earlier slide decks:
  - [Presentation Documents/Week 5 Slides.pptx](Presentation%20Documents/Week%205%20Slides.pptx)
  - [Presentation Documents/Week 6 Presentation.pptx](Presentation%20Documents/Week%206%20Presentation.pptx)
  - [Presentation Documents/Week 9 Slides.pptx](Presentation%20Documents/Week%209%20Slides.pptx)

### 4. Reproducibility instructions

This README gives the top-level reproduction map. Each subproject also has its own local README or script entry point where appropriate.

## Repository Map

```text
STAT-4830-AlphaGrad-project/
├── README.md
├── STAT4830_AlphaGrad_Final_Report.pdf
├── report.md
├── Presentation Documents/
├── data/                         # run artifacts, plots, summaries
├── docs/                         # setup notes and course artifacts
├── notebooks/                    # exploratory and analysis notebooks
├── parameter_golf/               # parameter-golf experiments and notes
├── scripts/
│   ├── circle_packing/
│   ├── airbench_gepa/
│   ├── airbench_autoresearch/
│   └── vector_db_bench/
└── third_party/
    └── vector-db-bench/          # upstream benchmark used in Part IV
```

## Environment Setup

The repository does not use one single environment for every experiment. The safest top-level setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Important notes:

- `requirements.txt` mostly covers the GEPA / orchestration side.
- AirBench GPU runs are executed through Modal rather than locally.
- VectorDBBench requires a Rust toolchain and a local clone of the upstream benchmark repo under [third_party/vector-db-bench](third_party/vector-db-bench).
- Some historical experiments assumed Conda environments such as `airbench_gepa`.

## Reproducing Results

## Part I: Test-Time RL on CP26 and AC1

Main files:

- [scripts/circle_packing/env_cp.py](scripts/circle_packing/env_cp.py)
- [scripts/circle_packing/verifier.py](scripts/circle_packing/verifier.py)
- Week 4 artifacts: [data/week4/](data/week4)

What is saved:

- best solution JSON
- step summaries
- metric logs used for the Week 4 analysis

Reproduction status:

- The repo preserves the verifiers, logged outputs, and report artifacts.
- Full reruns of the original test-time RL training require the original remote training stack and are not a one-command local reproduction.

Recommended way to verify this section:

- inspect [data/week4/best_solution.json](data/week4/best_solution.json)
- inspect [data/week4/metrics.jsonl](data/week4/metrics.jsonl)
- use the circle-packing verifier directly on saved solutions

## Part II: GEPA for Circle Packing and AirBench

### Circle Packing GEPA

Main code:

- [scripts/circle_packing/env_cp.py](scripts/circle_packing/env_cp.py)
- [scripts/circle_packing/verifier.py](scripts/circle_packing/verifier.py)
- notebooks and saved solver code under [notebooks/](notebooks)

What is saved:

- writeups in [week6_report.md](week6_report.md)
- visualization notebook outputs
- saved solver programs such as [notebooks/best_solver_code_gemini.py](notebooks/best_solver_code_gemini.py)

### AirBench GEPA

Main code:

- [scripts/airbench_gepa/run_gepa_airbench94.py](scripts/airbench_gepa/run_gepa_airbench94.py)
- [scripts/airbench_gepa/airbench_evaluator.py](scripts/airbench_gepa/airbench_evaluator.py)
- [scripts/airbench_gepa/modal_airbench.py](scripts/airbench_gepa/modal_airbench.py)
- [scripts/airbench_gepa/agent_team_proposer.py](scripts/airbench_gepa/agent_team_proposer.py)
- local notes: [scripts/airbench_gepa/README.md](scripts/airbench_gepa/README.md)

Typical commands:

```bash
python scripts/airbench_gepa/run_gepa_airbench94.py --dry-run
```

```bash
python scripts/airbench_gepa/run_gepa_airbench94.py \
  --max-metric-calls 20 \
  --reflection-model openai/gpt-5.4
```

Artifacts:

- [data/airbench/gepa_runs/](data/airbench/gepa_runs)

Notes:

- Actual training/evaluation runs depend on Modal and an available GPU-backed AirBench environment.
- The repo preserves all committed run logs and summaries needed to inspect the reported outcomes.

## Part III: Parameter Golf

Main directory:

- [parameter_golf/](parameter_golf)

Useful entry points:

- [parameter_golf/Overview.md](parameter_golf/Overview.md)
- [parameter_golf/FINDINGS.md](parameter_golf/FINDINGS.md)
- [parameter_golf/ALPHAGRAD_NOTES.md](parameter_golf/ALPHAGRAD_NOTES.md)

This part is best reproduced by following the notes and scripts in that folder directly. The experiments target a highly specific training regime and hardware budget, so this top-level README does not duplicate those details.

## Part IV: VectorDBBench

This section is the most self-contained to rerun locally.

Benchmark setup used in the project:

- upstream benchmark repo: [third_party/vector-db-bench](third_party/vector-db-bench)
- dataset: **SIFT1M**
- task: load 1M base vectors, answer top-10 nearest-neighbor queries for 10k query vectors, maximize QPS while keeping recall `>= 0.95`

Main local documentation:

- [scripts/vector_db_bench/README.md](scripts/vector_db_bench/README.md)
- [scripts/vector_db_bench/CONTRACT.md](scripts/vector_db_bench/CONTRACT.md)
- [scripts/vector_db_bench/program.md](scripts/vector_db_bench/program.md)

Main harnesses:

- [scripts/vector_db_bench/codex_cli_harness.py](scripts/vector_db_bench/codex_cli_harness.py)
- [scripts/vector_db_bench/modal_vdb_eval.py](scripts/vector_db_bench/modal_vdb_eval.py)

Typical local command:

```bash
python scripts/vector_db_bench/codex_cli_harness.py \
  --bench-repo third_party/vector-db-bench \
  --rounds 2 \
  --workers-per-round 3 \
  --strict-top-k 1
```

If benchmark inputs are stored elsewhere:

```bash
python scripts/vector_db_bench/codex_cli_harness.py \
  --bench-repo third_party/vector-db-bench \
  --base-vectors /path/to/base_vectors.json \
  --query-vectors /path/to/query_vectors.json \
  --ground-truth /path/to/ground_truth.json
```

Saved results and plots:

- Qwen snapshot:
  - [data/vector_db_bench/qwen3_meta_snapshots/](data/vector_db_bench/qwen3_meta_snapshots)
- later Codex superagent plots:
  - [data/vector_db_bench/qwen3_meta/plots/](data/vector_db_bench/qwen3_meta/plots)
- archived long campaign:
  - [data/vector_db_bench/failed_runs/codex_superagent_failed_overnight_2026-04-22/](data/vector_db_bench/failed_runs/codex_superagent_failed_overnight_2026-04-22)

What to inspect for the final reported result:

- final progression table:
  - [data/vector_db_bench/qwen3_meta/plots/codex_superagent_results_final_115.tsv](data/vector_db_bench/qwen3_meta/plots/codex_superagent_results_final_115.tsv)
- final graph:
  - [data/vector_db_bench/qwen3_meta/plots/codex_superagent_progress_final_115_styled.png](data/vector_db_bench/qwen3_meta/plots/codex_superagent_progress_final_115_styled.png)

## Key Saved Results

If you only want the main outputs used in the report, start here:

- Final report PDF:
  - [STAT4830_AlphaGrad_Final_Report.pdf](STAT4830_AlphaGrad_Final_Report.pdf)
- Final presentation:
  - [Presentation Documents/STAT 4830 Final Presentation.pdf](Presentation%20Documents/STAT%204830%20Final%20Presentation.pdf)
- Week 4 RL artifacts:
  - [data/week4/](data/week4)
- AirBench autoresearch runs:
  - [data/airbench/autoresearch_runs/](data/airbench/autoresearch_runs)
- AirBench GEPA runs:
  - [data/airbench/gepa_runs/](data/airbench/gepa_runs)
- VectorDBBench runs:
  - [data/vector_db_bench/](data/vector_db_bench)

## Notes on Reproducibility

This repository contains both:

- code needed to rerun many experiments, and
- saved artifacts from the original semester runs

Not every historical result is reproducible from a single local command, because some experiments depended on:

- remote APIs
- Modal jobs
- large external datasets
- long-running benchmark loops
- historical agent prompts and environment state

For that reason, this repository is structured so that a reader can do two things reliably:

1. inspect the exact saved artifacts used in the final report
2. rerun the main local harnesses for the GEPA, autoresearch, and VectorDBBench components

## Contact / Context

This repo was developed as a semester-long optimization project for STAT 4830. The weekly intermediate reports remain in the root for context:

- [week6_report.md](week6_report.md)
- [week8_report.md](week8_report.md)
- [week12_report.md](week12_report.md)

Those are historical milestones. The final report PDF is the authoritative summary.
