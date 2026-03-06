# AirBench GEPA Scaffold

This folder contains the initial scaffold for optimizing the AirBench 94-percent CIFAR-10 script with GEPA and Modal.

## Local setup

```bash
conda activate airbench_gepa
pip install -r requirements.txt
```

## Modal smoke test

This runs the committed AirBench seed script once on a Modal A100-40GB and caches CIFAR-10 in a Modal Volume.

```bash
modal run scripts/airbench_gepa/modal_airbench.py::smoke
```

## GEPA dry run

This evaluates the seed candidate remotely and writes artifacts under `data/airbench/gepa_runs/`.

```bash
python scripts/airbench_gepa/run_gepa_airbench94.py --dry-run
```

## Example optimization runs

Frontier model:

```bash
python scripts/airbench_gepa/run_gepa_airbench94.py \
  --max-metric-calls 20 \
  --reflection-model openai/gpt-5.4
```

Smaller reasoning baseline:

```bash
python scripts/airbench_gepa/run_gepa_airbench94.py \
  --max-metric-calls 20 \
  --reflection-model openai/o3-mini
```

Cheap ablation:

```bash
python scripts/airbench_gepa/run_gepa_airbench94.py \
  --max-metric-calls 20 \
  --reflection-model openai/gpt-5-nano
```

## Notes

- The committed seed is a harnessed copy of the upstream `airbench94_muon.py` baseline with a strict CLI and JSON output contract.
- The first version uses a Modal Function for repeated evaluation. Modal Sandboxes are a natural next step for tool-calling or agent-style live iteration.
- The evaluator uses lexicographic scoring: meet the accuracy target first, then minimize runtime.
- Each run writes `eval_log.jsonl`, `progress_curve.csv`, `milestones.json`, and `summary.json` so you can track both final quality and time-to-solution.
- Modal commands are intended to be run from your own terminal so you can inspect the live logs directly.
