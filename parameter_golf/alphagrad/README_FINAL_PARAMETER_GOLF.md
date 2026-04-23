# AlphaGrad Project README

Comprehensive guide for our STAT-4830 AlphaGrad Parameter Golf work: setup, approaches, run history, and how to reproduce everything from scratch.

---

## 1) What this project is

We are experimenting with OpenAI's Parameter Golf challenge:

- Train a language model under strict wallclock constraints.
- Evaluate with BPB (bits per byte) on FineWeb validation.
- Produce a compact artifact (`.ptz`) after quantization + compression.

Our local workflow uses Modal to launch multi-GPU runs and bring back logs/artifacts.

---

## 2) Core files in this repo

### Training / orchestration

- `parameter_golf/train_gpt_exploration.py`
  - Our exploration training script (11L/3x/int6/SWA/XSA/EMA defaults).
  - Includes plateau-based early stopping defaults.

- `parameter_golf/modal_exploration.py`
  - Main launcher used in this project.
  - Dispatches training scripts to Modal (8xH100).
  - Supports approach presets.
  - Performs preflight data checks.
  - Handles both int8 and int6 artifacts.
  - Writes run records to `output/iteration_log.txt`.

- `parameter_golf/alphagrad/modal_runner.py`
  - One-time data setup for Modal volume (`pg-fineweb-data`).
  - Downloads tokenizer + train/val shards to `/pg_data/...`.

### Documentation / run tracking

- `parameter_golf/ALPHAGRAD_NOTES.md` — high-level project notes.
- `parameter_golf/RUN_HISTORY.md` — curated run chronology.
- `parameter_golf/output/iteration_log.txt` — auto-appended per-run telemetry.

---

## 3) Environment setup (from scratch)

## Python

Use Python 3.11 (recommended for Modal CLI compatibility).

```powershell
cd parameter_golf
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install modal sentencepiece numpy
```

## Modal auth

```powershell
modal token set --token-id <TOKEN_ID> --token-secret <TOKEN_SECRET>
```

## One-time data prep (required)

```powershell
cd parameter_golf
modal run alphagrad/modal_runner.py
```

This populates Modal volume files used by runs:

- `/pg_data/tokenizers/fineweb_1024_bpe.model`
- `/pg_data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `/pg_data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`

---

## 4) How launching works now

`modal_exploration.py` currently does:

1. Select script via approach preset.
2. Preflight check for tokenizer/train/val files in volume.
3. Launch `torchrun --nproc_per_node=8` on Modal H100.
4. Parse BPB from logs:
   - `best_observed_bpb` = min of observed `val_bpb` lines.
   - `final_logged_bpb` = last observed `val_bpb`.
5. Detect artifact output:
   - `final_model.int8.ptz` **or**
   - `final_model.int6.ptz`
6. Save local artifact under `parameter_golf/output/`.
7. Append one line to `output/iteration_log.txt`.

### Timeout budget

Runner subprocess timeout is:

`MAX_WALLCLOCK_SECONDS + 600 + SERIALIZATION_BUFFER_SECONDS`

Default serialization buffer is 900s.

---

## 5) Plateau early-stop behavior

For exploration stack defaults, plateau controls are on by default:

- `EARLY_STOP_PATIENCE=2`
- `EARLY_STOP_MIN_DELTA=0.002`
- `EARLY_STOP_START_STEP=1000`

These are also passed through by `modal_exploration.py` unless overridden.

---

## 6) Current approach presets

In `modal_exploration.py`, `--approach` supports:

- `exploration_default`
  - Uses `train_gpt_exploration.py`.

- `pr1493_shallow_recurrence_pass_scaled`
  - Uses `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`.
  - Legal score-first TTT enabled.
  - Recurrence/pass schedule knobs passed through.
  - FA3 import fallback applied if missing.

- `record_chasing_no_ttt_gptq_15mb`
  - Uses `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`.
  - No TTT, XSA-all, larger BigramHash.
  - Targets ~15MB artifact class (`TARGET_MB=15.0`).
  - FA3 import fallback applied if missing.

---

## 7) Standard commands we use

## Baseline exploration run

```powershell
cd parameter_golf
modal run modal_exploration.py
```

## PR1493-style recurrence/TTT run

```powershell
cd parameter_golf
modal run modal_exploration.py --approach pr1493_shallow_recurrence_pass_scaled --iterations 3000 --max-wallclock 600
```

## Record-chasing no-TTT ~15MB run

```powershell
cd parameter_golf
modal run modal_exploration.py --approach record_chasing_no_ttt_gptq_15mb --max-wallclock 600
```

---

## 8) Run history summary (what we have done)

Reference baseline:

- Public naive baseline: `1.2244` BPB.

Our notable runs:

1. A10G smoke (`500 iters`, `120s`) — BPB ~4.1 (debug run), no artifact.
2. Full baseline path (`8xH100`, `600s`) — artifact produced (`10.888 MB`), roundtrip BPB not preserved in local logs.
3. Multiple failed starts due to missing tokenizer in volume (`/pg_data/tokenizers/fineweb_1024_bpe.model`).
4. Exploration runs reached intermediate BPBs (`~1.4073`, `~1.3903`) but ended before serialization in earlier runner settings.
5. PR1493-stack run with FA3 dependency issue (`flash_attn_interface` missing) — fixed via runtime fallback.
6. PR1493-stack run after fixes reached:
   - `legal_ttt_exact ... val_bpb=1.23959067`
   - artifact was produced as `final_model.int6.ptz` but initially not detected (runner expected int8 only) — fixed.

For exact chronology, see:

- `parameter_golf/RUN_HISTORY.md`
- `parameter_golf/output/iteration_log.txt`

---

## 9) How to read `iteration_log.txt`

Each run appends:

- timestamp
- run_id
- approach + override config
- key config lines extracted from logs
- `best_observed_bpb`
- `final_logged_bpb`
- `artifact_size`
- `serialized_model_size` (parsed from logs when available)
- status (`artifact_ok` / `no_artifact`)

---

## 10) Common failure modes and fixes

## Missing tokenizer / data in Modal volume

Symptom:

- `OSError: Not found: "/pg_data/tokenizers/fineweb_1024_bpe.model"`

Fix:

```powershell
cd parameter_golf
modal run alphagrad/modal_runner.py
```

Run in same Modal account/profile as training command.

## FA3 import missing

Symptom:

- `ModuleNotFoundError: No module named 'flash_attn_interface'`

Fix:

- Use approach presets with `FORCE_FA3_FALLBACK=1` (already wired in runner).

## BPB shown but no artifact

Causes we observed:

- Run terminated before serialization stage.
- Artifact extension mismatch (int6 file produced while runner only looked for int8).

Current status:

- runner now detects both `.int8.ptz` and `.int6.ptz`.

---

## 11) Current objective direction

Active direction we are testing:

- PR1493-style stack variants plus recurrence-aware pass controls.
- Record-chasing no-TTT GPTQ-style variants at ~15MB.
- Maintain strong run bookkeeping (artifact + BPB + serialized size each run).

---

## 12) Quick checklist before any run

1. Activate Python 3.11 venv.
2. Confirm Modal auth works.
3. Run `modal_runner.py` at least once in same account/profile.
4. Launch selected `--approach`.
5. Verify after run:
   - artifact exists in `parameter_golf/output/`
   - latest line appended in `output/iteration_log.txt`
   - BPB and serialized size values recorded.

