# AlphaGrad Run History

This file tracks the runs we have executed for the STAT-4830 project branch, with a focus on:
- run number
- approach/config
- observed BPB
- whether an artifact was produced
- what happened

## Baseline Reference

| Run # | Type | Approach | BPB | Artifact | Notes |
|---|---|---|---:|---|---|
| B0 | Public reference | Naive baseline (9L, 512d, 2x MLP, int8+zlib) | 1.2244 | Yes (public records) | Challenge baseline from `records/track_10min_16mb/2026-03-17_NaiveBaseline`. |

## Our Executed Runs

| Run # | Command / Context | Approach | Best observed BPB | Artifact | Outcome |
|---|---|---|---:|---|---|
| 1 | `modal run modal_exploration.py --iterations 500 --max-wallclock 120` (A10G smoke) | Early pipeline smoke test | ~4.1 | No | Short run + image/tooling issues; used for debugging only. |
| 2 | `modal run modal_exploration.py` (8xH100 baseline path) | Baseline-style full run | Not captured in local logs | Yes (`output/final_model_latest.int8.ptz`, 10.888 MB) | End-to-end produced artifact under 16 MB. |
| 3 | `modal run modal_exploration.py --iterations 20000 --max-wallclock 600` | Modified script, but missing tokenizer in Modal volume | N/A | No | Failed at startup: `/pg_data/tokenizers/fineweb_1024_bpe.model` missing. |
| 4 | `modal run modal_exploration.py` after data setup | Modified script (11L/3x/int6/SWA/XSA/EMA defaults) | 1.4073 (step 3000) | No | Trained to ~step 4000, then ended before serialization; no `final_int8_zlib_roundtrip` line. |
| 5 | `modal run modal_exploration.py` (next repeat) | Same modified defaults | 1.3903 (step 3000) | No | Same pattern: intermediate val logs exist, run ends before model-write/roundtrip stage. |
| 6 | `modal run modal_exploration.py --approach pr1493_shallow_recurrence_pass_scaled --iterations 3000 --max-wallclock 600` | PR1493-style preset before FA3 fallback | N/A | No | Failed at startup: missing `flash_attn_interface` in image. |
| 7 | same as Run 6 after FA3 fallback patch | PR1493 preset, SDPA fallback enabled | N/A | No | Failed at startup: missing tokenizer `/pg_data/tokenizers/fineweb_1024_bpe.model` in Modal app/account volume. |
| 8 | same preset after data available and fallback | PR1493 preset with legal score-first TTT | 1.2396 | Yes* | Run completed eval (`legal_ttt_exact`), produced `final_model.int6.ptz`; runner initially missed it (looked only for int8). |

## Current Modified Approach (train_gpt_exploration.py defaults)

- `NUM_LAYERS=11` (from 9)
- `MLP_MULT=3` (from 2)
- Mixed quantization with int6 packing for large non-embedding 2D tensors
- Sliding-window evaluation (`SWA_STRIDE=64`)
- Partial XSA (`XSA_START_LAYER=7`, i.e., last 4 layers)
- EMA (`EMA_DECAY=0.997`)

## Why "BPB but no artifact" can happen

`modal_exploration.py` parses BPB from any logged `val_bpb:...` line during training.  
The artifact only exists if training reaches the serialization tail.
Depending on script, artifact may be either:
- `final_model.int8.ptz`
- `final_model.int6.ptz`

So an interrupted/timed-out run can still report BPB from intermediate validation checkpoints while producing no artifact.

## Runner Status (current)

- Plateau params are passed through by default (`patience=2`, `min_delta=0.002`, `start_step=1000`).
- Preflight checks verify tokenizer + train/val shards before expensive launch.
- Artifact detection now supports both int8 and int6 outputs.
- Every run appends a one-line record to `output/iteration_log.txt`.

## Practical Next Run Settings

To stop earlier on plateau and improve chance of reaching serialization:

```powershell
cd parameter_golf
$env:EARLY_STOP_PATIENCE="2"
$env:EARLY_STOP_MIN_DELTA="0.002"
$env:EARLY_STOP_START_STEP="1000"
modal run modal_exploration.py --iterations 20000 --max-wallclock 1200
```

