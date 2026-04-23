#!/usr/bin/env python3
"""
Run train_gpt_exploration.py on a Modal GPU and stream logs back locally.

Authentication
--------------
One-time: run ``modal token set --token-id <key>`` to authenticate.

Data volume
-----------
The FineWeb data must already be in the Modal volume ``pg-fineweb-data``.
If not set up yet, run once:
  python -m alphagrad.modal_runner

Usage
-----
  cd parameter_golf
  modal run modal_exploration.py                         # defaults (8×H100, 600s wall)
  modal run modal_exploration.py --iterations 5000
  modal run modal_exploration.py --max-wallclock 300

Output
------
Logs are streamed to your terminal.
``output/final_model.int8.ptz`` is written locally when training finishes.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import modal

# ── Shared constants (same volume as alphagrad/modal_runner.py) ──────────────
DATA_REMOTE = "/pg_data"
DATASET_DIR = f"{DATA_REMOTE}/datasets/fineweb10B_sp1024"
TOKENIZER_PATH_REMOTE = f"{DATA_REMOTE}/tokenizers/fineweb_1024_bpe.model"

data_vol = modal.Volume.from_name("pg-fineweb-data", create_if_missing=True)

app = modal.App("pg-exploration")

exploration_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("gcc")  # Triton needs a C compiler to JIT-compile its CUDA driver shim
    .pip_install(
        "sentencepiece",
        "numpy",
        "huggingface-hub>=0.24",
        "datasets",
        "tqdm",
    )
)

# ── Remote function ───────────────────────────────────────────────────────────
@app.function(
    image=exploration_image,
    gpu="h100:8",
    volumes={DATA_REMOTE: data_vol},
    timeout=2400,
)
def run_exploration(script_content: str, config: dict, gpu_type: str = "h100x8") -> dict:
    """
    Run train_gpt_exploration.py on 8×H100 GPUs.

    Args:
        script_content: Full text of train_gpt_exploration.py.
        config:         Dict of env-var overrides (ITERATIONS, MAX_WALLCLOCK_SECONDS, ...).
        gpu_type:       Informational label only.

    Returns:
        {
          "log":      full stdout+stderr string,
          "bpb":      final val_bpb (float or None),
          "artifact": bytes of final_model.int8.ptz (or None if not produced),
        }
    """
    import os as _os

    # torch.compile is fully supported on H100 (Hopper / Triton).
    # Optional source patching is used for compatibility shims (e.g., FA3 import fallback).
    patched = script_content
    if str(config.get("FORCE_FA3_FALLBACK", "0")) in {"1", "true", "True"} and "flash_attn_interface" in patched:
        patched = patched.replace(
            "from flash_attn_interface import flash_attn_func as flash_attn_3_func",
            "try:\n"
            "    from flash_attn_interface import flash_attn_func as flash_attn_3_func\n"
            "except Exception:\n"
            "    flash_attn_3_func = None",
            1,
        )
        patched = patched.replace(
            "        y = flash_attn_3_func(q, k, v, causal=True)",
            "        if flash_attn_3_func is not None:\n"
            "            y = flash_attn_3_func(q, k, v, causal=True)\n"
            "        else:\n"
            "            y = F.scaled_dot_product_attention(\n"
            "                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),\n"
            "                attn_mask=None, is_causal=True,\n"
            "                enable_gqa=(self.num_kv_heads != self.num_heads),\n"
            "            ).transpose(1, 2)",
            1,
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(patched)
        script_path = f.name

    env = _os.environ.copy()
    env["DATA_PATH"] = DATASET_DIR
    env["TOKENIZER_PATH"] = TOKENIZER_PATH_REMOTE
    for k, v in config.items():
        env[str(k)] = str(v)

    # Match the training script wallclock cap (default 600s).
    # proc_timeout includes:
    #   training wallclock
    # + 600s buffer for compile/validation overhead during training
    # + serialization buffer for final model write + round-trip eval.
    wallclock = float(config.get("MAX_WALLCLOCK_SECONDS", 600))
    serialization_buffer = int(config.get("SERIALIZATION_BUFFER_SECONDS", 900))
    proc_timeout = int(wallclock) + 600 + max(serialization_buffer, 0)

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", script_path]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=proc_timeout,
            cwd="/tmp",  # script writes final_model.int8.ptz to cwd; we look in /tmp
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired as e:
        def _decode(b: bytes | str | None) -> str:
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="replace")
            return b or ""
        output = _decode(e.stdout) + "\n" + _decode(e.stderr)

    # Parse observed val_bpb values from logs.
    matches = re.findall(r"val_bpb:([\d.]+)", output)
    bpb = float(matches[-1]) if matches else None
    best_bpb = min((float(m) for m in matches), default=None)
    size_matches = re.findall(r"Serialized model [^:]+:\s*([0-9]+)\s+bytes", output)
    serialized_model_bytes = int(size_matches[-1]) if size_matches else None

    # Read artifact if produced (support both int8 and int6 outputs).
    artifact: bytes | None = None
    artifact_name: str | None = None
    for ptz_name in ("final_model.int8.ptz", "final_model.int6.ptz"):
        ptz_path = f"/tmp/{ptz_name}"
        if _os.path.exists(ptz_path):
            with open(ptz_path, "rb") as af:
                artifact = af.read()
            artifact_name = ptz_name
            break
    else:
        # training script writes to cwd; also check script dir
        for candidate in [
            _os.path.join(_os.path.dirname(script_path), "final_model.int8.ptz"),
            _os.path.join(_os.path.dirname(script_path), "final_model.int6.ptz"),
            "final_model.int8.ptz",
            "final_model.int6.ptz",
        ]:
            if _os.path.exists(candidate):
                with open(candidate, "rb") as af:
                    artifact = af.read()
                artifact_name = _os.path.basename(candidate)
                break

    return {
        "log": output,
        "bpb": bpb,
        "best_bpb": best_bpb,
        "serialized_model_bytes": serialized_model_bytes,
        "artifact": artifact,
        "artifact_name": artifact_name,
    }


@app.function(
    image=exploration_image,
    volumes={DATA_REMOTE: data_vol},
    timeout=300,
)
def check_data_ready() -> dict:
    """Verify tokenizer + dataset files exist in the mounted Modal volume."""
    import glob as _glob
    import os as _os

    tokenizer_ok = _os.path.exists(TOKENIZER_PATH_REMOTE)
    train_count = len(_glob.glob(f"{DATASET_DIR}/fineweb_train_*.bin"))
    val_count = len(_glob.glob(f"{DATASET_DIR}/fineweb_val_*.bin"))
    return {
        "tokenizer_ok": tokenizer_ok,
        "train_shards": train_count,
        "val_shards": val_count,
        "tokenizer_path": TOKENIZER_PATH_REMOTE,
        "dataset_dir": DATASET_DIR,
    }


# ── Local entrypoint ──────────────────────────────────────────────────────────
# Modal parses typed parameters from the CLI automatically.
# Usage:  modal run modal_exploration.py --gpu h100 --iterations 20000
@app.local_entrypoint()
def main(
    gpu: str = "a10g",
    approach: str = "exploration_default",
    script: str = "",
    iterations: int = 0,
    max_wallclock: float = 0.0,
    warmdown_iters: int = 0,
    serialization_buffer_seconds: int = 900,
    early_stop_patience: int = 2,
    early_stop_min_delta: float = 0.002,
    early_stop_start_step: int = 1000,
    seed: int = 0,
    run_id: str = "",
    val_loss_every: int = 0,
) -> None:
    # Build config dict from non-default values only
    config: dict[str, object] = {}
    if iterations:
        config["ITERATIONS"] = iterations
    if max_wallclock:
        config["MAX_WALLCLOCK_SECONDS"] = max_wallclock
    if warmdown_iters:
        config["WARMDOWN_ITERS"] = warmdown_iters
    # Always pass through plateau controls by default (caller can override).
    config["EARLY_STOP_PATIENCE"] = early_stop_patience
    config["EARLY_STOP_MIN_DELTA"] = early_stop_min_delta
    config["EARLY_STOP_START_STEP"] = early_stop_start_step
    # Always pass through explicit serialization budget.
    config["SERIALIZATION_BUFFER_SECONDS"] = serialization_buffer_seconds
    if seed:
        config["SEED"] = seed
    if run_id:
        config["RUN_ID"] = run_id
    if val_loss_every:
        config["VAL_LOSS_EVERY"] = val_loss_every

    # Approach presets.
    # - exploration_default: local train_gpt_exploration.py defaults
    # - pr1493_shallow_recurrence_pass_scaled: closest local PR1493-style stack
    #   (legal score-first TTT + parallel residual/banked weights), with shallow settings
    #   and pass-related knobs pre-populated in env for reproducibility.
    # - record_chasing_no_ttt_gptq_15mb: high-upside no-TTT stack based on the
    #   AR self-generated GPTQ + XSA-all recipe with an explicit ~15 MB target.
    base_dir = Path(__file__).resolve().parent
    if approach == "exploration_default":
        default_script = base_dir / "train_gpt_exploration.py"
    elif approach == "pr1493_shallow_recurrence_pass_scaled":
        default_script = (
            base_dir
            / "records"
            / "track_10min_16mb"
            / "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
            / "train_gpt.py"
        )
        # Keep legal score-first TTT stack enabled.
        config.setdefault("TTT_ENABLED", 1)
        # Allow a larger artifact target (25 MB class) by scaling model shape up.
        config.setdefault("NUM_LAYERS", 12)
        config.setdefault("MODEL_DIM", 640)
        config.setdefault("MLP_MULT", 3.5)
        config.setdefault("XSA_LAST_N", 3)
        # Recurrence/gain-scaling intent knobs (3-pass).
        # Note: these are passed through for recurrence-aware stacks and tracked in logs.
        config.setdefault("TRAINING_DEPTH_RECURRENCE", 3)
        config.setdefault("EVAL_DEPTH_RECURRENCE", 3)
        config.setdefault("PASS_SCALE_MODE", "learned_per_pass")
        config.setdefault("PASS_QK_GAIN_MODE", "learned_per_pass")
        config.setdefault("PASS_SCALE_SCHEDULE", "1.00,0.88,0.78")
        config.setdefault("PASS_QK_GAIN_SCHEDULE", "5.20,4.85,4.55")
        # Keep a base qk gain but avoid forcing one shared 5.25 everywhere.
        config.setdefault("QK_GAIN_INIT", 4.8)
        # Soft target for this preset (informational; use parsed size logs for truth).
        config.setdefault("TARGET_ARTIFACT_MB", 25)
        # PR1493 stack expects FA3 module that may be absent in the base Modal image.
        # Force an SDPA compatibility fallback by patching script content at runtime.
        config.setdefault("FORCE_FA3_FALLBACK", 1)
    elif approach == "record_chasing_no_ttt_gptq_15mb":
        default_script = (
            base_dir
            / "records"
            / "track_10min_16mb"
            / "2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072"
            / "train_gpt.py"
        )
        # Keep this test aligned with the strongest no-TTT direction.
        config.setdefault("TTT_ENABLED", 0)
        config.setdefault("XSA_LAST_N", 11)      # XSA on all layers
        config.setdefault("BIGRAM_VOCAB_SIZE", 3072)
        config.setdefault("BIGRAM_DIM", 112)
        config.setdefault("WARMDOWN_ITERS", 4000)
        # Keep model around 15 MB compressed target class.
        config.setdefault("TARGET_MB", 15.0)
        # FA3 compatibility fallback for environments without flash_attn_interface.
        config.setdefault("FORCE_FA3_FALLBACK", 1)
    else:
        raise SystemExit(
            f"Unknown approach '{approach}'. "
            f"Use 'exploration_default', 'pr1493_shallow_recurrence_pass_scaled', "
            f"or 'record_chasing_no_ttt_gptq_15mb'."
        )

    script_path = Path(script).expanduser().resolve() if script else default_script
    if not script_path.is_file():
        print(f"ERROR: cannot find {script_path}", file=sys.stderr)
        raise SystemExit(1)
    script_content = script_path.read_text(encoding="utf-8")

    print(f"Dispatching {script_path.name} to Modal (8×H100)...")
    print(f"Approach preset: {approach}")
    print(f"Config overrides: {config or '(none — using script defaults)'}")
    print("Logs will stream here; artifact saved to output/ when done.\n")

    # Fast preflight to avoid expensive launches when Modal volume is missing files.
    preflight = check_data_ready.remote()
    if not preflight["tokenizer_ok"] or preflight["train_shards"] == 0 or preflight["val_shards"] == 0:
        print("Preflight failed: required data files are missing in Modal volume.")
        print(
            f"  tokenizer_ok={preflight['tokenizer_ok']} path={preflight['tokenizer_path']}\n"
            f"  train_shards={preflight['train_shards']} val_shards={preflight['val_shards']} "
            f"dataset_dir={preflight['dataset_dir']}"
        )
        print("\nRun once to populate this Modal app/account volume:")
        print("  modal run alphagrad/modal_runner.py")
        raise SystemExit(1)

    result = run_exploration.remote(script_content, config, gpu)

    print(result["log"])

    if result["bpb"] is not None:
        print(f"\nFinal val_bpb: {result['bpb']:.4f}")
    else:
        print("\nval_bpb not found in log output.")
    if result.get("best_bpb") is not None:
        print(f"Best observed val_bpb: {result['best_bpb']:.4f}")
    if result.get("serialized_model_bytes") is not None:
        print(f"Serialized model size (from logs): {result['serialized_model_bytes'] / 1e6:.2f} MB")

    artifact_size_mb: float | None = None
    if result["artifact"]:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(exist_ok=True)
        label = config.get("RUN_ID", "latest")
        source_name = result.get("artifact_name") or "final_model.int8.ptz"
        ext = ".int6.ptz" if source_name.endswith(".int6.ptz") else ".int8.ptz"
        out_path = out_dir / f"final_model_{label}{ext}"
        out_path.write_bytes(result["artifact"])
        artifact_size_mb = len(result["artifact"]) / 1e6
        print(f"Artifact saved → {out_path}  ({artifact_size_mb:.2f} MB)")
    else:
        print("No artifact (final_model.int8.ptz) was produced by the run.")

    # Persist one-line run record after every run.
    # Single file: parameter_golf/output/iteration_log.txt
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    log_file = out_dir / "iteration_log.txt"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_label = str(config.get("RUN_ID", "latest"))
    approach = f"{approach}::{script_path.name}"
    if config:
        approach += f" + overrides={config}"
    # Include salient run characteristics when present in stdout.
    for key in ("attention_mode:", "tie_embeddings:", "early_stop:", "train_batch_tokens:"):
        for line in result["log"].splitlines():
            if key in line:
                approach += f" | {line.strip()}"
                break
    best_bpb_text = f"{result['best_bpb']:.6f}" if result.get("best_bpb") is not None else "N/A"
    final_bpb_text = f"{result['bpb']:.6f}" if result.get("bpb") is not None else "N/A"
    artifact_size_text = f"{artifact_size_mb:.3f} MB" if artifact_size_mb is not None else "N/A"
    serialized_size_text = (
        f"{result['serialized_model_bytes'] / 1e6:.3f} MB"
        if result.get("serialized_model_bytes") is not None
        else "N/A"
    )
    status = "artifact_ok" if result["artifact"] else "no_artifact"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{run_time} | run_id={run_label} | approach={approach} | "
            f"best_observed_bpb={best_bpb_text} | final_logged_bpb={final_bpb_text} | "
            f"artifact_size={artifact_size_text} | serialized_model_size={serialized_size_text} | "
            f"status={status}\n"
        )
    print(f"Run record appended → {log_file}")
