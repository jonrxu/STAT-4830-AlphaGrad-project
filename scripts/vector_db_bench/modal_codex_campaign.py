#!/usr/bin/env python3
"""Run the full Codex superagent campaign entirely on Modal.

Codex bursts, cargo build, and benchmark all run inside a single persistent
Modal container. The workspace is stored in a Modal Volume so runs survive
interruption and can be resumed by re-running the same command.

Prerequisites:
  - Modal authenticated (`modal token new`)
  - OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env (Codex needs one)
  - VECTOR_DB_BENCH_ROOT in your .env (for the local data mount)

Usage:
  # foreground (see output live):
  modal run scripts/vector_db_bench/modal_codex_campaign.py

  # detached (fire and forget — check Modal dashboard for progress):
  modal run --detach scripts/vector_db_bench/modal_codex_campaign.py

  # custom cycles / model:
  modal run scripts/vector_db_bench/modal_codex_campaign.py \\
    --cycles 40 --codex-model o4-mini
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REMOTE_BENCH = "/opt/vector-db-bench"
REMOTE_DATA = "/opt/vdb-data"
REMOTE_SCRIPTS = "/opt/scripts"
REMOTE_RUN_ROOT = "/vdb_runs/campaign"

_in_modal_container = bool(os.environ.get("MODAL_TASK_ID", ""))

if not _in_modal_container:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SCRIPTS_DIR = REPO_ROOT / "scripts" / "vector_db_bench"

    _bench_root = os.environ.get("VECTOR_DB_BENCH_ROOT", "").strip()
    if not _bench_root:
        raise RuntimeError("Set VECTOR_DB_BENCH_ROOT in your .env before running.")
    _bench_path = Path(_bench_root).resolve()
    if not _bench_path.is_dir():
        raise RuntimeError(f"VECTOR_DB_BENCH_ROOT is not a directory: {_bench_path}")

    _data_root = os.environ.get("VECTOR_DB_BENCH_DATA", "").strip()
    _data_path = Path(_data_root).resolve() if _data_root else _bench_path / "data"
    if not _data_path.is_dir():
        raise RuntimeError(f"Data directory missing: {_data_path}")
else:
    REPO_ROOT = Path("/opt")
    SCRIPTS_DIR = Path(REMOTE_SCRIPTS) / "vector_db_bench"
    _bench_path = Path(REMOTE_BENCH)
    _data_path = Path(REMOTE_DATA)

# ---------------------------------------------------------------------------
# Persistent volume for workspace + run artifacts
# ---------------------------------------------------------------------------

campaign_volume = modal.Volume.from_name("vdb-codex-campaign", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: Rust + Node.js + Codex CLI + Python + project scripts + bench data
# ---------------------------------------------------------------------------

_BENCH_IGNORE = [".git", "target", ".idea", ".vscode", "data"]

if not _in_modal_container:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "curl",
            "ca-certificates",
            "build-essential",
            "pkg-config",
            "libssl-dev",
            "clang",
            "git",
            "nodejs",
            "npm",
        )
        .run_commands(
            # Rust toolchain
            'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable',
            "bash -lc 'source /root/.cargo/env && rustc --version && cargo --version'",
            # Codex CLI
            "npm install -g @openai/codex",
            "codex --version",
        )
        .env({
            "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "RUST_BACKTRACE": "1",
        })
        .add_local_dir(str(_bench_path), remote_path=REMOTE_BENCH, ignore=_BENCH_IGNORE)
        .add_local_dir(str(_data_path), remote_path=REMOTE_DATA)
        .add_local_dir(str(SCRIPTS_DIR), remote_path=f"{REMOTE_SCRIPTS}/vector_db_bench")
    )
else:
    image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("vdb-codex-campaign")

# ---------------------------------------------------------------------------
# Campaign function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/vdb_runs": campaign_volume},
    secrets=[modal.Secret.from_dotenv(str(REPO_ROOT / ".env"))],
    timeout=86400,  # 24 hours; re-run to resume if campaign exceeds this
    cpu=4.0,
    memory=16384,
)
def run_campaign(
    cycles: int = 40,
    goal_qps: float = 22000.0,
    codex_model: str = "",
    codex_timeout_seconds: int = 1800,
    quick_benchmark_queries: int = 1000,
) -> dict[str, Any]:
    import json
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_SCRIPTS}/vector_db_bench/qwen3_meta:" + env.get("PYTHONPATH", "")

    script = f"{REMOTE_SCRIPTS}/vector_db_bench/qwen3_meta/run_meta_harness_codex_superagent.py"

    argv = [
        sys.executable, script,
        "--bench-repo", REMOTE_BENCH,
        "--data-dir", REMOTE_DATA,
        "--blank-seed-source", f"{REMOTE_BENCH}/skeleton",
        "--bootstrap-dir", f"{REMOTE_SCRIPTS}/vector_db_bench/qwen3_meta/meta_harness_codex_superagent/bootstrap",
        "--run-root", REMOTE_RUN_ROOT,
        "--cycles", str(cycles),
        "--goal-qps", str(goal_qps),
        "--codex-timeout-seconds", str(codex_timeout_seconds),
        "--quick-benchmark-queries", str(quick_benchmark_queries),
        "--codex-sandbox", "danger-full-access",
        "--no-modal-eval",
    ]
    if codex_model:
        argv += ["--codex-model", codex_model]

    print(f"[campaign] starting: {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, env=env)
    return {"returncode": result.returncode}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    cycles: int = 40,
    goal_qps: float = 22000.0,
    codex_model: str = "",
    codex_timeout_seconds: int = 1800,
    quick_benchmark_queries: int = 1000,
) -> None:
    result = run_campaign.remote(
        cycles=cycles,
        goal_qps=goal_qps,
        codex_model=codex_model,
        codex_timeout_seconds=codex_timeout_seconds,
        quick_benchmark_queries=quick_benchmark_queries,
    )
    print(f"[campaign] finished: {result}")
