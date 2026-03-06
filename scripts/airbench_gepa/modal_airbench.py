#!/usr/bin/env python3
"""Modal execution harness for AirBench candidate scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "airbench-gepa"
DEFAULT_GPU = "A100-40GB"
REMOTE_DATA_DIR = "/vol/cifar10"
TORCH_WHEEL_INDEX = "https://download.pytorch.org/whl/cu124"

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "torchvision==0.19.1",
    extra_index_url=TORCH_WHEEL_INDEX,
)

cifar_volume = modal.Volume.from_name(
    "airbench-cifar10",
    create_if_missing=True,
    version=2,
)


def _tail(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _extract_last_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    payload = json.loads(stdout.strip())
    if not isinstance(payload, dict):
        raise ValueError("candidate output JSON must be an object")
    return payload


def build_script_args(
    *,
    target_accuracy: float,
    trials: int,
    warmup_trials: int,
    data_dir: str = REMOTE_DATA_DIR,
    json_only: bool = True,
    verbose: bool = False,
) -> list[str]:
    args = [
        "--data-dir",
        data_dir,
        "--trials",
        str(trials),
        "--warmup-trials",
        str(warmup_trials),
        "--target-accuracy",
        str(target_accuracy),
    ]
    if json_only:
        args.append("--json-only")
    if verbose:
        args.append("--verbose")
    return args


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={REMOTE_DATA_DIR: cifar_volume},
    timeout=60 * 30,
    cpu=8,
    memory=32768,
)
def run_airbench_candidate(
    solver_code: str,
    script_args: list[str],
    timeout_seconds: int = 60 * 15,
    print_subprocess_logs: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="airbench_candidate_") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(solver_code, encoding="utf-8")

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path), *script_args],
                cwd=tmpdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "failure_type": "timeout",
                "message": f"candidate timed out after {timeout_seconds}s",
                "runtime_seconds": time.time() - started_at,
                "stdout_tail": _tail(exc.stdout or ""),
                "stderr_tail": _tail(exc.stderr or ""),
            }

        runtime_seconds = time.time() - started_at
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if print_subprocess_logs and stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if print_subprocess_logs and stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)

        # Persist dataset downloads cached under the volume mount.
        cifar_volume.commit()

        if completed.returncode != 0:
            return {
                "ok": False,
                "failure_type": "runtime_error",
                "message": f"candidate exited with code {completed.returncode}",
                "runtime_seconds": runtime_seconds,
                "returncode": completed.returncode,
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }

        try:
            payload = _extract_last_json(stdout)
        except Exception as exc:
            return {
                "ok": False,
                "failure_type": "invalid_json",
                "message": f"could not parse candidate JSON output: {exc}",
                "runtime_seconds": runtime_seconds,
                "returncode": completed.returncode,
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }

        payload["_modal"] = {
            "app_name": APP_NAME,
            "gpu": DEFAULT_GPU,
            "data_dir": REMOTE_DATA_DIR,
            "runtime_seconds": runtime_seconds,
        }

        return {
            "ok": True,
            "runtime_seconds": runtime_seconds,
            "returncode": completed.returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "result": payload,
        }


@app.local_entrypoint(name="smoke")
def smoke(
    seed_path: str = str(Path(__file__).with_name("seeds") / "airbench94_baseline.py"),
    trials: int = 1,
    warmup_trials: int = 1,
    target_accuracy: float = 0.94,
) -> None:
    solver_code = Path(seed_path).read_text(encoding="utf-8")
    response = run_airbench_candidate.remote(
        solver_code=solver_code,
        script_args=build_script_args(
            target_accuracy=target_accuracy,
            trials=trials,
            warmup_trials=warmup_trials,
            json_only=True,
            verbose=True,
        ),
        print_subprocess_logs=True,
    )
    print(json.dumps(response, indent=2, sort_keys=True, default=str))
