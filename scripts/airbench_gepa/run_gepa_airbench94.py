#!/usr/bin/env python3
"""Optimize the AirBench 94-percent CIFAR-10 script with GEPA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import gepa.optimize_anything as oa
import modal

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from airbench_evaluator import AirbenchEvalConfig, AirbenchEvalResult, evaluate_solver_code
    from modal_airbench import app, run_airbench_candidate
else:
    from .airbench_evaluator import AirbenchEvalConfig, AirbenchEvalResult, evaluate_solver_code
    from .modal_airbench import app, run_airbench_candidate


DEFAULT_SEED_PATH = Path(__file__).with_name("seeds") / "airbench94_baseline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-solver-path",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="Seed candidate script to optimize.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/airbench/gepa_runs") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=94.0,
        help="Target CIFAR-10 TTA accuracy. Values >1 are treated as percentages.",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of scored AirBench trials per evaluation.")
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=1,
        help="Warmup trials to run before scored trials to amortize compilation.",
    )
    parser.add_argument(
        "--verify-trials",
        type=int,
        default=3,
        help="Number of scored trials for the final best-candidate verification pass.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60 * 15)
    parser.add_argument("--max-metric-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="openai/gpt-5.4",
        help="LiteLLM model used for GEPA reflection.",
    )
    parser.add_argument(
        "--cache-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GEPA evaluator caching.",
    )
    parser.add_argument(
        "--modal-show-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show Modal app/image logs while the run is active.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate the seed candidate once and stop.",
    )
    return parser.parse_args()


def normalize_target_accuracy(raw_value: float) -> float:
    return raw_value / 100.0 if raw_value > 1.0 else raw_value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def write_progress_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "metric_call",
        "candidate_kind",
        "score",
        "best_score_so_far",
        "mean_accuracy",
        "best_accuracy_so_far",
        "mean_time_seconds",
        "best_target_mean_time_so_far",
        "meets_target",
        "is_new_best_score",
        "is_first_target_hit",
        "elapsed_wall_clock_seconds",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def load_seed_solver(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Seed solver not found: {path}")
    return path.read_text(encoding="utf-8")


def build_objective(target_accuracy: float) -> str:
    return (
        "Optimize `solver_code` to train CIFAR-10 on a single NVIDIA A100-40GB GPU "
        f"to at least {100.0 * target_accuracy:.2f}% test-time-augmented accuracy as quickly as possible."
    )


def build_background(target_accuracy: float, trials: int, warmup_trials: int) -> str:
    return (
        "Runtime environment:\n"
        "- Evaluator runs the script remotely on Modal using Python 3.11, torch 2.4.1, torchvision 0.19.1.\n"
        "- Hardware target is one NVIDIA A100-40GB GPU.\n"
        "- CIFAR-10 is cached on a Modal Volume mounted at /vol/cifar10.\n\n"
        "Output contract:\n"
        "- Script must accept CLI flags: --data-dir, --trials, --warmup-trials, --target-accuracy, --json-only.\n"
        "- Final stdout line must be a JSON object containing at least: mean_accuracy, mean_time_seconds, trials.\n"
        "- mean_accuracy may be reported as fraction or percentage; evaluator normalizes it.\n"
        "- mean_time_seconds must be positive.\n\n"
        "Scoring:\n"
        f"- Target accuracy is {100.0 * target_accuracy:.2f}%.\n"
        "- Any candidate that meets the target beats every candidate that misses it.\n"
        "- Among target-meeting candidates, lower mean_time_seconds is better.\n\n"
        "Evaluation protocol:\n"
        f"- Each score uses {trials} measured trial(s) after {warmup_trials} warmup trial(s).\n"
        "- Do not hardcode outputs, skip training, or fake metrics.\n"
        "- Prefer meaningful improvements to architecture, optimizer, augmentation, batch sizing, compile behavior, "
        "or schedule. Avoid adding non-standard dependencies."
    )


def summarize_result(result: AirbenchEvalResult, metric_call: int, candidate_kind: str) -> dict[str, Any]:
    row = {
        "metric_call": metric_call,
        "candidate_kind": candidate_kind,
        "score": result.score,
        "valid": result.valid,
        "failure_type": result.failure_type,
        "message": result.message,
        "runtime_seconds": result.runtime_seconds,
        "mean_accuracy": result.mean_accuracy,
        "mean_time_seconds": result.mean_time_seconds,
        "trials": result.trials,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    if "meets_target" in result.extra:
        row["meets_target"] = result.extra["meets_target"]
    if "accuracy_margin" in result.extra:
        row["accuracy_margin"] = result.extra["accuracy_margin"]
    return row


def annotate_progress(
    row: dict[str, Any],
    eval_history: list[dict[str, Any]],
    *,
    elapsed_wall_clock_seconds: float,
) -> dict[str, Any]:
    best_score_so_far = row["score"]
    best_accuracy_so_far = row["mean_accuracy"]
    best_target_mean_time_so_far = row["mean_time_seconds"] if row.get("meets_target") else None
    previous_target_seen = False

    if eval_history:
        last = eval_history[-1]
        best_score_so_far = max(float(last["best_score_so_far"]), float(row["score"]))
        prev_best_accuracy = last.get("best_accuracy_so_far")
        if prev_best_accuracy is None:
            best_accuracy_so_far = row["mean_accuracy"]
        elif row["mean_accuracy"] is None:
            best_accuracy_so_far = prev_best_accuracy
        else:
            best_accuracy_so_far = max(float(prev_best_accuracy), float(row["mean_accuracy"]))

        prev_best_target_time = last.get("best_target_mean_time_so_far")
        if prev_best_target_time is not None:
            best_target_mean_time_so_far = float(prev_best_target_time)
        if row.get("meets_target") and row.get("mean_time_seconds") is not None:
            curr_time = float(row["mean_time_seconds"])
            if best_target_mean_time_so_far is None or curr_time < best_target_mean_time_so_far:
                best_target_mean_time_so_far = curr_time
        previous_target_seen = any(bool(prev.get("meets_target")) for prev in eval_history)

    row["elapsed_wall_clock_seconds"] = elapsed_wall_clock_seconds
    row["best_score_so_far"] = best_score_so_far
    row["best_accuracy_so_far"] = best_accuracy_so_far
    row["best_target_mean_time_so_far"] = best_target_mean_time_so_far
    row["is_new_best_score"] = (
        not eval_history or float(row["score"]) > float(eval_history[-1]["best_score_so_far"])
    )
    row["is_first_target_hit"] = bool(row.get("meets_target")) and not previous_target_seen
    return row


def build_milestones(eval_history: list[dict[str, Any]]) -> dict[str, Any]:
    milestones: dict[str, Any] = {
        "total_metric_calls_observed": len(eval_history),
        "total_wall_clock_seconds": eval_history[-1]["elapsed_wall_clock_seconds"] if eval_history else 0.0,
    }
    if not eval_history:
        return milestones

    best_score_row = max(eval_history, key=lambda row: float(row["score"]))
    best_accuracy_row = max(
        (row for row in eval_history if row.get("mean_accuracy") is not None),
        key=lambda row: float(row["mean_accuracy"]),
        default=None,
    )
    first_target_row = next((row for row in eval_history if row.get("meets_target")), None)
    fastest_target_row = min(
        (row for row in eval_history if row.get("meets_target") and row.get("mean_time_seconds") is not None),
        key=lambda row: float(row["mean_time_seconds"]),
        default=None,
    )

    milestones.update(
        {
            "best_score_metric_call": best_score_row["metric_call"],
            "best_score_elapsed_wall_clock_seconds": best_score_row["elapsed_wall_clock_seconds"],
            "best_score": best_score_row["score"],
        }
    )
    if best_accuracy_row is not None:
        milestones.update(
            {
                "best_accuracy_metric_call": best_accuracy_row["metric_call"],
                "best_accuracy_elapsed_wall_clock_seconds": best_accuracy_row["elapsed_wall_clock_seconds"],
                "best_accuracy": best_accuracy_row["mean_accuracy"],
            }
        )
    if first_target_row is not None:
        milestones.update(
            {
                "first_target_metric_call": first_target_row["metric_call"],
                "first_target_elapsed_wall_clock_seconds": first_target_row["elapsed_wall_clock_seconds"],
                "first_target_score": first_target_row["score"],
                "first_target_mean_accuracy": first_target_row["mean_accuracy"],
                "first_target_mean_time_seconds": first_target_row["mean_time_seconds"],
            }
        )
    else:
        milestones.update(
            {
                "first_target_metric_call": None,
                "first_target_elapsed_wall_clock_seconds": None,
                "first_target_score": None,
                "first_target_mean_accuracy": None,
                "first_target_mean_time_seconds": None,
            }
        )
    if fastest_target_row is not None:
        milestones.update(
            {
                "fastest_target_metric_call": fastest_target_row["metric_call"],
                "fastest_target_elapsed_wall_clock_seconds": fastest_target_row["elapsed_wall_clock_seconds"],
                "fastest_target_mean_accuracy": fastest_target_row["mean_accuracy"],
                "fastest_target_mean_time_seconds": fastest_target_row["mean_time_seconds"],
            }
        )
    else:
        milestones.update(
            {
                "fastest_target_metric_call": None,
                "fastest_target_elapsed_wall_clock_seconds": None,
                "fastest_target_mean_accuracy": None,
                "fastest_target_mean_time_seconds": None,
            }
        )
    return milestones


def get_best_solver_code(result_obj: Any) -> str:
    best = result_obj.best_candidate
    if isinstance(best, str):
        return best
    if isinstance(best, dict):
        if "solver_code" in best and isinstance(best["solver_code"], str):
            return best["solver_code"]
        for value in best.values():
            if isinstance(value, str):
                return value
    raise ValueError("Could not resolve solver code from result.best_candidate.")


def _run_with_modal_context(args: argparse.Namespace, eval_cfg: AirbenchEvalConfig) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.perf_counter()

    seed_solver_code = load_seed_solver(args.seed_solver_path)
    objective = build_objective(eval_cfg.normalized_target_accuracy())
    background = build_background(
        target_accuracy=eval_cfg.normalized_target_accuracy(),
        trials=eval_cfg.trials,
        warmup_trials=eval_cfg.warmup_trials,
    )

    write_json(
        args.run_dir / "run_config.json",
        {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "args": vars(args),
            "eval_config": asdict(eval_cfg),
            "objective": objective,
            "background": background,
        },
    )
    (args.run_dir / "seed_solver.py").write_text(seed_solver_code, encoding="utf-8")

    eval_history: list[dict[str, Any]] = []

    seed_eval = evaluate_solver_code(seed_solver_code, eval_cfg, run_airbench_candidate)
    seed_row = summarize_result(seed_eval, metric_call=1, candidate_kind="seed")
    eval_history.append(
        annotate_progress(
            seed_row,
            eval_history=[],
            elapsed_wall_clock_seconds=time.perf_counter() - run_started_at,
        )
    )
    write_json(args.run_dir / "seed_eval.json", seed_eval.as_side_info())

    if args.dry_run:
        write_jsonl(args.run_dir / "eval_log.jsonl", eval_history)
        write_progress_csv(args.run_dir / "progress_curve.csv", eval_history)
        write_json(args.run_dir / "milestones.json", build_milestones(eval_history))
        print(
            f"[dry-run] seed accuracy={seed_eval.mean_accuracy} "
            f"time={seed_eval.mean_time_seconds} score={seed_eval.score}"
        )
        print(f"[dry-run] artifacts: {args.run_dir}")
        return 0

    metric_call_counter = len(eval_history)

    def evaluator(candidate: dict[str, str], opt_state: oa.OptimizationState | None = None):
        nonlocal metric_call_counter
        metric_call_counter += 1
        solver_code = candidate.get("solver_code", "")
        eval_result = evaluate_solver_code(solver_code, eval_cfg, run_airbench_candidate)
        row = summarize_result(eval_result, metric_call_counter, candidate_kind="gepa")
        if opt_state is not None:
            row["best_example_evals_seen"] = len(opt_state.best_example_evals)
        row = annotate_progress(
            row,
            eval_history=eval_history,
            elapsed_wall_clock_seconds=time.perf_counter() - run_started_at,
        )
        eval_history.append(row)
        oa.log(
            "metric_call=", metric_call_counter,
            "score=", eval_result.score,
            "mean_accuracy=", eval_result.mean_accuracy,
            "mean_time_seconds=", eval_result.mean_time_seconds,
            "failure_type=", eval_result.failure_type,
        )
        return eval_result.score, eval_result.as_side_info()

    config = oa.GEPAConfig(
        engine=oa.EngineConfig(
            run_dir=str(args.run_dir),
            seed=args.seed,
            display_progress_bar=True,
            max_metric_calls=args.max_metric_calls,
            cache_evaluation=args.cache_evaluation,
            capture_stdio=False,
        ),
        reflection=oa.ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_minibatch_size=1,
        ),
    )

    result = oa.optimize_anything(
        seed_candidate={"solver_code": seed_solver_code},
        evaluator=evaluator,
        objective=objective,
        background=background,
        config=config,
    )

    best_solver_code = get_best_solver_code(result)
    verify_cfg = replace(eval_cfg, trials=args.verify_trials)
    best_verified = evaluate_solver_code(best_solver_code, verify_cfg, run_airbench_candidate)

    (args.run_dir / "best_solver.py").write_text(best_solver_code, encoding="utf-8")
    write_json(args.run_dir / "best_verified_eval.json", best_verified.as_side_info())
    write_json(args.run_dir / "gepa_result.json", result.to_dict())
    write_jsonl(args.run_dir / "eval_log.jsonl", eval_history)
    write_progress_csv(args.run_dir / "progress_curve.csv", eval_history)
    milestones = build_milestones(eval_history)
    write_json(args.run_dir / "milestones.json", milestones)

    summary = {
        "num_candidates": result.num_candidates,
        "best_idx": result.best_idx,
        "best_aggregate_score": result.val_aggregate_scores[result.best_idx],
        "best_verified_score": best_verified.score,
        "best_verified_valid": best_verified.valid,
        "best_verified_mean_accuracy": best_verified.mean_accuracy,
        "best_verified_mean_time_seconds": best_verified.mean_time_seconds,
        "reflection_model": args.reflection_model,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "total_wall_clock_seconds": time.perf_counter() - run_started_at,
        "run_dir": str(args.run_dir),
    }
    summary.update(milestones)
    write_json(args.run_dir / "summary.json", summary)

    print("[gepa] optimization complete")
    print(f"[gepa] best aggregate score      : {summary['best_aggregate_score']:.10f}")
    print(f"[gepa] best verified mean acc   : {summary['best_verified_mean_accuracy']}")
    print(f"[gepa] best verified mean time  : {summary['best_verified_mean_time_seconds']}")
    print(f"[gepa] best verified valid      : {summary['best_verified_valid']}")
    print(f"[gepa] run artifacts            : {args.run_dir}")
    return 0


def main() -> int:
    args = parse_args()
    eval_cfg = AirbenchEvalConfig(
        target_accuracy=normalize_target_accuracy(args.target_accuracy),
        trials=args.trials,
        warmup_trials=args.warmup_trials,
        timeout_seconds=args.timeout_seconds,
    )

    if args.modal_show_output:
        with modal.enable_output(), app.run():
            return _run_with_modal_context(args, eval_cfg)
    with app.run():
        return _run_with_modal_context(args, eval_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
