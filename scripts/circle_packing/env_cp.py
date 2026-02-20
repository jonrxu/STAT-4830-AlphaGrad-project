from types import SimpleNamespace
from typing import Any

from tinker_cookbook.utils import logtree

from tasks.alphaevolve_cp.task import CirclePackingTask
from tasks.alphaevolve_cp.prompt import CP_IMPROVEMENT_TEMPLATE
from tasks.alphaevolve_cp.verifier import validate_packing
from tinker_cookbook.recipes.ttt.state import CirclePackingState, State
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d
    
default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 1,
        "rew_type": "linear",
        "fail_score": 0.0,
        "eval_timeout": 305,
        "n_item": 26,
        "worst_perf_log": 0.0,
        "pin_cores": False
    }
}

def verify_cp(
    generation: str,
    step: int,
    num_cpus_per_task: int = 1,
    problem_idx: int = 26,
    eval_timeout: int = 300,
    log_path: str = "",
    state: CirclePackingState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["n_item"] = problem_idx
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    task = CirclePackingTask(config_ns, log_path)
    out = task.compute_score(generation, step=step, state=state)

    # Convert [centers, radii, sum] to [[x,y,r], ...] format
    circles = None
    raw_constr = out.get("result_construction")
    if raw_constr and len(raw_constr) >= 2:
        centers, radii = raw_constr[0], raw_constr[1]
        try:
            circles = [[float(centers[i][0]), float(centers[i][1]), float(radii[i])] for i in range(len(radii))]
        except Exception:
            circles = None

    return {
        "sum_radii": out["score"],
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out["performance"],
        "result_construction": circles,
        "stdout": out.get("stdout", ""),
    }


def parse_cp_problem_idx(problem_idx: str) -> tuple[int, str | None]:
    """Parse problem_idx like '26', '32'"""
    return int(problem_idx), "v1"


class CirclePackingEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n, improvement_version = parse_cp_problem_idx(self.problem_idx)
        self.problem_idx = n
        self.improvement_version = improvement_version

    def _get_improvement_prompt(self, state: CirclePackingState) -> str:
        """Build contextual improvement prompt with value context."""
        import inspect
        validator_src = inspect.getsource(validate_packing)
        target = 2.636 if self.problem_idx == 26 else 2.940
        
        # state_only mode: show construction but not code (forces model to write fresh code)
        hide_code = "state_only" in str(self.improvement_version) if self.improvement_version else False
        has_code = state.code and state.code.strip() and not hide_code
        
        # Value context: show before/after if we have parent values
        if state.parent_values and state.value is not None:
            before_sum = state.parent_values[0]
            after_sum = state.value
            value_ctx = f"\nHere are the sum of radii before and after running the code above (higher is better): {before_sum:.6f} -> {after_sum:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {target - after_sum:.6f}. Further improvements will also be generously rewarded."
        elif state.value is not None:
            value_ctx = f"\nCurrent sum of radii (higher is better): {state.value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {target - state.value:.6f}. Further improvements will also be generously rewarded."
        else:
            value_ctx = f"\nTarget sum of radii: {target}"
        # Show previous stdout if available
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        prompt = CP_IMPROVEMENT_TEMPLATE
        prompt = prompt.replace("<<<N>>>", str(self.problem_idx))
        prompt = prompt.replace("<<<VALIDATOR_SRC>>>", validator_src)
        
        if has_code:
            prompt = prompt.replace("<<<LAST_CODE>>>", state.code)
        else:
            prompt = prompt.replace("<<<LAST_CODE>>>", "# No previous code available.")
        
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        return prompt

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int = 1,
        problem_idx: int = 26,
        eval_timeout: int = 300,
        log_path: str = "",
        state: CirclePackingState = None,
        **kwargs
    ) -> dict[str, Any]:
        return verify_cp(generation, step, num_cpus_per_task, problem_idx, eval_timeout, log_path, state)

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "problem_idx": self.problem_idx,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "state": self.state,
        }

    def _get_timeout_response(self) -> dict[str, Any]:
        return {
            "sum_radii": 0.0,
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
            "result_construction": None,
            "stdout": "",
        }

    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "sum_radii": 0.0,
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
            "result_construction": None,
            "stdout": "",
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        sum_radii = outs.get("sum_radii", outs.get("score", 0.0))
        return sum_radii if correctness > 0 else 0.0

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> CirclePackingState:
        parent_state = self.initial_state
        parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []
        return CirclePackingState(
            timestep=step_idx,
            construction=outs.get("result_construction"),
            code=parsed_code,
            value=outs.get("sum_radii", outs.get("score", 0.0)),  # higher = better
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        sum_radii = outs.get("sum_radii", outs.get("score", 0.0))
        correctness = outs.get("correctness", 0.0)
        return {
            "format": correct_format,
            "score": sum_radii,
            "correctness": correctness,
            "correct": correctness,
            "performance": sum_radii if correctness > 0 else None,  # sum of radii, None for incorrect
            "performance/best": sum_radii if correctness > 0 else None,  # higher is better, use max
            "initial_performance": self.initial_state.value,
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message['content'],
            "ref": outs.get("msg", ""),
        }

