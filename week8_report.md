# Week 8 Report — GEPA for AirBench CIFAR-10

## Problem Statement

### What are you optimizing?
We are optimizing a **full Python training program** for the [AirBench CIFAR-10 benchmark](https://github.com/KellerJordan/cifar10-airbench). The concrete objective in our current pipeline is:

- run on a single **NVIDIA A100-40GB** GPU;
- train and evaluate a CIFAR-10 model under the AirBench-style protocol;
- achieve at least **94.00%** test-time-augmented (TTA) accuracy;
- among programs that reach the threshold, minimize the reported **`mean_time_seconds`**.

The unit we optimize is **`solver_code`**, i.e. a complete standalone Python script. The model does not emit a config or a patch in the final external interface; it emits an entire runnable program that our evaluator sends to Modal and executes on an A100.

This is therefore a **program optimization** problem, not a pure hyperparameter search problem.

### Why does this problem matter?
This benchmark is a good test of LLM-driven optimization for several reasons:

- The output is fully **verifiable**: the candidate script either runs and produces measured accuracy/time, or it fails.
- The objective is **engineering-relevant**: optimize a real training recipe, not a toy text task.
- The task combines **algorithm design**, **systems constraints**, and **runtime robustness**. A candidate must be fast, accurate, and compatible with compiled PyTorch execution on the target GPU.
- The benchmark has a strong known reference implementation, so we can test whether an LLM-driven optimization loop can improve or at least match a strong seed.

### What is the exact success criterion?
The current scoring rule is lexicographic:

1. Candidates that do **not** reach 94.00% are ranked by accuracy only.
2. Any candidate that **does** reach 94.00% beats every candidate that misses it.
3. Among target-meeting candidates, lower `mean_time_seconds` is better.

Implementation note:
- In `scripts/airbench_gepa/airbench_evaluator.py`, this is encoded as:
  - `score = mean_accuracy` if the candidate misses target;
  - `score = 1000 + 1 / mean_time_seconds` if it meets target.
- The `1000 + ...` term is only a ranking device so that every target-meeting candidate sorts above every sub-threshold candidate.

### What are the important constraints?
The evaluator enforces a strict contract on candidate programs:

- Candidate must be a complete Python script.
- Candidate must accept CLI flags:
  - `--data-dir`
  - `--trials`
  - `--warmup-trials`
  - `--target-accuracy`
  - `--json-only`
  - `--preflight`
  - `--verbose`
- Final stdout line must be JSON containing at least:
  - `mean_accuracy`
  - `mean_time_seconds`
  - `trials`
- Candidate must run under:
  - Python 3.11
  - `torch==2.4.1`
  - `torchvision==0.19.1`
  - a single A100 target device
- Candidate must remain compatible with `torch.compile` and compiled half-precision execution.

### What data and artifacts do we need?
There is no supervised dataset in the usual ML sense. Instead, the pipeline produces and consumes:

- a seed program: `scripts/airbench_gepa/seeds/airbench94_baseline.py`
- GEPA-generated candidate programs
- Modal execution logs and runtime metadata
- per-candidate evaluation JSON
- per-run summaries and milestone logs
- multi-agent session artifacts for proposal generation

The main experiment artifacts are written under:
- `data/airbench/gepa_runs/...`

### What can go wrong?
This problem has several realistic failure modes:

- **runtime failure** in compiled PyTorch paths;
- **dtype mismatch** between normalized inputs and half-precision convolution weights;
- **CUDAGraph-related errors** under repeated compiled inference/TTA calls;
- **contract breakage** such as missing required CLI flags or malformed final JSON;
- **semantic regression** where the script runs but accuracy collapses;
- **hardware inconsistency** if Modal provisions the wrong A100 variant;
- **rate limits / quota exhaustion** in the LLM proposal layer;
- **noisy ranking** near the 94.00% threshold when evaluation uses only one measured trial.

---

## Technical Approach

### High-level formulation
We currently study the **seeded** version of the problem first:

- Start from a strong AirBench-style reference script.
- Let GEPA mutate that full program.
- Evaluate each mutated program remotely on Modal.
- Keep the best-scoring program according to the evaluator.

This is an intentional first phase. The goal for Week 8 was not yet a clean seedless discovery experiment; it was to make sure the end-to-end optimization loop works on a real training benchmark and to understand where it breaks.

### Core implementation
The AirBench optimization stack now lives under `scripts/airbench_gepa/`.

#### 1. Remote execution harness
- `scripts/airbench_gepa/modal_airbench.py`

This file is responsible for:
- receiving a candidate script;
- materializing it into a temporary file on Modal;
- executing it with the required CLI flags;
- streaming subprocess logs;
- returning parsed JSON plus runtime metadata;
- enforcing that the actual hardware matches the requested `A100-40GB` target.

A key lesson from this week is that we had to distinguish:
- **benchmark time**: the candidate's own `mean_time_seconds`, typically around `2.5-2.6s` for a good AirBench run;
- **evaluation wall-clock time**: the full cold-start Modal execution, which can exceed one minute because of process startup, `torch.compile`, Triton autotuning, warmup, and orchestration overhead.

#### 2. Evaluator and scoring
- `scripts/airbench_gepa/airbench_evaluator.py`

This file:
- performs local syntax validation;
- dispatches the candidate to Modal;
- parses the returned JSON;
- scores the candidate under the 94%-then-time rule;
- records runtime failures and diagnostic tails;
- retries GPU mismatches when Modal attaches the wrong device.

We added a **preflight execution path** this week so a candidate can fail fast before paying the full warmup/trial cost.

#### 3. GEPA runner
- `scripts/airbench_gepa/run_gepa_airbench94.py`

This is the top-level orchestration script. It:
- loads the seed program;
- loads `.env` so the reflection model can authenticate locally;
- initializes GEPA with full `solver_code` candidates;
- logs per-eval summaries;
- writes run artifacts such as:
  - `summary.json`
  - `milestones.json`
  - `eval_log.jsonl`
  - candidate snapshots under `candidates/`

#### 4. Seed solver
- `scripts/airbench_gepa/seeds/airbench94_baseline.py`

This is a wrapped AirBench-style baseline that:
- exposes the required CLI contract;
- performs preflight, warmup, and scored trials;
- prints final JSON;
- serves as the initial seeded program.

### Evolution of the proposal mechanism

#### Phase 1: single-model full-program mutation
The first working version used GEPA in the most direct form:
- one reflection model;
- one full `solver_code` string;
- candidate program fully rewritten at each mutation step.

This confirmed that GEPA could in fact optimize a complete training script, but it quickly revealed that large whole-file rewrites were high-variance and failure-prone.

#### Phase 2: refiner inner loop
We then added a repair loop so that within one outer GEPA iteration the system could:
- propose a candidate;
- evaluate it;
- if it failed, feed the failure back immediately;
- refine and retry before returning to the outer loop.

This improved the search behavior in principle because failures were no longer always terminal at the proposal stage.

#### Phase 3: multi-agent proposer
The major Week 8 algorithmic change was the introduction of a **team-based proposer**:
- `scripts/airbench_gepa/agent_team_proposer.py`

The team architecture is:
- **Coordinator**: decides which sections of the code to focus on and assigns performance vs. reliability tasks;
- **Performance worker**: proposes targeted speed/accuracy improvements;
- **Reliability worker**: proposes fixes to dtype, compile, CLI, and runtime robustness;
- **Integrator**: merges the section-level edits into a full candidate script;
- **Reviewer**: critiques the integrated candidate before it is returned.

Internally, the code is split into semantic sections such as:
- `setup_code`
- `data_code`
- `model_code`
- `train_eval_code`
- `cli_code`

Externally, GEPA still evaluates a complete standalone script.

This design gave us the behavior we wanted conceptually:
- the system can reason about and edit any section of the program;
- the final artifact remains a clean full program;
- proposal-time memory can incorporate prior failures and prior best candidates.

### Infrastructure fixes completed this week
A large part of the week was spent stabilizing the system. The major fixes were:

1. **`cloudpickle` save failure**
- GEPA initially failed while saving state due to a missing local `cloudpickle` dependency.
- We fixed the AirBench runner to disable unnecessary cloudpickle-based state saving for this use case.

2. **`.env` / OpenAI auth loading**
- Early runs failed because the local reflection process did not see `OPENAI_API_KEY`.
- We added explicit `.env` loading and fail-fast checks in `run_gepa_airbench94.py`.

3. **Verbose runtime logging**
- The first dry runs were too quiet to debug.
- We added verbose streaming for:
  - seed evaluation;
  - candidate evaluation;
  - Modal subprocess logs;
  - per-eval local summaries.

4. **Preflight execution**
- We added `--preflight` to catch bad candidates earlier, especially compile- and dtype-related failures.

5. **Team proposer robustness**
- The first team version had two framework bugs:
  - reviewer was shown a truncated candidate and falsely flagged it as incomplete;
  - integrator parsing was too brittle and failed on extra non-JSON text.
- We fixed this by:
  - removing misleading truncation in reviewer context;
  - saving raw agent outputs;
  - making JSON extraction more tolerant;
  - falling back to the last coherent candidate when a team round fails.

6. **GPU mismatch enforcement**
- Modal sometimes attached `A100-80GB` even when we requested `A100-40GB`.
- This made results non-comparable.
- We added runtime checking in `modal_airbench.py` and retry logic in `airbench_evaluator.py` so mismatched devices are rejected and logged as `gpu_mismatch`.

### Validation and observability
The system now logs enough information to support serious debugging:

- `summary.json`: run-level aggregate metrics;
- `milestones.json`: first/best-hit timestamps and counts;
- `eval_log.jsonl`: per-evaluation timeline and candidate metadata;
- `candidates/metric_*.py`: exact source code evaluated;
- `candidates/metric_*.json`: exact evaluation result per candidate;
- `agent_team/session_###/`: coordinator/worker/integrator/reviewer artifacts and raw model outputs.

This is sufficient to reconstruct:
- what code was proposed;
- what the system was trying to improve;
- why a candidate failed;
- whether a failure was due to the proposal itself, the runtime, or the surrounding infrastructure.

---

## Initial Results

### What we accomplished this week
By the end of Week 8, we had a working AirBench optimization pipeline with:

- full-program GEPA optimization over `solver_code`;
- remote A100 execution on Modal;
- preflight + warmup + scored trials;
- local `.env`-driven reflection model calls;
- optional refiner loop;
- a functioning multi-agent proposal mechanism;
- hardware consistency checks for `A100-40GB`.

This is substantial progress over the start of the week, when the main question was still whether the AirBench benchmark could even be driven through GEPA end-to-end.

### Key technical discovery: benchmark time vs. evaluation wall-clock
One of the most important findings was that the AirBench benchmark time and the actual optimization-loop cost are different quantities.

For example:
- the seed program often reports `mean_time_seconds` near `2.57s`;
- but a cold Modal evaluation can take well over a minute.

The gap is explained by:
- Python startup;
- imports;
- CIFAR tensor loading;
- `torch.compile` setup;
- Triton autotuning / precompilation;
- warmup trial;
- orchestration overhead.

This matters because it means the engineering bottleneck for search is not only model quality; it is also candidate evaluation latency.

### Chronology of major runs

#### Run A: first stable team-based run
- Directory: `data/airbench/gepa_runs/20260306_191944`
- Configuration:
  - `proposal_strategy=team`
  - `team_rounds=1`
  - `max_refinements=0`
  - `max_metric_calls=3`

What happened:
- The team proposer ran end-to-end without the earlier JSON/reviewer crashes.
- The seed remained the best candidate.
- One real team-generated candidate was evaluated and was dramatically worse:
  - best search-time seed accuracy: `0.9389`
  - best verified seed accuracy/time: `0.93997`, `2.5095s`
  - one GEPA candidate collapsed to roughly `10%` accuracy and was slower.

Interpretation:
- The framework was stable enough to produce and test non-seed programs.
- Candidate quality was still poor.

#### Run B: hardware-enforced run where seed crossed target
- Directory: `data/airbench/gepa_runs/20260306_194056`
- Configuration:
  - `proposal_strategy=team`
  - `team_rounds=1`
  - `max_refinements=0`
  - `max_metric_calls=5`

What happened:
- The seed evaluated on the correct `A100-40GB` and hit the target in search-time evaluation:
  - `0.9402` accuracy;
  - `2.5811s` measured time.
- One GEPA candidate failed with a CUDAGraph overwrite error.
- Another candidate was rejected by the new GPU mismatch guard because Modal attached the wrong A100 variant.
- Final best solver was still the seed.

Interpretation:
- The GPU consistency fix worked.
- The system still struggled to produce safe speed-oriented edits.

#### Run C: deeper agentic run with refiner and two-round team proposer
- Directory: `data/airbench/gepa_runs/20260306_220019`
- Configuration from `run_config.json`:
  - `proposal_strategy=team`
  - `team_rounds=2`
  - `max_refinements=2`
  - `max_metric_calls=12`
  - `trials=1`
  - `verify_trials=1`

Headline numbers from `summary.json`:
- `num_true_gepa_candidates_observed = 6`
- `total_evaluated_candidates_observed = 8`
- `total_metric_calls = 12`
- `total_wall_clock_seconds = 1440.04`
- `best_idx = 0` (the seed)
- `best_accuracy = 0.9396999478`
- `best_verified_mean_accuracy = 0.9412999749`
- `best_verified_mean_time_seconds = 2.5725560963`

Important interpretation:
- The “best verified” result is still the seed.
- `best_solver.py` is byte-identical to `seed_solver.py`.
- No non-seed program improved on the seed.

### What the agent team was actually trying to do
The coordinator and workers were not making arbitrary edits. Their proposals were goal-directed.

From `data/airbench/gepa_runs/20260306_220019/agent_team/session_001/round_01_coordinator.json`:
- the coordinator recognized that the current model was missing target by only about `0.03` percentage points;
- it explicitly instructed the team to focus on small, surgical changes;
- it emphasized `train_eval_code`, `data_code`, and runtime robustness.

From the corresponding integrator artifact:
- the proposed edits included:
  - slightly stronger augmentation (`translate`, `cutout`);
  - EMA on the head;
  - richer mirrored 4-crop TTA;
  - safer `torch.load` behavior;
  - harder CLI / JSON handling;
  - more explicit preflight and trial orchestration.

This is important because it shows the agent was reading the context correctly. The proposals were not random; they were just not yet sufficiently safe or effective.

### Candidate failure modes observed
The latest deeper run exposed the dominant remaining failure patterns.

#### 1. Dtype mismatch under compiled half-precision convs
- Candidate artifact: `data/airbench/gepa_runs/20260306_220019/candidates/metric_0004_gepa.json`
- Failure:
  - `Input type (float) and bias type (c10::Half) should be the same`

Interpretation:
- Candidate modified normalization/data handling in a way that promoted activations to `float32` while the compiled model weights and biases remained half-precision.
- This is a real semantic/runtime bug in the candidate.

#### 2. Severe semantic regression despite valid execution
- Candidate artifact: `data/airbench/gepa_runs/20260306_220019/candidates/metric_0005_gepa.json`
- Result:
  - accuracy `0.7887`
  - time `3.4646s`

Interpretation:
- The candidate was valid in the sense that it ran and returned metrics.
- But its training/evaluation logic was much worse than the seed.
- The preflight metrics in the payload already hinted at instability (`sample_tta_accuracy` near random-guess level).

#### 3. CLI contract breakage
- Candidate artifact: `data/airbench/gepa_runs/20260306_220019/candidates/metric_0006_gepa.json`
- Failure:
  - candidate did not accept `--verbose`

Interpretation:
- Even with the explicit contract in the prompt, the model sometimes rewrites the CLI and removes required arguments.
- This is a straightforward preventable failure mode.

#### 4. CUDAGraph overwrite failure
- Candidate artifact: `data/airbench/gepa_runs/20260306_220019/candidates/metric_0008_gepa.json`
- Failure message:
  - accessing tensor output of compiled CUDAGraphs after it has been overwritten by a subsequent run

Interpretation:
- The candidate introduced a TTA / repeated-call pattern that is unsafe under compiled graph reuse.
- This is a realistic systems bug specific to compiled PyTorch execution, not a superficial formatting issue.

### Proposal-layer failure: quota exhaustion
The deeper 12-call run did not really get twelve full rounds of fresh multi-agent reasoning.

Evidence:
- `data/airbench/gepa_runs/20260306_220019/agent_team/session_002/round_02_error.json`
- `data/airbench/gepa_runs/20260306_220019/agent_team/session_003/round_01_error.json`
- and the same pattern through later sessions

The error was:
- `litellm.RateLimitError: ... You exceeded your current quota ...`

Interpretation:
- The team-based proposal system is materially more expensive in LLM calls than the single-agent version.
- After the second session, later sessions were increasingly dominated by quota exhaustion rather than real search.
- Therefore, `max_metric_calls=12` overstates the amount of true agentic exploration performed in that run.

### Current best result and how to interpret it
The current best incumbent is still the seed program.

Most defensible statement based on the latest clean artifacts:
- The seeded AirBench baseline remains the best solver found.
- On successful 40GB runs it is consistently around:
  - `94.0% - 94.1%` TTA accuracy;
  - `~2.57s - 2.58s` measured time.

However, we should be careful not to over-interpret single-run numbers because the current search protocol uses `trials=1` and `verify_trials=1`. Since the seed sits very close to the 94% threshold, one-trial variation can change whether it is treated as “valid” by the search.

### What we learned this week

1. **The infrastructure problem is mostly solved.**
We now have a working benchmark/evaluator/proposal stack with enough logging to debug real candidate failures.

2. **The search problem is now the main bottleneck.**
The remaining challenge is not “can GEPA run AirBench?” It is “can the proposer generate better full programs without repeatedly breaking dtype, CLI, or compiled-runtime assumptions?”

3. **The multi-agent design is directionally reasonable.**
The coordinator and workers are proposing sensible edits given the context. The issue is not lack of task awareness; it is insufficient safety and too much variance in code changes.

4. **The benchmark is noisy around the 94% threshold.**
Single-trial evaluation makes accept/reject decisions unstable when the seed is already very close to the target.

5. **Proposal cost matters.**
The agentic system is more expressive, but it also consumes substantially more LLM budget and can collapse into rate-limit errors before a long search completes.

### Current limitations
- No non-seed improvement has been found yet.
- Search-time scoring is noisy because `trials=1` and `verify_trials=1`.
- Repeated failure classes are still reaching Modal evaluation.
- The multi-agent proposer is expensive enough to trigger quota issues in longer runs.
- The current study is seeded; a seedless from-scratch experiment remains future work.

---

## Next Steps

### Immediate improvements needed
1. **Reduce proposal-layer cost**
- Either use a cheaper model for some internal team roles or reduce the number of agent calls per proposal round.
- The current `gpt-5.4` team stack is too expensive for long runs under the present quota.

2. **Stabilize search-time evaluation near the threshold**
- Move from `trials=1` to at least `trials=2` for search, or otherwise soften the threshold decision rule.
- Right now a candidate just above or below 94% in one trial can be mis-ranked.

3. **Add harder candidate safety gates before Modal**
- Reject candidates that:
  - break the CLI contract;
  - introduce known dtype inconsistencies;
  - use CUDAGraph-unsafe repeated compiled inference patterns.

### Technical challenges to address
- How do we keep the search space as **full programs** while making edits more local and safer?
- How much proposal diversity is actually helpful before code quality collapses?
- Can the refiner and reviewer be made strict enough to prevent obviously bad candidates from consuming full remote evaluations?

### Questions that remain open
- Is the multi-agent proposer actually better than a simpler single-agent proposer once cost is taken into account?
- Should we keep a strong seed-first track and a separate seedless discovery track as two distinct experiments?
- What is the right evaluation budget allocation between:
  - more proposal iterations;
  - more scored trials per iteration;
  - stricter final verification?

### Alternative approaches to try next
1. **Cheaper internal team model / stronger final integrator model**
- Use a smaller model for coordinator and reviewer, while keeping a stronger model only for integrator or final rewrite.

2. **Stricter static and semantic linting before Modal**
- Add local checks for missing CLI flags, suspicious dtype handling, and unsafe TTA patterns.

3. **Longer runs only after proposal cost is controlled**
- More iterations and reflections are only useful if the system can actually sustain them without falling into rate-limit failure.

4. **Seedless experiment after the seeded baseline is stable**
- Once the seeded loop is better understood, run the same pipeline without the AirBench reference as a separate, cleaner scientific experiment.

### What we learned so far
- AirBench is a strong and realistic benchmark for program optimization by LLMs.
- The difference between benchmark speed and search-time evaluation cost is operationally critical.
- The proposal mechanism is now sophisticated enough to generate plausible engineering changes, but not yet safe or efficient enough to beat the seed.
- Week 8 therefore produced a solid optimization framework and a clear map of the remaining bottlenecks, even though it did not yet produce a new best solver.

### Main files and run directories referenced in this report
Code:
- `scripts/airbench_gepa/modal_airbench.py`
- `scripts/airbench_gepa/airbench_evaluator.py`
- `scripts/airbench_gepa/run_gepa_airbench94.py`
- `scripts/airbench_gepa/agent_team_proposer.py`
- `scripts/airbench_gepa/seeds/airbench94_baseline.py`

Representative run artifacts:
- `data/airbench/gepa_runs/20260306_191944`
- `data/airbench/gepa_runs/20260306_194056`
- `data/airbench/gepa_runs/20260306_220019`
