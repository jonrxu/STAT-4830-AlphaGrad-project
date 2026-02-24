# Week 6 Report — GEPA for CP26 Circle Packing

## Problem Statement

### What are you optimizing?
We optimize **CP26 circle packing**: place **26 circles** inside the unit square $[0,1]\times[0,1]$ with **no overlaps** and **full containment**, maximizing the **sum of radii**:
- Decision variables: circle centers $(x_i, y_i)$ and radii $r_i$ for $i=1,\dots,26$.
- Objective: maximize $\sum_{i=1}^{26} r_i$.
- In our pipeline, the model does **not** output $(x_i,y_i,r_i)$ directly. Instead, it outputs **Python solver code** (a script) that computes a candidate packing and prints JSON. We execute that code locally and score it with an independent verifier.

### Why does this problem matter?
Circle packing is a useful benchmark for **verifiable LLM-driven optimization**:
- The objective is a single numeric value (sum of radii) and constraints are deterministic, enabling **automatic evaluation** with no human labels.
- It tests whether LLMs can generate **working algorithms/programs** under hard constraints, which is representative of many “LLM writes code that must pass a checker” tasks.

### How will you measure success?
We measure:
1. **Solution quality:** best verified score $\max \sum r_i$ obtained during a run.
2. **Efficiency:** how quickly/cheaply we reach strong scores (e.g., best score vs. number of evaluations, runtime, and API calls).
We also track reliability:
- **Validity rate:** fraction of candidates that run and produce a valid 26-circle packing.
- **Failure types:** runtime error, timeout, invalid JSON, wrong number of circles, overlap, out-of-bounds.

### What are your constraints?
- **Strict output contract:** solver must print valid JSON:
  ```json
  {"circles": [[x,y,r], ...], "sum_radii": float}
  ```
  with **exactly 26** circles.
- **Safety/runtime:** candidate code runs in a subprocess with a **300s timeout** and is evaluated as untrusted code.
- **Independent scoring:** we do not trust a solver’s claimed score; we recompute and validate geometry.
- **Compute budget:** we call an LLM API. GEPA uses a *reflection* LLM to mutate code and the solver code can call an LLM at most once per execution (to control cost).

### What data do you need?
No dataset is required. The “data” is generated online:
- Seed solver code (initial program).
- GEPA-generated solver programs.
- Verifier outputs (validity + score + diagnostics).
- Optional artifacts written by the solver (e.g., `best_score.json`) and GEPA outputs (best solver code).

### What could go wrong?
- **Invalid code dominates**: most candidates may crash, print malformed JSON, or violate the output contract.
- **Solver “lies” about score**: model may print a high `sum_radii` without a valid packing; we detect and reject this.
- **Floating point edge cases**: borderline overlaps or boundary violations require stable tolerances.
- **API limits/cost**: rate limits can halt runs; model choice affects both cost and capability.
- **Search stagnation**: GEPA mutations may not explore enough diversity (especially without tracking prompt history).

---

## Technical Approach

### Mathematical formulation
We solve:
$$
\max_{\{x_i,y_i,r_i\}} \ \sum_{i=1}^{26} r_i
$$
Subject to containment for each circle:
$$
r_i \le x_i \le 1-r_i,\quad r_i \le y_i \le 1-r_i,\quad r_i\ge 0
$$
and non-overlap for all $i\ne j$:
$$
\sqrt{(x_i-x_j)^2+(y_i-y_j)^2} \ge r_i+r_j.
$$

### Algorithm choice and justification (GEPA “optimize_anything”)
We use GEPA’s `optimize_anything` in **Agent Architecture Evolution** mode:
- A **reflection LLM** proposes edits to the solver program (`solver_code`).
- The **solver program** is executed locally and may call an `llm_function(prompt)` **once** to ask for a strategy; all geometry is computed locally (NumPy/SciPy).
- A deterministic **evaluator** runs the candidate program and verifies constraints independently.

Why GEPA?
- It avoids training weights and instead performs **structured search over programs**, guided by verifier feedback.
- It matches the project goal of improving **best score per unit budget**, since each evaluation is measurable and comparable.

### Implementation strategy (including PyTorch)
**Current implementation (Week 6):**
- Python orchestration + local execution:
  - `optimize_anything(...)` with `GEPAConfig`, `EngineConfig`, and `ReflectionConfig`.
  - `EngineConfig(max_metric_calls=...)` controls how many candidates are evaluated; `cache_evaluation=True` reduces repeated work.
- Candidate evaluation:
  - Write candidate solver code to a temp `.py` file.
  - Run via `subprocess.run(..., timeout=300)`.
  - Parse stdout JSON and validate.

**Where PyTorch fits (strategy for extension):**
- GEPA itself does not require PyTorch, but PyTorch can improve throughput and enable future methods:
  1. **Vectorized validation**: compute all pairwise distances with Torch tensors for speed on GPU for large $n$ (generalization beyond 26).
  2. **Differentiable relaxation**: represent penalty terms for overlaps/boundary violations and optimize continuous parameters with gradient descent (a complementary baseline).
  3. **Prompt tuning / small-model policy**: if we later train a “prompt generator” or a student model, we would implement training loops in PyTorch.

### Validation methods
We validate at multiple levels:
- **Independent geometry checks**:
  - Boundary checks for each circle.
  - Pairwise overlap checks using a tolerance $10^{-9}$.
- **Score integrity check**: compare solver’s claimed `sum_radii` with our recomputed sum (flag mismatches).
- **Smoke tests**:
  - Evaluate the seed solver standalone to ensure the end-to-end execution and JSON parsing work.
- **Artifact checks**: confirm exactly 26 circles are produced.

### Resource requirements and constraints
- **CPU**: local evaluation, overlap checks, and (optional) SciPy optimization inside the solver.
- **LLM API**:
  - Reflection calls (to mutate code).
  - Solver calls (at most one per candidate run) to request a packing strategy.
- **Budgets used in notebook**:
  - OpenAI o3-mini run configured with `BUDGET=100` (intended strict 1:1 reflection/solver accounting).
  - Gemini run configured with `max_metric_calls=50` (terminated early due to rate limits).

---

## Initial Results

### Evidence the implementation works
- The evaluator executes generated code, parses JSON output, and independently verifies constraints.
- If a solver prints an invalid packing or misreports `sum_radii`, the evaluator assigns score $0$ and returns diagnostics.
- GEPA produces new candidate solver programs over iterations (logged as “Proposed new text for solver_code”).

### Basic performance metrics
**Run A — Reflection LM: OpenAI o3-mini**
- Best verified sum of radii: **2.4698824525** (reported as 2.4699).
- Valid packing: **True**, exactly 26 circles.
- Relative to target 2.636: achievement **≈ 93.7%**.

**Run B — Reflection LM: Gemini 2.5 Flash (budget 50 metric calls)**
- Best score observed in GEPA logs: **2.6359830849** by **Iteration 21** (matches the target/SOTA reference used in the notebook).
- The run hit a **Gemini quota/rate limit** (free-tier limit), which prevented continuing smoothly.

Additional observations from collaborator experiments:
- Gemini 2.5 Lite can reach **2.634292** when randomness is controlled (seeded), but results were not consistently reproducible without fixing randomness.
- Prompting the model with exaggerated “best-so-far” values did not reliably improve performance.

### Test case results
- The seed solver satisfies the evaluator contract (valid JSON format; produces 26 circles).
- GEPA-generated solvers are evaluated deterministically via the same verifier and either pass validation (nonzero score) or fail with clear diagnostics.

### Current limitations
- **Result interface mismatch**: notebook cells attempted to read `result.best_score`, but the GEPA result stores the best program in `result.best_candidate`; this caused `AttributeError` after completion prints.
- **Rate limits** dominated the Gemini run and reduced effective optimization steps.
- **Mutation myopia**: mutation prompts primarily reflect the most recent/best candidate rather than full historical exploration paths.

### Resource usage measurements
- Each evaluation runs a subprocess with a **300s timeout**; most candidates finish much faster.
- API usage is capped by design (reflection calls + ≤1 solver call per evaluation), but free-tier quotas can still halt runs.

### Unexpected challenges
- **Reproducibility**: random seeds inside solver code significantly affect outcomes.
- **Output robustness**: small JSON formatting errors can zero out otherwise good ideas, so format discipline is critical.

---

## Next Steps

### Immediate improvements needed
- Fix the reporting layer: compute “best score” by **re-evaluating `result.best_candidate`** with the evaluator (avoid relying on a nonexistent `best_score` attribute).
- Add structured logs per iteration (best-so-far score, validity, failure type, code length, and API call counts).
- Enforce determinism by requiring fixed seeds and documenting seed handling.

### Technical challenges to address
- Reduce invalid candidates via stronger contract prompting and minimal self-checks inside the solver.
- Handle API throttling gracefully (retry/backoff, batching, or switching tiers / local inference).
- Improve exploration by incorporating more history than just the latest best candidate.

### Questions you need help with
- What is the best efficiency metric: score vs. wall-clock, vs. #API calls, vs. tokens?
- How much solver-side optimization (SciPy) is worth it relative to generating more candidates?

### Alternative approaches to try
- **History-aware GEPA**: archive prompts/programs to avoid missing good branches and to prevent rediscovery.
- **Two-model setup**: cheaper model for reflection and a stronger model for solver calls (or vice versa).
- **PyTorch baseline**: differentiable penalty optimization of circle parameters as a comparison.

### What you’ve learned so far
- Independent verification is essential for honest optimization.
- Model choice and quota constraints can dominate the outcome: Gemini reflection reached near-target quickly but was limited by free-tier quotas; o3-mini was more stable but plateaued lower.
- Reproducibility and output-format robustness are as important as “smart” optimization ideas.
