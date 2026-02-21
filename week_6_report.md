# GEPA for CP26 Circle Packing (Report Draft)

## Problem Statement

### What are we optimizing?
We optimize **CP26 circle packing in the unit square**: place **26 circles** inside a **1×1 square** (no overlap, fully contained) to **maximize the total packed radius**:

- **Decision variables:** circle centers $(x_i, y_i)$ and radii $r_i$, for $i = 1,\dots,26$
- **Objective:** maximize $\sum_{i=1}^{26} r_i$

In our setup, the LLM does **not** directly output $(x_i,y_i,r_i)$. Instead, it outputs **Python code** (a solver function such as `run_packing()`) that computes a candidate packing, which we then verify and score.

### Why does this problem matter?
Circle packing is a clean benchmark for **verifiable optimization with LLMs**:
- The task has **hard geometric constraints** and a **single numeric score**, so there is no ambiguity in evaluation.
- It tests whether LLMs can propose **effective algorithms/programs**, not just text.
- It is a good proxy for broader tasks where we want models to generate **correct, executable code** that optimizes a measurable objective.

### How will we measure success?
We measure success along two axes:

1. **Solution quality:** best achieved score $ \max \sum r_i $ among valid packings.
2. **Efficiency:** cost to achieve high quality (e.g., score vs. number of samples, runtime, and/or API usage).

We also track intermediate reliability metrics:
- **Validity rate:** fraction of generated programs that execute successfully and produce valid packings.
- **Failure breakdown:** syntax/runtime errors, timeouts, overlaps, out-of-bounds, etc.

### What are our constraints?
- **No training / no model weight updates** for the GEPA phase (we treat the LLM as a black-box generator).
- **Compute budget:** we are calling an **LLM API**, so we must control sampling volume and token usage.
- **Runtime/safety:** generated code must run in a sandbox with strict **timeouts** and restricted execution.
- **Deterministic evaluation:** the verifier must be consistent; numerical tolerances must be handled carefully.

### What data do we need?
This project does not require a labeled dataset. The “data” is generated online:
- **Seed prompt set** (initial population of prompts)
- **Generated solver programs** from the LLM
- **Verifier outputs**: valid/invalid flags, scores, error types, runtime stats
- **Artifacts**: best code per generation, summary tables, plots

### What could go wrong?
- **Invalid code dominates**: the model produces non-runnable or non-conforming outputs, wasting samples.
- **Security issues**: generated code attempts unsafe operations; mitigated via sandboxing.
- **Verifier edge cases**: floating-point tolerance could incorrectly accept/reject near-touching circles.
- **Search collapse**: prompt evolution may converge to prompts that are verbose or brittle, improving format but not score.
- **Hidden costs**: prompt mutation calls could become expensive if not controlled.

---

## Technical Approach

### Mathematical formulation
We solve:

$$
\max_{\{x_i,y_i,r_i\}} \ \sum_{i=1}^{26} r_i
$$

Subject to containment (each circle fully inside the unit square):
$$
r_i \le x_i \le 1 - r_i,\quad r_i \le y_i \le 1 - r_i,\quad r_i \ge 0
$$

And non-overlap for all $i \ne j$:
$$
\sqrt{(x_i-x_j)^2+(y_i-y_j)^2} \ge r_i + r_j
$$

Our verifier checks these constraints and assigns:
- **Reward/score** = $\sum r_i$ if valid
- **Score = 0** (or invalid) if constraints fail or code errors

### Algorithm choice: GEPA (prompt evolution, fixed model)
We use a GEPA-style loop where we **evolve prompts**, not model weights.

At a high level:
1. Maintain a **population of prompts** $\mathcal{P}$.
2. For each prompt $p \in \mathcal{P}$, sample $K$ candidate programs from the LLM.
3. Evaluate each program with the deterministic verifier.
4. Score each prompt by aggregate metrics (e.g., best score, validity rate).
5. Select top prompts and **mutate** them (LLM-assisted prompt rewriting) to form the next generation.

**Justification:** This directly targets the professor’s goal—reaching strong solutions **efficiently**—while avoiding the overhead and complexity of fine-tuning.

### Implementation strategy (Python + modular interfaces; PyTorch-ready)
Even though GEPA does not require weight training, we structure the code so it can later swap between:
- **LLM API** calls (current) and
- a **local PyTorch model** (future, if we switch to GPU credits).

Planned modules:
- `LLMClient`: generates solver code samples given a prompt (and generates mutated prompts).
- `Verifier/SandboxRunner`: safely executes generated code with timeouts and checks constraints.
- `GEPA Loop`: population → evaluate → select → mutate → next generation.
- Logging/artifacts: save prompts, scores, failure types, and best programs per generation.

(Concrete file paths, configs, and hyperparameters will be added once implementation is finalized.)

### Validation methods
We validate correctness at multiple levels:
- **Unit tests** for the verifier (known valid/invalid packings).
- **Sandbox tests** for code execution (timeouts, restricted imports, deterministic behavior).
- **End-to-end smoke tests**: one generation with small $P$ and $K$ to confirm the pipeline runs.
- **Reproducibility checks**: fixed seeds for sampling where supported; consistent tolerances.

### Resource requirements and constraints
- Local CPU for parallel evaluation (Ray/multiprocessing).
- LLM API usage for:
  - solver program generation (main cost driver),
  - prompt mutation (smaller overhead).
- Strict timeouts for each candidate program to cap worst-case runtime.

---

## Initial Results

### Evidence the pipeline works
Before GEPA, our group built a working CP26 pipeline where an LLM generates solver code and a verifier scores it. In that earlier phase, we achieved near–state-of-the-art CP26 scores, demonstrating that:
- the verifier correctly distinguishes valid vs invalid solutions,
- the environment supports repeated sample → execute → score loops,
- the metric meaningfully improves as generation quality improves.

### GEPA implementation status (current iteration)
- **GEPA loop:** in progress; currently implementing population evaluation + prompt mutation + selection.
- **Verifier + sandbox:** reused from earlier work; already functional.

### Basic performance metrics (planned reporting format)
Once GEPA runs are complete, we will report:
- Best score per generation
- Validity rate per generation
- Best score vs total samples
- Failure mode breakdown (syntax error / timeout / overlap / out-of-bounds)
- Resource usage: runtime and API usage (tokens/cost if available)

### Current limitations
- No history/archive yet (prompts may be re-proposed).
- Mutation strategy may initially be naive (e.g., single mutation prompt template).
- We have not yet tuned multi-objective selection (score vs validity vs cost).

### Unexpected challenges (so far)
- Prompting for **runnable, constrained code** is sensitive: small changes can improve formatting but hurt algorithmic creativity.
- Sandbox constraints must be carefully balanced: too strict reduces useful libraries; too loose risks unsafe behavior.

---

## Next Steps

### Immediate improvements needed
- Stabilize the GEPA loop end-to-end and run a **small-scale experiment**:
  - Example debug run: $P=8, K=8, G=5$
- Add logging artifacts per generation (prompts + best program + evaluation summaries).
- Create stronger seed prompts focused on:
  - strict output schema,
  - self-check / repair loop,
  - multi-start / local optimization strategy hints.

### Technical challenges to address
- Better selection criteria than “best score only”:
  - incorporate validity rate (avoid prompts that rarely succeed),
  - incorporate cost (avoid overly long prompts or outputs).
- Improve mutation diversity to prevent convergence to near-identical prompts.

### Questions we need help with
- What is the best objective for prompt selection: best-of-K score, median-of-valid, or a Pareto tradeoff?
- How should we structure verifier feedback to best drive useful prompt mutations?

### Alternative approaches to try
- Two-model setup: cheap model for mutation + strong model for solver generation.
- Hybrid GEPA + search memory: keep a small “elite prompt set” across generations.
- Later extension: use GPU credits to run a local model and reduce API dependence.

### History tracking (planned enhancement)
In the next iteration, we will add prompt-level memory to improve efficiency:
- Maintain a **prompt archive** (hash/canonicalize prompts) to avoid re-evaluating duplicates.
- Cache evaluation results keyed by (prompt, sampling params) to prevent wasted API spend.
- Optionally reject near-duplicates by similarity thresholds.

### What we’ve learned so far
- Deterministic verification makes this a strong testbed for “LLM-as-optimizer.”
- A large fraction of cost is driven by invalid outputs, so prompt design and mutation strategy are crucial.
- GEPA offers a simpler path to efficiency than model fine-tuning, especially under tight compute budgets.
