# Week 8 Self-Critique (OODA) — GEPA for AirBench CIFAR-10

## OBSERVE: Reading the Report Critically
Re-reading the Week 8 report as a first-time reader, the document is much stronger on **systems understanding** than on **optimization success**. The report now clearly explains the problem, the evaluator, the Modal execution model, the distinction between benchmark time and full evaluation wall-clock time, and the evolution from a single-agent GEPA proposer to a multi-agent team-based proposer. That is real technical progress, and the report does a good job of grounding it in concrete scripts and run directories.

However, the central experimental outcome is still weak: **we did not find a better solver than the seed**. A skeptical reader would reasonably say that the project currently demonstrates an increasingly sophisticated optimization harness, but not yet a convincing improvement in the optimized artifact itself. The report is honest about that, which is good, but the burden is now on us to show that the additional complexity we introduced (refiner loop, team proposer, Modal hardware checks, richer logging) is actually buying us something scientifically rather than just making the system harder to operate.

The second issue is that the evaluation protocol is still not strong enough for confident model-selection claims. In the latest runs, the seed sits very close to the 94% threshold, and because search-time scoring used `trials=1`, the exact same seed can fall slightly below or slightly above the constraint depending on the run. That means some of our accept/reject decisions are being driven by single-trial noise rather than stable differences in program quality. So while the report is clear, the evidence base under the report is still somewhat brittle.

The third issue is that the agentic system is not yet cost-aligned with the experimental budget. The report correctly notes that later sessions in the deeper run were dominated by quota errors rather than real proposal/reflection. That means the nominal search depth was not the actual search depth. If a run says `max_metric_calls=12` but only the early sessions performed meaningful multi-agent reasoning before rate limits took over, then our experiment configuration is overstating what was truly tested.

## ORIENT: Analyzing Our Work

### Strengths
- **The infrastructure story is now credible.** We moved from “can this benchmark even run through GEPA?” to a functioning end-to-end system with Modal execution, evaluator contracts, preflight checks, GPU mismatch enforcement, and artifact logging. That is real engineering progress.
- **The report is evidence-linked.** Unlike earlier iterations, the Week 8 report points directly to the scripts and run directories that support the claims. A reader can inspect `scripts/airbench_gepa/*` and `data/airbench/gepa_runs/*` and see exactly what happened.
- **The multi-agent idea was not purely speculative.** We actually implemented and exercised a coordinator/performance/reliability/integrator/reviewer stack rather than just proposing it abstractly.
- **Failure analysis is stronger than before.** We now know the recurring failure classes with specificity: dtype mismatch in compiled half precision, CUDAGraph overwrite patterns, CLI contract regressions, and quota-driven collapse in the proposal layer.

### Areas for Improvement
- **We optimized the harness more than the solver.** Much of the week’s effort went into making the system run, log, and recover cleanly. That was necessary, but it also means the optimization algorithm itself was not subjected to enough clean iterations to justify strong conclusions about its effectiveness.
- **The proposal mechanism is still too unconstrained for the reliability we need.** Even with a reviewer and a refiner loop, the system still emits candidates that break required CLI flags, violate dtype discipline, or introduce CUDAGraph-unsafe evaluation logic. That suggests our proposal-side safety checks are not strong enough.
- **Search-time evaluation is too noisy near the hard threshold.** With `trials=1`, a candidate that is effectively the same quality can be labeled “valid” or “invalid” depending on single-run variance. That is a poor basis for ranking when the seed is already near 94%.
- **The multi-agent stack is too expensive for current quotas.** The deeper run showed that an expressive proposer is only useful if we can sustain it. Right now, later sessions fell into `RateLimitError`, which means the architecture is not yet matched to the available budget.
- **We still lack a clean comparison against a simpler baseline.** We have not yet shown that the multi-agent proposer is better than a single-agent proposer under matched cost, nor that the refiner loop improves accepted-candidate quality under matched evaluation budget.

### Critical Risks and Assumptions
We are currently assuming that “more agentic structure” is directionally helpful, but we do not yet have evidence that it improves search efficiency under realistic API constraints. If the team proposer consumes too much budget and fails before it can generate a meaningful number of strong candidates, it may be strictly worse than a simpler reflection loop.

We are also assuming that using the strong AirBench seed as a starting point is acceptable for this stage of the project. That is reasonable as an engineering benchmark, but it weakens any broader claim about program discovery “from scratch.” If we are not explicit about seeded vs. seedless experiments, the scientific framing becomes muddy.

Finally, we are assuming that single-trial search-time evaluation is good enough for iterative optimization. The current runs suggest it is not. If the incumbent is near the threshold, one-trial noise can dominate the search trajectory, and then the optimizer is partly learning the noise distribution rather than the true program quality.

## DECIDE: Concrete Next Actions (max 3; within a week)
- **Stabilize the evaluator before running deeper searches.** Change search-time evaluation from `trials=1` to at least `trials=2`, or otherwise implement a softer thresholding/ranking rule near 94%. This is the highest-value methodological fix because it directly affects whether improvements are real or just variance.
- **Reduce proposal cost and enforce harder pre-eval safety gates.** Either downgrade some internal agent roles to a cheaper model or simplify the team call graph, and add explicit local rejection rules for missing CLI flags, known dtype-mismatch patterns, and unsafe compiled TTA/CUDAGraph logic. Right now too many candidates fail for avoidable reasons, and too much quota is spent getting there.
- **Run one controlled ablation on the proposer itself.** Compare the current team proposer against a simpler proposer under the same evaluation budget and hardware policy. Without this, we cannot justify the additional architectural complexity of the multi-agent system.

## ACT: Resource Needs
We need enough OpenAI quota to sustain a full multi-agent run without collapsing into rate limits halfway through; otherwise the current architecture cannot be evaluated fairly. We also need a slightly more stable evaluation budget on Modal, specifically enough room to score candidates with at least two measured trials during search rather than one. Finally, we need one short iteration cycle dedicated to proposal-side safety rather than new algorithmic ideas: hard local linting for CLI contract compliance, dtype discipline, and compiled-eval safety would likely save more time and money than simply increasing the outer loop budget again.
