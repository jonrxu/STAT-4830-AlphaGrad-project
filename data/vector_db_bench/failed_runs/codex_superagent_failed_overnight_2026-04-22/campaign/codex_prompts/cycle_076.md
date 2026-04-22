        You are running a persistent Codex superagent campaign for vector-db-bench.

        Objective:
        - reach a valid 22000.00+ QPS solution and beat the public target
        - keep the external benchmark spec official
        - improve both the Rust solution and your own local harness inside this workspace

        Hard constraints:
        - keep the benchmark target official
        - preserve the official dataset, recall threshold of 0.95, and CPU pinning policy
        - do not modify the benchmark repo or data outside this workspace
        - the driver evaluates after this cycle and promotes the best valid state

        Operating mode:
        - this workspace persists across cycles
        - there is no artificial tool-call budget for this cycle
        - use web research when useful
        - you may directly edit the Rust solution
        - you may directly edit your local harness under `.meta_codex/`
        - you may create helper scripts under `.meta_codex/tools/` and use them in later cycles
        - optimize for speed of convergence, not for tiny cosmetic changes

Official-style workspace tools available now:
- `.meta_codex/tools/build_project`
- `.meta_codex/tools/run_correctness_test`
- `.meta_codex/tools/run_benchmark`
- `.meta_codex/tools/run_profiling`
- `.meta_codex/tools/get_status`
- use these first-class wrappers when you want benchmark-style actions during the cycle
- there is no `finish` tool here because the driver ends and evaluates the cycle externally


        Read first:
        - .meta_codex/README.md
        - .meta_codex/official_constraints.md
        - .meta_codex/strategy.md
        - .meta_codex/design_spec.md
        - .meta_codex/research_notes.md
        - .meta_codex/benchmark_policy.md
        - .meta_codex/incumbent_record.md
        - .meta_codex/milestones.md
        - .meta_codex/campaign_journal.md
        - .meta_codex/progress_state.json
        - any recent cycle summaries under `.meta_codex/recent_cycles/`

        Workspace guidance:
        - if the current workspace is badly broken, restore from `.meta_codex/mainline_snapshot/` or repair directly
        - keep `.meta_codex/campaign_journal.md` current with major decisions
        - prefer architecture changes that move toward real ANN / IVF-style shortlist generation
        - avoid fake ANN designs that still depend on global full-scan behavior
        - create local scripts when they will make future cycles faster

        Cycle:
        - this is cycle 76
        - work directly in this workspace
        - end with a concise summary of what changed, current score expectations, and the next best move
