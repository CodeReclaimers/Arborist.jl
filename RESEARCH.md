# Research Project — Arborist Correctness Deep Dive

<!--
  SETUP: Replace bracketed sections with your project specifics.
  This file is read by Claude Code at the start of each adversarial loop iteration.
  It tells the orchestrator what the project is, what matters, and what's off-limits.
-->

## What This Project Is

We are investigating the correctness of all genetic programming algorithms currently implemented in Arborist.jl.  A useful result is determining the following for each algorithm: Does the algorithm match the implementation generally expected by genetic programming practitioners?  (And if not, describe how it deviates.)  Is the algorithm implemented to the engineering standard one might expect for a production-ready system?  (If not, provide a ranked list of software defects or poor design choices.)  Are the references to the algorithm in arborist_paper_observations.md correct?  Does the description in arborist_paper_observations.md match the code as currently implemented?  Do the comments and docstrings match the code?

## Source Material

<!-- Point to the files, codebase, papers, or data the research is grounded in. -->

- src/
- test/
- arborist_paper_observations.md

## What's Known (Starting State)

<!--
  Summarize what's already established. This prevents the loop from
  rediscovering things you already know. Update this section as the
  loop produces new findings.
-->

- All 1579 tests pass (fast tier ~52s, full benchmarks ~83s)
- A code audit already found and fixed 5 defects with 290 new test assertions (commit 8d1f3d4)
- Known limitation: ExprGenome `@eval` adds methods to Julia's method table on every evaluation; long runs accumulate thousands of methods
- Known limitation: AntGenome not thread-safe — uses module-level `Ref` for simulator state
- Known limitation: TreeGenome serialize/deserialize format mismatch — `serialize` outputs infix, `deserialize` parses prefix; binary ops do not round-trip
- Known limitation: GraphGenome.deserialize not implemented — returns `nothing` with a warning; LLM mutation of GraphGenome is non-functional
- Known limitation: Distributed NEAT innovation ID collisions across workers — conflicting node IDs from separate workers
- Known limitation: ExprGenome serialize round-trip is partial (~80% success rate due to `Float32(literal)` forms)
- Fitness sharing sign error (division instead of multiplication) was already found and fixed
- The ExprGenome `deserialize` parser was fixed (2026-03-25) to accept control flow

## Research Questions

<!-- What specifically are you trying to learn? Rank by priority. -->

The project implements 5 distinct algorithm areas. For each, the questions below apply:

**Algorithm areas:**
- ExprGenome GP (Expr-tree code generation, `@eval`-based evaluation)
- TreeGenome GP (DynamicExpressions.jl-based symbolic regression)
- GraphGenome / NEAT (neural topology evolution)
- Island model (sequential, synchronous distributed, asynchronous distributed)
- Operators and selection (mutation variants, crossover, tournament selection, speciation, fitness sharing)

**Questions for each area:**

1. Does the implementation match what a GP practitioner would expect from the standard algorithm? (If not, describe how it deviates.)
2. Is the implementation at production-ready engineering quality? (If not, provide a ranked list of software defects or poor design choices.)
3. Are the references to the algorithm in arborist_paper_observations.md correct?
4. Does the description in arborist_paper_observations.md match the code as currently implemented?
5. Do the comments and docstrings match the code?

## Out of Scope

<!--
  CRITICAL: This section prevents the loop from wandering into territory
  where automated research can't produce useful answers. Be explicit.
-->

**Do NOT investigate autonomously:**
- Fixes for any discovered bugs or discrepancies.
- Performance optimization — this audit is about correctness, not speed.
- The examples/ directory (bin packing, sorting, etc.) — these are application-level, not core algorithm correctness.
- Documentation style or organization.

**Do NOT change without human approval:**
- Any files already in the project.

## Investigation Style

<!--
  What does "run an experiment" mean for this project? Pick the modes
  that apply and delete the rest. The orchestrator uses this to decide
  how to test Gemini's findings.
-->

Applicable investigation modes:
- **Code experiment**: Write and run a script that exercises a claimed behavior with pass/fail criteria
- **Formal analysis**: Check whether algorithm logic matches reference descriptions
- **Counterexample search**: Construct inputs that expose deviations from expected behavior
- **Literature search**: Verify that references in arborist_paper_observations.md are accurate

## Project Values

<!--
  These encode YOUR judgment. The orchestrator follows them when making
  decisions autonomously. Delete any that don't apply, add your own.
-->

- **Empirical honesty over elegance.** If an investigation contradicts current assumptions, the assumptions change. Record inconvenient findings.
- **Boring specificity over compelling vagueness.** Concrete numbers and reproducible results beat theoretical arguments. If you can't test it, flag it as speculative.
- **Integration over polish.** Prefer investigating how components interact over refining single components further.

## Stopping Criteria

<!--
  When should the loop pause for human review? Defaults below work well;
  adjust if needed.
-->

- After every **3 iterations**, stop and write a summary for human review
- If fewer than **2 actionable findings** in the last iteration
- If **3 consecutive iterations** produced only minor refinements and no structural findings
- If any finding challenges a **core assumption** listed in "Out of Scope"
- If total wall-clock time exceeds **2 hours**
- When in doubt, **stop and ask**
