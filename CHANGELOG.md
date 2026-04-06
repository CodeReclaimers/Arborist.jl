# Changelog

All notable changes to Arborist.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-04-06

Initial release.

### Added

#### Core framework
- Problem/Algorithm/Solve API following SciML conventions
- `GeneticProgramming` algorithm with tournament selection, elitism,
  subtree crossover, subtree/point/hoist/expansion mutation
- `GPProblem`, `GPResult` with per-generation best/mean fitness history
- Explicit RNG threading for reproducible runs (`parallel=false`)
- Parallel population evaluation via `Threads.@threads`
- `convergence_threshold` parameter for early termination on minimization
  problems

#### Genome types
- `ExprGenome` — typed AST-based genome with `@eval` compilation; supports
  loops, conditionals, mutable state, side-effectful primitives
- `TreeGenome{T}` — DynamicExpressions.jl-backed expression-tree genome
  for fast vectorized symbolic regression (8.4× faster than `ExprGenome`
  on the Koza suite); now part of the core module rather than a package
  extension
- `GraphGenome` — NEAT-style neural topology genome with innovation
  numbers, structural mutation (add node, add connection),
  innovation-number-aligned crossover, and compatibility distance
- `AntGenome` — side-effectful imperative genome for agent control
  problems (Santa Fe ant trail and similar)

#### Multi-objective GP
- `NSGAII` algorithm — non-dominated sorting + crowding distance with
  (μ+λ) survivor selection
- `AbstractMultiObjectiveEvaluator` interface (`evaluate_multi`,
  `objective_names`)
- `ParsimonyEvaluator{E}` — wraps any single-objective `AbstractEvaluator`
  into a two-objective `(fitness, complexity)` problem; replaces
  `bloat_penalty` with a real Pareto front
- `NSGAIIResult` with the Pareto front, full final population, generation
  count, wall time, objective names, and per-generation 2D hypervolume
- Example: `examples/nsga2_regression.jl`

#### Speciation
- `ThresholdSpeciation` — NEAT-style speciation by genomic distance with
  stagnation culling and configurable fitness sharing
- `BehavioralSpeciation` — speciation by behavioral fingerprint (probe-
  based), groups syntactically distinct programs that make the same
  decisions; empirically the most reliable variant on the bin packing
  benchmark (lowest cross-seed variance)
- Configurable `sharing_formula`: `:none`, `:linear`, `:sqrt`, `:log2`
  (adapted for minimization: `shared = raw × penalty(species_size)`)
- `apply_sharing(raw, size, formula)` helper for custom speciation
  implementations

#### Island model
- `IslandModel` algorithm with three execution backends:
  sequential (single-process), synchronous distributed, and asynchronous
  distributed (`Distributed.jl` workers)
- `RingTopology`, `CompleteTopology`, `RandomTopology` for migration
  routing
- `MigrantGenome` + `to_migrant`/`from_migrant` for cross-process
  serialization

#### LLM operator
- `LLMMutationOperator` — FunSearch/AlphaEvolve-style LLM-as-mutation
  operator that serializes an `ExprGenome` to source, prompts an LLM
  for a meaningfully modified variant, and parses + type-checks the
  response. Silent fallback to a classical operator on any failure
  (API error, timeout, parse failure, type-check failure)
- Backends: Anthropic, OpenAI, and local Ollama (any OpenAI-compatible
  endpoint). Uses `Downloads.jl` (Julia stdlib) — no third-party HTTP
  dependency
- `_http_post` `Ref{Function}` hook for mocking in tests

#### Security
- `ASTSanitizer` — opt-in function call whitelist for `@eval`-based
  genomes; defense-in-depth against code injection in evolved programs
- Default whitelist (`DEFAULT_SAFE_CALLS`) covers all standard math and
  control-flow operators; users can extend it for domain primitives

#### Evaluators
- `TableFitnessEvaluator` — input/output table evaluation for
  `ExprGenome` with per-row time limits
- `TreeFitnessEvaluator` — vectorized data-matrix evaluation for
  `TreeGenome`
- `GraphEvaluator` — forward-propagation evaluation for `GraphGenome`
- `AntEvaluator` — Santa Fe ant trail simulator
- `SymbolicRegressionEvaluator` — convenience wrapper that builds a
  `TreeFitnessEvaluator` from a target function and a sampling domain

#### Examples and benchmarks
- `examples/feynman_regression.jl` — Feynman symbolic regression
  benchmark (Udrescu & Tegmark)
- `examples/lorenz_recovery.jl` — Lorenz attractor recovery benchmark
- `examples/sorting.jl` + `examples/sorting_distributed.jl` — sorting
  algorithm evolution with curriculum learning, comparison-count
  penalty, and seed template ablation
- `examples/bin_packing.jl` — bin packing with classical / LLM /
  behavioral-speciation modes; results synthesized in
  `examples/bin_packing_overnight_results.md` and
  `examples/bin_packing_combined_results.md`
- `examples/nsga2_regression.jl` — multi-objective symbolic regression
- Test benchmarks: Koza-1/2/3, 4-bit boolean parity, XOR (NEAT), Max
  Ones, x² symbolic regression, Santa Fe ant trail (smoke test)

#### Operators
- `SubtreeMutation`, `PointMutation`, `HoistMutation`, `ExpansionMutation`
- `SubtreeCrossover`
- `TournamentSelection`

#### Tooling
- Tiered test execution (`ARBORIST_RUN_BENCHMARKS=true` /
  `GENPROG_RUN_BENCHMARKS=true` for the slower benchmark tier)
- GitHub Actions CI on Ubuntu and Windows for Julia 1.10, 1.11, and
  nightly
- Documenter.jl-based documentation site

### Changed
- **`TreeGenome` is now part of the core module** rather than a package
  extension. DynamicExpressions.jl is a hard dep of Arborist; users no
  longer need to call `Base.get_extension(...)` to access `TreeGenome`,
  `TreeFitnessEvaluator`, or `SymbolicRegressionEvaluator`. Same for
  `LLMMutationOperator`, which was never an extension despite the
  prior documentation claiming otherwise.
- **`LLMMutationOperator` switched from HTTP.jl to `Downloads.jl`**
  (Julia stdlib). Eliminates a third-party HTTP dependency and matches
  the convention used by other parts of the framework.
- README and `docs/src/*.md` rewritten to document the actual exported
  API. Previous code blocks referenced an extension-based API
  (`Base.get_extension(Arborist, :DynExprExt)`, `:LLMOperatorExt`,
  `using HTTP`) that did not exist in the codebase; copy-pasting them
  produced `MethodError` or `nothing` from `get_extension`.
- `_breed_next_generation!` extracted to deduplicate the breeding loop
  shared by `GeneticProgramming`, `IslandModel`, and `NSGAII`.
- `Project.toml`: removed duplicate `DynamicExpressions` entry from
  `[extras]` (already in `[deps]`).
- `Manifest.toml` removed from version control (libraries should not
  pin downstream environments).

### Fixed
- **GraphGenome non-deterministic Dict iteration** (6 sites). All
  iteration over node/connection Dicts now goes through sorted helpers
  for reproducibility.
- **GraphGenome crossover** now produces two distinct children instead
  of two copies of the same offspring.
- **NEAT compatibility distance**: disjoint and excess genes are now
  counted separately, matching the original NEAT formulation.
- **Async island deadlock**: `put!` on a full channel could deadlock the
  worker pool; replaced with non-blocking offer + retry.
- **ExprGenome migrant naturalization**: migrants from other islands
  are now re-typed against the receiving island's `GenState` so that
  cross-island crossover does not produce type-inconsistent ASTs.
- **`FunctionDetails.isequal`**: was reflexively true for unrelated
  primitives with the same name; now compares the full type signature.
- **`crossover_rate + mutation_rate > 1.0`** is now caught at
  `GeneticProgramming` construction with a clear `ArgumentError`
  rather than silently producing invalid offspring distributions.
- **LLM operator JSON escape/unescape**: control characters and
  backslash escapes are now round-trip safe; previously a backslash-n
  in source would deserialize as a literal newline.
- **LLM operator robustness**: `deserialize` and `sanitize` are wrapped
  in `try`/`catch`; any exception falls back silently to the configured
  classical operator instead of aborting the generation.
- **Five defects found by code audit** (`8d1f3d4`) plus 290 additional
  test assertions covering the gaps that allowed them through.
- **Eight findings from a 3-iteration adversarial research loop**
  covering correctness, reproducibility, and edge cases in the solve
  loop, distance metrics, and migration paths.
- **`docs/src/quickstart.md` XOR example**: passed `Vector` to
  `GraphEvaluator` (which requires `Matrix{Float64}`) and was missing
  `reset_innovation_counter!()`.
- **Original README LLM example**: used `TreeGenome{Float32}` with
  `LLMMutationOperator`, but the operator only dispatches on
  `ExprGenome`. Anyone copy-pasting the example would have hit a
  `MethodError`.

### Removed
- Dead `max_depth` field from `GeneticProgramming` (was never read).
- Adversarial-loop scratch artifacts (after all 8 findings were
  addressed and committed).
