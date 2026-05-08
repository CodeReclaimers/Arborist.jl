# Changelog

All notable changes to Arborist.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.2] — 2026-05-08

### Documentation

- Add Zenodo DOI (10.5281/zenodo.20076827) to `CITATION.cff`, the
  README bibtex block, and a Zenodo badge in the README header.

## [0.1.1] — 2026-05-07

### Security

- **`ASTSanitizer` whitelist bypass via indirect calls.** Strengthen the
  sanitizer to require that the callable in any `:call` expression is a
  plain `Symbol` present in `allowed_calls`. Previously, indirect calls
  built from arrays, tuples, or refs (e.g. `[open][1]()`,
  `(run,)[1]()`) could route around the whitelist and reach unsafe
  functions through `@eval`. The fix only affects the `ExprGenome`
  `@eval` path with sanitization enabled; `TreeGenome` and `GraphGenome`
  do not use `@eval` and were not affected.

### Fixed

- Track the true all-time best genome from the evaluated initial population
  across `ExprGenome`, `TreeGenome`, and `GraphGenome` solve paths, including
  runs where later generations regress or `elitism=0`.
- Apply `bloat_penalty` consistently in `GraphGenome` solves.
- Validate public evolutionary algorithm configuration earlier, including
  population size, generations, rates, elitism, bloat penalty, island counts,
  and migration settings; operator compatibility checks now respect zero
  mutation or crossover rates.
- Seed `HallOfFame` archives from the initial population and preserve them
  through checkpoint/resume.

## [0.1.0] — 2026-04-24

Initial public release. The scope expanded between the original
pre-registration readiness cut (2026-04-06) and the final registration
date as the framework continued to land research-grade capabilities —
notably six classical GP benchmark families (Phase E), ten neuroevolution
benchmarks (Phases A–D), and nine infrastructure additions (Phase F.0
through F.8). All are folded into this single 0.1.0 entry below.

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
- `FitnessProportionateSelection` — classical roulette-wheel selection
  adapted to the lower-is-better convention (weight `1 / (ε + f_i − f_min)`)
- `RankSelection(; selection_pressure=1.5)` — linear-ranking selection
  with `selection_pressure ∈ [1.0, 2.0]`
- `TruncationSelection(; ratio=0.5)` — keep the top `ratio` fraction by
  fitness, then sample uniformly

#### Tooling
- Tiered test execution (`ARBORIST_RUN_BENCHMARKS=true` /
  `GENPROG_RUN_BENCHMARKS=true` for the slower benchmark tier)
- GitHub Actions CI on Ubuntu, Windows, and macOS for Julia 1.10,
  1.11, and nightly
- Documenter.jl-based documentation site

#### Research-grade infrastructure (Phase F)
- **F.0 Per-case fitness API + structured RunLog.** `evaluate_cases(g, e)`
  generic with opt-in per evaluator (raises `MethodError` when
  unimplemented). `RunLog` + `GenerationLog` + `record!` capturing
  per-generation best / mean / median / worst fitness, species count
  and sizes, unique-structure hash diversity, per-operator attempt
  and success tallies, and wall time. `SpeciationSnapshot` carrier
  threads through `_apply_speciation!` so the solve path can log
  without recomputing. `log::Union{Nothing, RunLog}` kwarg on every
  single-objective `solve`.
- **F.1 Lexicase + epsilon-lexicase selection.** `LexicaseSelection`
  and `EpsilonLexicaseSelection(; epsilon=0.0)`. Polymorphic
  `select_parent(strategy, sel_fits, case_fits, rng)` dispatch
  replaces the hardcoded `_tournament_select`. `needs_cases(selection)`
  trait — when `true`, solve materializes the per-case matrix via
  `evaluate_cases`. Auto-epsilon uses MAD per case (La Cava et al. 2016).
- **F.2 GraphGenome deserialize + LLM-on-NEAT.**
  `deserialize(::Type{GraphGenome}, s, n_inputs, n_outputs; reassign_innovations=false)`
  is implemented (was a `nothing`-returning stub). Tolerates LLM preambles
  and code fences. `reassign_innovations=true` issues fresh innovation
  IDs via `_next_innovation!()` so LLM-generated IDs cannot collide
  with the parent pool. `mutate(::LLMMutationOperator, ::GraphGenome, rng)`
  dispatches for LLM-driven NEAT mutation.
- **F.3 Constant optimization for TreeGenome SR.** Periodic BFGS pass
  on top-K individuals every N generations. Central finite-difference
  gradients; dense BFGS with Armijo line search and a rollback guard
  so fitness cannot regress. `ConstantOptimization(; frequency, top_k,
  max_iter, tol, fd_step)` field on `GeneticProgramming`.
  `optimize_constants!(g, e; kwargs)` entry point for manual use.
- **F.4 Novelty Search + MAP-Elites.** `NoveltyArchive{B}`
  thread-safe bounded-size store, `NoveltySearchEvaluator{F,D,B}` wrapping
  a fingerprint function and distance metric (returns
  `-mean_knn_distance`). `MAPElites <: AbstractEvolutionaryAlgorithm`
  with grid-based archive and its own solve loop sampling parents from
  the filled archive; `MAPElitesArchive{G}`, `MAPElitesResult{G}`,
  `coverage(archive)`, `qd_score(archive)`.
- **F.5 Checkpoint / resume + operator-success tracking.**
  `Checkpoint{G}` + `save_checkpoint(ckpt, path)` (atomic via
  tmp+rename) + `load_checkpoint(path)` (format-version + Julia-version
  stamp, rejects mismatches). `solve(...; checkpoint_every,
  checkpoint_path, resume_from, allow_signature_mismatch)` on
  `ExprGenome` and `TreeGenome` paths. All-time-best tracking added so
  the best genome survives elitism loss. Per-operator attempt / success
  tallies populated via a new `operator_name(op) -> Symbol` generic.
- **F.6 Plots.jl recipes via package extension.**
  `ext/ArboristRecipesBaseExt.jl` defines `@recipe` rules for
  `GPResult` (fitness trajectory), `NSGAIIResult` (Pareto front, 2D
  and 3D), `RunLog` (3-subplot fitness / species / structures), and
  `MAPElitesResult` (coverage + QD histories). `@userplot`s
  `plothypervolumetrajectory(nsga2_result)` and
  `plotarchive(map_elites_archive)`. Weak dep on `RecipesBase`; zero
  runtime cost without Plots.
- **F.7 Automatically Defined Functions (Koza-style ADFs).**
  `ADFGenome{T}` with a main tree plus N Koza-style ADF subroutines at
  fixed binary arity. Macro-expansion via `expand_adfs(g) -> Node{T}`:
  walks the main tree, replacing ADF-slot ops with copies of the ADF
  body and substituting ARG references. Full `AbstractGenome` interface
  (mutation, crossover, distance, complexity, serialize).
- **F.8 CMA-ES.** Full (μ_w/μ, λ)-CMA-ES with rank-1 and rank-μ
  covariance updates, evolution-path step-size control, `hsig`
  Heaviside step. `CMAES <: AbstractEvolutionaryAlgorithm` with
  auto-λ (Hansen 2016 default). `flatten_weights(g)` /
  `unflatten_weights!(g, w)` interface — opt-in per genome;
  `GraphGenome` adaptor sorts by innovation number for determinism.
  `LinearAlgebra` (stdlib) added as a direct dependency.

#### Neuroevolution benchmarks (Phases A–D)
- **GraphGenome framework integration.** `NEATDefaultMutation` +
  five individual mutation operators (`WeightPerturbMutation`,
  `WeightReplaceMutation`, `AddConnectionMutation`, `AddNodeMutation`,
  `ToggleConnectionMutation`) and `NEATCrossover`. Works with NSGA-II
  (Pareto of fitness vs enabled-connection count), sequential
  `IslandModel`, and distributed `IslandModel` (per-worker disjoint
  innovation ID ranges via `init_innovation_range!`). `GraphEvaluator`
  supports recurrent networks via `allow_recurrent=true` with
  `relaxation_passes=N`; samples are treated as a time sequence with
  persistent node state. `neat_defaults()` returns
  `(mutation_ops, crossover_ops)` for drop-in use.
- **`EpisodicEvaluator`.** Closed-loop control-task evaluator for
  `GraphGenome`. Declarative API: the user provides `(initial_state,
  dynamics, reward, done, observe, decode_action)` callables; the RNG
  is threaded through `initial_state(rng)` for reproducibility;
  `n_episodes` rollouts are averaged per fitness evaluation. Returns
  `-mean_reward` (lower-is-better).
- **Phase A: classification benchmarks.** 3-bit / 5-bit parity,
  two-spirals (plus an NSGA-II variant).
- **Phase B: control-task benchmarks.** Single-pole cart-pole,
  double-pole (Markovian), mountain car — all with `GraphGenome` +
  `EpisodicEvaluator`. Benchstone-registered.
- **Phase C: non-Markovian stress benchmarks.** Variable-delay
  sequence recall (test-only).
- **Phase D: modularity, time series, multi-objective showcases.**
  Retina left-and-right modularity benchmark, Mackey-Glass τ=17
  time-series prediction, NSGA-II retina with (error, modularity_proxy)
  objectives.

#### Classical GP benchmarks (Phase E)
- **Nguyen-1..10** (`nguyen_regression.jl`): canonical modern SR suite
  per McDermott et al. 2012. `TreeGenome` + protected pdiv / plog /
  psqrt. 3/5 seeds gate at fitness < 0.01 (Nguyen-7 relaxed to 2/5).
- **Keijzer-4, Keijzer-11** (`keijzer_extrapolation.jl`): extrapolation
  benchmarks. K-4 reports train fitness plus interior and out-of-range
  test RMSE as diagnostics; K-11 uses 20 sparse training points with
  a 20×20 test grid.
- **6-bit / 11-bit multiplexer** (`multiplexer.jl`): Koza's Boolean
  scaler using Boolean-complete {AND, OR, NAND, NOR, XOR, NOT}.
- **UCI Iris** (`iris_classification.jl`): Fisher's 3-class dataset,
  150 rows embedded inline (no network dependency). One-vs-rest with
  three `TreeGenome` evolutions and argmax at classification. Gate
  4/5 seeds ≥ 90% test accuracy on a fixed 80/20 split.
- **Acrobot swing-up** (`acrobot_neat.jl`): Sutton 1996 two-link
  under-actuated pendulum, RK4 integration at dt=0.2.

#### Bin packing research enhancements
- **`behavioral_initialize`:** MAP-Elites-style diverse population
  seeding. Generates a `pool_size` random-program pool, evaluates,
  fingerprints, greedy-bins by fingerprint distance, and returns the
  best-fitness representative from each distinct bin.
- **NSGA-II bin packing (canonical):** `BPMultiObjectiveEvaluator` with
  two objectives `(fitness, failed_placements)`. The 2026-04-14
  ablation study found that a third `success_rate` objective is
  redundant given `failed_placements`; the historical 3-objective
  formulation is reproducible via the `three_objective` ablation.
- **Configurable prompt enrichment** (`MutationContext`,
  `AbstractPromptSection`, `FitnessSection`, `ElitesSection`,
  `GenerationSection`, `render_enrichment`). Populated once per
  generation by the solve loop; sections render into the LLM user
  message.
- **`LLMCallStats`:** per-operator accumulator tracking call counts,
  API successes / failures / fallback skips, input / output tokens
  (from API usage fields), input / output character counts, and
  cumulative latency.
- **`debug_log::Union{IO, Nothing}`** field on `LLMMutationOperator`:
  when set, writes the full user message, raw LLM response, and
  outcome for each call.
- **"Container" rename.** `bin` renamed to `container` across the
  bin-packing example and extension primitives to improve clarity
  for the generalized heuristic-discovery framing.

#### TreeGenome improvements
- **`IslandModel` support for `TreeGenome`.** Migration transports
  `Node{T}` directly across workers so op indices remain valid against
  every island's shared `OperatorEnum`.
- **`serialize` / `deserialize` round-trip** fixed via a `Meta.parse`
  walker instead of string-based substitution. Prefix-notation and
  unary-op round-trips now succeed.

#### Run-management additions
- **All-time-best Hall of Fame.** New `HallOfFame{G}` archive type,
  opt-in via `solve(...; hall_of_fame_size=K)`. Returns a top-K
  best-first archive of distinct fitnesses observed across all
  generations (de-duplicated within `1e-12` to avoid storing
  structurally equivalent solutions). The archive survives elitism
  loss and is exposed as `result.hall_of_fame`. `K=0` (default)
  disables the archive and leaves `result.hall_of_fame === nothing`;
  the older 8-arg `GPResult` constructor is preserved for backward
  compatibility.
- **Warm-starting `solve` from a seed pool.** New `initial_population`
  kwarg on the single-objective `GeneticProgramming` and on the
  sequential `IslandModel` solve paths. Element type and length must
  match `algorithm.pop_size` (or `n_islands * pop_size` for
  `IslandModel`). Mutually exclusive with `resume_from`.
- **Ephemeral Random Constants (Koza-style).** `GeneticProgramming` now
  carries an optional `constant_sampler::Union{Function, Nothing}` field
  threaded into `TreeGenome` random-tree creation and point-mutation
  leaf creation via task-local storage. `erc_uniform(lo, hi)` is the
  canonical helper, returning a `(rng) -> T` callable that samples
  uniformly. Legacy default (`T(randn(rng))`) is preserved when the
  field is `nothing`.

#### Run utilities
- **`Arborist.Benchmarks` submodule.** Canonical GP and neuroevolution
  benchmark problems as reusable data generators consumed by user
  code: symbolic regression (`nguyen(n)`, `keijzer(variant)`,
  `koza(name)`, `pagie()`), classification (`iris()`,
  `two_spirals()`), Boolean (`multiplexer(address_bits)`,
  `parity(n_bits)`), control (`cartpole()`, `mountain_car()`,
  `acrobot()`, `double_pole(; markovian)`), sequence/memory
  (`sequence_memory(length)`, `sequence_recall(; delay)`), and
  classic neuroevolution (`xor_env()`). Each generator returns a
  `NamedTuple` carrying dataset (or environment callables), shape
  metadata, target descriptions, and sensible success thresholds —
  decoupled from evaluator choice.
- **Cross-cutting helpers.** `train_test_split(X, y; test_size, rng,
  stratify)` for deterministic train/test partition with optional
  class-stratified sampling; `summarize(xs)` returning
  `(; mean, std, median, min, max, q25, q75, n)` (non-finite entries
  excluded so a single `Inf` does not poison the report); and
  `run_multi_seed(f, seeds; parallel=false)` for multi-seed runs
  with optional `Threads.@threads` parallelism.

#### Visualization, inspection, and bloat control
- **Graphviz DOT export.** `to_dot(g)` and `to_dot(io, g)` produce a
  self-contained DOT document for every genome type — `TreeGenome`,
  `ExprGenome`, `ADFGenome`, `AntGenome`, and `GraphGenome`. The `dot`
  binary is *not* a runtime dependency; `to_dot` only emits the source
  document. Tree-shaped genomes render as a top-down DAG; `GraphGenome`
  renders left-to-right with role-distinguished node shapes and
  weight-labeled edges (disabled connections drawn dashed).
- **`Base.show` methods** for `GPResult`, `NSGAIIResult`,
  `MAPElitesResult`, `RunLog`, `Checkpoint`, `HallOfFame`, and every
  concrete genome — so the REPL no longer dumps internal field structure
  for the most common return types.
- **`tree_depth(g) -> Int`** generic exported alongside `complexity(g)`.
  Implemented for `TreeGenome`, `ExprGenome`, `AntGenome`, and
  `ADFGenome`; reports the longest root-to-leaf path.
- **Per-operator `max_depth` / `max_size` caps.** `SubtreeMutation`,
  `PointMutation`, `HoistMutation`, `ExpansionMutation`, and
  `SubtreeCrossover` now accept `max_depth` and `max_size` keyword
  arguments. When set, the operator returns the parent unchanged
  rather than producing an offspring that exceeds the caps. Both
  default to `nothing` (no cap). These supersede the dead struct-level
  `GeneticProgramming.max_depth` field that was removed during cleanup
  (see *Removed* below) — the new caps are per-operator, so different
  operators in the same `mutation_ops` vector can carry different
  limits.
- **`default_protected_function_set()`** plus the corresponding Koza
  protected primitives `pdiv`, `plog`, `psqrt`, `pexp`, `pinv` (and the
  shared `PROTECTED_EPS` constant). The set is the standard SR
  starting point and is what the Phase E benchmarks (Nguyen, Keijzer)
  evolve against.
- **Extended `ACTIVATION_FNS`** with the canonical CPPN / HyperNEAT
  activation set — `gauss`, `sin`, `abs`, `step` — joining the existing
  `sigmoid`, `tanh`, `relu`, `identity`. `AddNodeMutation` /
  `NEATDefaultMutation` still default `hidden_activations` to
  `{sigmoid, tanh, relu}`, so the new activations are opt-in via the
  keyword argument.

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
- **`examples/` reorganized:** research apparatus (tuning harness,
  ablation drivers, shell orchestration, 11 files) moved under
  `examples/research/`. Tutorial-level scripts remain at the top of
  `examples/`. Two `README.md` index files added.
- **Session notes migrated to refstore.** Per-session progress logs
  no longer live in the repo; new updates go to the refstore project
  `genprog_jl` via `mcp__refstore__write_session_note`. See
  `CLAUDE.md`'s Session Notes section.

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
  `LLMMutationOperator`, but the operator did not (and still does
  not) dispatch on `TreeGenome`. Anyone copy-pasting the example
  would have hit a `MethodError`. The corrected example uses
  `ExprGenome`.
- **LLM operator JSON extraction**: `\uXXXX` unicode escapes in the
  response text are now decoded correctly before being handed to the
  deserializer.
- **LLM operator literal type coercion**: numeric literals in
  deserialized output are now coerced to the parent genome's expected
  type before the type-check pass, preventing spurious rejection of
  LLM output that returned `1.0` where the parent had `1.0f0`.
- **Windows CI flake**: LLM latency measurement now uses `time_ns()`
  instead of `time()` to avoid millisecond-granularity round-trip
  glitches on Windows runners.
- **`ASTSanitizer` false rejection**: domain-specific function calls
  from LLM output (bin-packing extension primitives) are no longer
  rejected as unsafe when the caller has registered them in the
  allow-list.
- **Docs-build size limit**: the single-page `api.md` was a 220 KiB
  `@autodocs` block over Documenter's 200 KiB threshold, failing
  every docs-CI run. Split into an overview + four per-topic sub-pages
  under `docs/src/api/`.

### Removed
- Dead `max_depth` field from `GeneticProgramming` (was never read).
  The replacement is per-operator `max_depth` / `max_size` keyword
  arguments on `SubtreeMutation`, `PointMutation`, `HoistMutation`,
  `ExpansionMutation`, and `SubtreeCrossover` (see
  *Visualization, inspection, and bloat control* above).
- Adversarial-loop scratch artifacts (after all 8 findings were
  addressed and committed).
- **Legacy imperative API:** `Individual`, `Population`,
  `evaluate_individual!`, `evolve!`, and the supporting
  `tournament_select` / `mutate_individual` / `crossover_individuals`
  helpers are removed. Callers should use the
  Problem/Algorithm/Solve path: `solve(::GPProblem, ::GeneticProgramming)`.
  The corresponding tests in `test/unit/test_coverage_gaps.jl` are
  removed — the replacement API is covered by the other unit tests.
- Pre-migration session progress logs (`progress-YYYYMMDD.md`, 20
  files) from the repo tree. Project migrated to refstore session
  notes on 2026-04-22. Logs remain recoverable via git history.
- Internal planning and paper-draft artifacts (`archive/`,
  `arborist_paper_observations.md`, `plan_treegenome_parser_fix.md`).
- Per-run bin packing ablation result files (50 tracked `.md` files,
  mostly `bin_packing_results_scaling_*_seed*.md` and
  `_unseeded_*_seed*.md`). Canonical synthesis files
  (`bin_packing_overnight_results.md`,
  `bin_packing_combined_results.md`) are retained.
- Statistics.jl dropped as a direct dependency (Phase E housekeeping).
