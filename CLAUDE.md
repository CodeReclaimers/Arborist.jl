# Arborist.jl

> **Note to human readers:** this file is internal guidance for Claude Code
> agents working on the Arborist.jl codebase. It is not user-facing
> documentation — for that, see `README.md`, `docs/src/`, or the hosted
> documentation site. The file is tracked in the public repo so that future
> agent sessions inherit consistent project conventions, known limitations,
> and deferred-work context.

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Session Notes

Session updates for this project go to **refstore session notes**, not to
`progress-YYYYMMDD.md` files. Record via `mcp__refstore__write_session_note`
(project key `genprog_jl`, display name `Arborist.jl`) — content and cadence
rules in `CLAUDE.shared.md` still apply (record every commit, include
decisions + rejected alternatives, verification evidence, what was not done).

The pre-migration `progress-YYYYMMDD.md` files were removed from the repo
tree on 2026-04-23 (Item 2a of the release-readiness plan) but remain
recoverable from git history — use `git log --all --diff-filter=D
--name-only` to rediscover paths, then `git show <sha>:<path>` to retrieve
content.

## Project Status

All tests pass. Fast tier: 3864 tests, ~1m44s. Full benchmarks: 4202 tests, ~27m23s. The single "broken" reported in the test summary is `@test_skip "ANTHROPIC_API_KEY not set"` in `test/integration/test_llm_operator.jl` — `@test_skip` is reported in the same column as `@test_broken`.

**Core framework** (Phases 1-6): Complete. ExprGenome, TreeGenome, AntGenome, GraphGenome. LLM mutation operator. Island model (sequential, sync distributed, async distributed). Speciation (threshold, behavioral). AST sanitizer.

**GraphGenome framework integration (2026-04-20)**: GraphGenome is a first-class genome across the framework. Operator dispatch via `NEATDefaultMutation` + five individual mutation operators (`WeightPerturbMutation`, `WeightReplaceMutation`, `AddConnectionMutation`, `AddNodeMutation`, `ToggleConnectionMutation`) and `NEATCrossover`. Works with NSGA-II (Pareto of fitness vs enabled-connection count), sequential IslandModel, and distributed IslandModel (per-worker disjoint innovation ID ranges via `init_innovation_range!`). `GraphEvaluator` supports recurrent networks via `allow_recurrent=true` with `relaxation_passes=N`; samples are treated as a time sequence with persistent node state. Use `neat_defaults()` to get `(mutation_ops, crossover_ops)` for drop-in use. `_validate_ops` throws a clear `ArgumentError` when incompatible ops (e.g., default ExprGenome ops) are passed with a GraphGenome problem.

**EpisodicEvaluator (2026-04-21)**: Closed-loop control-task evaluator for GraphGenome. Declarative by design — user provides `(initial_state, dynamics, reward, done, observe, decode_action)` callables, rng is threaded through `initial_state(rng)` so reproducibility is structurally enforced, `n_episodes` rollouts are averaged per fitness evaluation. Returns `-mean_reward` (lower-is-better convention). Reuses the forward-pass helper `_apply_node_activations!` shared with `GraphEvaluator`. Works with the GraphGenome `solve` path (constraint widened from `E<:GraphEvaluator` to `E<:AbstractEvaluator`; n_in/n_out now derived from `input_signature`/`output_signature`). See test/benchmarks/cartpole_neat.jl, double_pole_neat.jl, mountain_car_neat.jl for concrete uses. The design choice (declarative default, with a `StatefulEpisodicEvaluator` escape hatch intentionally deferred) is documented in the project memory at `~/.claude/projects/-home-alan-GenProg-jl/memory/episodic_evaluator_design.md`.

**Phase A/B neuroevolution benchmarks (2026-04-21)**: XOR and sequence-memory are joined by 3-bit / 5-bit parity, two-spirals classification (+ NSGA-II variant), single-pole cart-pole, double-pole (Markovian), and Mountain Car. All benchstone-registered (parity-5 is test-only — too seed-sensitive for a 5-rep promotion gate). Plan document: `~/.claude/plans/yes-create-a-plan-smooth-wand.md`. Phase C (non-Markovian stress: double-pole no-velocity, T-Maze, variable-delay sequence recall) and Phase D (retina modularity, Mackey-Glass, NSGA-II retina) are landed.

**Phase E classical GP benchmarks (2026-04-21)**: Rounds out the benchmark suite with five additions selected to cover GP areas the neuroevolution phases leave thin — classical symbolic regression, Boolean scaling, and practical classification.
- **Nguyen-1..10** (`nguyen_regression.jl`): canonical modern SR suite per McDermott et al. 2012. TreeGenome + protected pdiv/plog/psqrt. All 10 equations gate 3/5 seeds at fitness < 0.01 (Nguyen-7 relaxed to 2/5; historically the hardest).
- **Keijzer-4, Keijzer-11** (`keijzer_extrapolation.jl`): extrapolation benchmarks. K-4 reports train fitness plus interior and out-of-range test RMSE as diagnostics; K-11 uses 20 sparse training points with 20×20 test grid. Gates on train fitness; extrapolation RMSE is displayed only.
- **6-bit / 11-bit multiplexer** (`multiplexer.jl`): Koza's canonical Boolean scaler. Uses Boolean-complete {AND, OR, NAND, NOR, XOR, NOT} because DynamicExpressions supports only unary/binary operators (Koza's IF primitive is ternary). The IF-free formulation is strictly harder and has a sharp local optimum at fitness=0.25 ("half-mux"); gates are calibrated accordingly — 6-bit requires 3/5 seeds < 0.3 (the perfect-fit gate was dropped 2026-05-03 because IF-free convergence is unreliable across Julia patch versions: 1.10 stalls at 0.0625, 1.11 reaches 0); 11-bit is forward-progress only. Promotion to tight gates would require a ternary primitive in DynamicExpressions.
- **UCI Iris** (`iris_classification.jl`): Fisher's canonical 3-class dataset, 150 rows embedded inline (no network dependency). One-vs-rest with three TreeGenome evolutions, argmax at classification. Gate 4/5 seeds ≥ 90% test accuracy on a fixed 80/20 stratified split. Empirical: 5/5 at 94% mean.
- **Acrobot swing-up** (`acrobot_neat.jl`): Sutton (1996) two-link under-actuated pendulum, RK4 integration at dt=0.2. Pattern mirrors mountain_car_neat.jl. Gate 3/5 seeds fitness < 150 (mean episode < 150 steps). Empirical: 5/5 at ~66 steps.

**NSGA-II multi-objective GP**: `NSGAII` algorithm with non-dominated sorting, crowding distance, and (mu+lambda) survivor selection. `ParsimonyEvaluator` wraps any single-objective evaluator into 2 objectives (fitness + complexity). Returns `NSGAIIResult` with Pareto front and hypervolume history. Example: `examples/nsga2_regression.jl`.

**NSGA-II bin packing (canonical)**: `BPMultiObjectiveEvaluator` in `examples/bin_packing.jl` uses 2 objectives — `(fitness, failed_placements)`. The 2026-04-14 ablation study showed that a third `success_rate` objective is redundant given `failed_placements` (matches best fitness within 10⁻⁴ while producing a noisier Pareto front and hypervolume stuck at 0.0). The historical 3-objective formulation is reproducible via the `three_objective` ablation in `examples/research/run_nsga2_ablations.jl`.

**Bin packing experiments** (2026-03-23 to 2026-03-24, LLM rerun 2026-04-09): Comprehensive experiments via `--experiment=` flags in `examples/bin_packing.jl`. Full results in `examples/bin_packing_overnight_results.md`.
- **LLM operator** (Qwen3-Coder 30B via Ollama): ~3x sample efficiency in generation count, ~12x wall-time overhead. Multi-seed mean test 1.0664 ± 0.0084 vs classical 1.0717 ± 0.0112. 3/5 seeds beat Best Fit (vs classical 2/5). Original 2026-03-24 LLM data was invalidated by parser + sanitizer bugs that silently fell back to classical ~99% of the time; corrected 2026-04-09.
- **Behavioral speciation**: Lowest variance across seeds (std 0.0027 vs 0.0112 classical). Most reliable but doesn't beat pure best-fit.
- **Best result**: seed 1337 classical GP, test 1.0527 (3.4% better than Best Fit); seed 1337 LLM, test 1.0531 (4.5% better than BF). Extended classical runs (300-800 gen) reach 1.0623.
- Experiment modes: `llm`, `bimodal`, `extended_classical`, `extended_llm`, `multiseed_classical`, `multiseed_behavioral`, `multiseed_llm`, `template_baseline`, `timenorm_classical`, `combine_results`.

**Phase F research-grade framework gaps (2026-04-22)**: Closes eight capability gaps a neuroevolution / GP researcher would notice in the first hour, selected from a survey of the library against modern GP literature. Plan document: `~/.claude/plans/phase-f-framework-gaps.md`. All phases landed in a single day.
- **F.0 Per-case fitness API + structured RunLog**: `evaluate_cases(g, e)` generic (opt-in per evaluator; `MethodError` on miss). `RunLog` + `GenerationLog` + `record!` with best/mean/median/worst fitness, species count + sizes, unique-structure hash diversity, per-operator attempt/success (F.5), wall time. `SpeciationSnapshot` carrier threads through `_apply_speciation!` as an optional kwarg so the solve path can log without recomputing speciation. `log::Union{Nothing, RunLog}` kwarg on every single-objective solve path. Commit `1ce761c`.
- **F.1 Lexicase + epsilon-lexicase selection**: `LexicaseSelection` and `EpsilonLexicaseSelection(; epsilon=0.0)`. Polymorphic `select_parent(strategy, sel_fits, case_fits, rng)` dispatch replaces the hardcoded `_tournament_select` in `_breed_next_generation!`. `needs_cases(selection)` trait — when `true`, solve materializes the per-individual per-case matrix via `evaluate_cases`. Auto-epsilon uses MAD per case (La Cava et al. 2016). Modal-regression test-only benchmark documents the end-to-end path; epsilon-lexicase beats tournament on 4/5 seeds in that problem, but pure lexicase does not — advantage is problem-class-dependent. Commit `d1eb1f2`.
- **F.2 GraphGenome deserialize + LLM-on-NEAT**: `deserialize(::Type{GraphGenome}, s, n_inputs, n_outputs; reassign_innovations=false)` implemented (was a `nothing`-returning stub). Tolerates LLM preamble / code fences. `reassign_innovations=true` issues fresh innovation IDs via `_next_innovation!()` so LLM-generated IDs don't collide with the parent pool. `mutate(::LLMMutationOperator, ::GraphGenome, rng)` mirrors the ExprGenome path; fallback_op must dispatch on GraphGenome (e.g. `NEATDefaultMutation()`). Commit `ecdd0c0`.
- **F.3 Constant optimization for TreeGenome SR**: Periodic BFGS pass on top-K individuals every N generations. Central finite-difference gradients (Zygote would be too heavy; `differentiable_eval_tree_array` doesn't actually return gradients). `ConstantOptimization(; frequency=25, top_k=5, max_iter=50, tol=1e-8, fd_step=1e-3)` field on `GeneticProgramming`. Dense BFGS with Armijo line search; rollback guard so a genome's fitness can't get worse. `optimize_constants!(g, e; kwargs)` entry point for manual use. Commit `541d652`.
- **F.4 Novelty Search + MAP-Elites**: `NoveltyArchive{B}` thread-safe bounded-size store + `NoveltySearchEvaluator{F,D,B}` wrapping a fingerprint function and distance metric; `evaluate_genome` returns `-mean_knn_distance`. `MAPElites <: AbstractEvolutionaryAlgorithm` with grid-based archive, own solve loop that samples parents from the filled archive; `MAPElitesArchive{G}`, `MAPElitesResult{G}`, `coverage(archive)`, `qd_score(archive)`. Commit `ebbce0f`.
- **F.5 Checkpoint/resume + operator-success tracking**: `Checkpoint{G}` + `save_checkpoint(ckpt, path)` (atomic via tmp+rename) + `load_checkpoint(path)` (format-version + Julia-version stamp, rejects mismatches). `solve(...; checkpoint_every, checkpoint_path, resume_from, allow_signature_mismatch)` on ExprGenome and TreeGenome solve paths. `_algorithm_signature` hashes hyperparameter scalars + operator type identities (not `hash(alg)` — the struct holds closures whose hash is identity-based). All-time-best tracking added: `GPResult.best_*` survives elitism loss. `operator_name(op) -> Symbol` generic; `_breed_next_generation!` fills `op_track` which the solve loop tallies into `GenerationLog.operator_attempted` / `operator_success`. Commit `8d002fc`.
- **F.6 Plots.jl recipes via package extension**: `ext/ArboristRecipesBaseExt.jl` defines `@recipe` rules for `GPResult` (fitness trajectory), `NSGAIIResult` (Pareto front, 2D + 3D), `RunLog` (3-subplot: fitness, species, structures), `MAPElitesResult` (coverage + QD histories). `@userplot`s: `plothypervolumetrajectory(nsga2_result)`, `plotarchive(map_elites_archive)`. Weak dep on RecipesBase; zero runtime cost without Plots. Commit `b4a1549`.
- **F.7 Automatically Defined Functions (ADFs)**: `ADFGenome{T}` with main tree + N Koza-style ADF subroutines, fixed binary arity. Main tree's operator enum is augmented with N placeholder binary slots representing ADF calls. Macro-expansion via `expand_adfs(g) -> Node{T}`: walks the main tree, replaces ADF-slot ops with copies of the ADF body with ARG references substituted. ARG_i encoded as feature index `n_features + i + 1`. Full AbstractGenome interface (mutation, crossover, distance, complexity, serialize); `evaluate_adf(g, X, y)`. ADF-from-ADF calls not supported (prevents infinite expansion). Commit `cf45257`.
- **F.8 CMA-ES**: Full (μ_w/μ, λ)-CMA-ES with rank-1 + rank-μ covariance updates, evolution-path step-size control, hsig Heaviside step. `CMAES <: AbstractEvolutionaryAlgorithm` with auto-λ (Hansen 2016 default). `flatten_weights(g)` / `unflatten_weights!(g, w)` interface, opt-in per genome; `GraphGenome` adaptor sorts by innovation for determinism. LinearAlgebra added to deps (stdlib). Rastrigin-5 best_fitness ≈ 2.98 vs random baseline ~50. Commit `8534143`.

## Deferred Research Roadmap

Surveyed gaps that Phase F did not implement, either because of scope caps per phase or because they represent larger separate projects. Captured here so future work has a durable record.

### Neuroevolution extensions
- **HyperNEAT / CPPN / ES-HyperNEAT**. Dominant method for scaling neuroevolution past direct-encoded networks. Needs a CPPN genome, a substrate abstraction, and a HyperNEAT-specific evaluator. Significant surface; separate project.
- **Deep-neuroevolution / policy-gradient hybrids**. Flux.jl integration for gradient-trained weights on evolved topologies. Would pair naturally with F.8's CMA-ES topology freeze.
- **Age-layered population structure (ALPS)**. Deceptive-landscape diversity via age cohorts. Modest effort; complements F.4's QD approaches.
- **HyperNEAT-specific activation sets (partial)**. `ACTIVATION_FNS` now has {sigmoid, tanh, relu, identity, gauss, sin, abs, step} — the common CPPN set. `AddNodeMutation` / `NEATDefaultMutation` still default their `hidden_activations` kwarg to just {sigmoid, tanh, relu}, so CPPN activations require an opt-in. Not yet included: softsign, elu, swish, cos (less standard for NEAT but used in some architectures).
- **GraphGenome network topology plot recipe**. Would need a layout algorithm (spring / layered / hierarchical) — ideally another weakdep extension against GraphPlot.jl / NetworkLayout.jl rather than rolling our own.
- **Content-aware cross-worker innovation matching for distributed IslandModel**. Currently disjoint-range; content matching would let NEAT crossover align structurally identical mutations across workers.

### Selection strategies
- **Fitness-proportionate / roulette-wheel**, **rank**, **stochastic universal sampling (SUS)**, **truncation**. Round out the selection menu beyond F.1's tournament + lexicase.
- **Lexicase + NSGA-II**. Combining per-case filtering with Pareto sorting is an active research area; out of F.1's single-objective scope.
- **Semantic-aware selection** (semantic neutrality check before accepting offspring). Sits next to F.3's constant opt as a "local refinement" pass.

### Genome representations
- **Cartesian GP (CGP) / Linear GP / PushGP**. `LinearGenome` is a stub; each of these representations is a separate major implementation. All three are mainstream GP research branches.
- **Grammar-based GP / Grammatical Evolution (CFG-GP)**. Yet another mainstream branch. Sits separately from tree- and graph-based genomes.
- **Strongly Typed GP** beyond what DynamicExpressions offers. For richer type constraints than `T` alone.
- **ADF extensions**: mixed arity per ADF; nested ADF-from-ADF calls (requires cycle detection); `ADFFitnessEvaluator` that integrates F.7's ADFGenome with the existing `evaluate_genome` dispatch; LLM mutation for ADFGenome.
- **Geometric semantic GP operators** (Moraglio-Krawiec-Johnson). Semantic crossover/mutation that combine parents by semantic interpolation rather than syntactic swap.
- **Automatically Defined Macros / Loops / Iterations** (Koza's ADM/ADL/ADI). F.7 covers ADFs; the others follow similar patterns.

### Search strategies
- **Steady-state GP**. One-at-a-time replacement rather than generational.
- **Cooperative co-evolution** (SANE, ESP): multiple populations evolving sub-solutions. `competitive` and `host-parasite` variants likewise.
- **Adaptive hyperparameters**: self-adaptive mutation rates; `ThresholdSpeciation` auto-tuning to target species count (canonical NEAT behavior); operator-bandit credit assignment using F.5's operator-success tracking as the signal.
- **BIPOP-CMA-ES / IPOP-CMA-ES**. Multi-restart variants that extend F.8. Standard practice for hard problems.
- **Separable CMA-ES (sep-CMA-ES)**. Diagonal-covariance variant for high-dim problems where the full O(n^3) eigendecomposition per generation becomes expensive.

### Infrastructure
- **Checkpoint/resume for AntGenome, GraphGenome, IslandModel, NSGA-II**. F.5 landed for ExprGenome + TreeGenome only; extending is mechanical threading per solve path.
- **GPU evaluation path** (CUDA.jl / KernelAbstractions.jl for batch evaluation). Would pair most naturally with TreeGenome SR where DynamicExpressions has vectorized eval.
- **Stateful environment parallelism** — `StatefulEpisodicEvaluator` already flagged as deferred in the existing EpisodicEvaluator section above.
- **First-class test-problem generators** (`Benchmarks.nguyen(n)`, `Benchmarks.cartpole()`, etc). Each benchmark hand-rolls its data inline; a generator module would let users run the same problems outside `test/`.
- **Hall-of-Fame / all-time archive beyond per-run best**. F.5 tracks a single all-time best; an archive of the top-K across all generations would support richer analysis.
- **Per-island checkpointing for IslandModel**. F.5 ships global-best checkpoint only.
- **Per-species size dynamics plot recipe**. RunLog already records `species_sizes` per generation; stacked-area recipe is a natural follow-on to F.6.
- **`plot_operator_success(log)`**. F.5 populates operator dicts; a dedicated recipe would make per-operator hit rates visually scannable.

### Symbolic regression polish
- **Ephemeral random constants (Koza)**. Node-creation-time constant sampling. Complements F.3's BFGS refinement.
- **Comparison harness against SymbolicRegression.jl / PySR**. The closest Julia ecosystem neighbors; a head-to-head benchmark would be useful documentation.
- **ADF-aware constant optimization**. F.3's BFGS + F.7's ADFGenome don't currently compose. Extension: walk the expanded tree's constants for BFGS; requires mapping back to the pre-expansion constant refs.
- **NSGA-II constant optimization**. F.3 integrates with single-objective GP; multi-objective integration needs a scalarization choice.

### LLM operator extensions
- **LLM mutation for TreeGenome**. Prefix-notation system prompt exists (`DEFAULT_TREE_GP_SYSTEM_PROMPT`) but the `mutate(::LLMMutationOperator, ::TreeGenome, rng)` method isn't wired.
- **LLM mutation for AntGenome, ADFGenome**. Same pattern as the ExprGenome and (new in F.2) GraphGenome paths.

## Post-Registration Manual Steps

These steps happen after this Claude Code session:

1. **Registry PR**: Comment `@JuliaRegistrator register` on the latest commit at https://github.com/CodeReclaimers/Arborist.jl (requires JuliaRegistrator GitHub App). Alternatively, follow https://github.com/JuliaRegistries/General#registering-a-new-package
2. **Wait for merge**: New package registry PRs typically take 3 days
3. **Tag release**: `git tag v0.1.0 && git push --tags`
4. **Mint Zenodo DOI**: Connect the GitHub repo to Zenodo and create a release
5. **Update README.md**: Replace the placeholder citation block with the actual DOI
6. **Post Discourse announcement**: Review `docs/discourse_announcement.md` and post to https://discourse.julialang.org/c/package-announcements
7. **Set up Documenter deployment**: Add `DOCUMENTER_KEY` secret to GitHub repo settings, then push to trigger docs build

## Architecture

```
src/
  Arborist.jl              # module, exports, includes
  abstractions.jl          # 8 abstract types + AbstractTopology
  sanitizer.jl             # ASTSanitizer -- function call whitelist
  genome/
    codegen.jl             # Expr-tree code generation (GenState, FunctionSet)
    evolution.jl           # crossover, Individual, Population, evolve! (legacy API)
    expr_genome.jl         # ExprGenome, GPProblem, serialize/deserialize
    ant_genome.jl          # AntGenome -- side-effectful programs
    graph_genome.jl        # GraphGenome + GraphEvaluator + EpisodicEvaluator (control tasks)
    adf_genome.jl          # ADFGenome -- main tree + N ADF subroutines via macro expansion (Phase F.7)
  operators/
    mutation.jl            # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
    crossover.jl           # SubtreeCrossover
    selection.jl           # TournamentSelection
    neat_mutation.jl       # NEAT operators: WeightPerturb/Replace, AddConnection/Node, Toggle, NEATDefaultMutation, NEATCrossover, neat_defaults()
  evaluators.jl            # TableFitnessEvaluator
  algorithm.jl             # GeneticProgramming, IslandModel
  solve.jl                 # solve(), _parallel_evaluate!, _breed_next_generation!, select_parent
  result.jl                # GPResult
  defaults.jl              # default_function_set(), boolean_function_set()
  run_log.jl               # RunLog, GenerationLog, record!, SpeciationSnapshot (Phase F.0/F.5)
  speciation.jl            # NoSpeciation, ThresholdSpeciation, BehavioralSpeciation
  topology.jl              # RingTopology, CompleteTopology, RandomTopology
  migration.jl             # MigrantGenome, to_migrant/from_migrant
  distributed_island.jl    # Distributed.jl island model (sync + async)
  llm_operator.jl          # LLMMutationOperator (uses Downloads.jl stdlib)
  prompt_context.jl        # MutationContext + AbstractPromptSection -- LLM prompt enrichment
  tree_genome.jl           # TreeGenome, TreeFitnessEvaluator, SymbolicRegressionEvaluator, optimize_constants! (Phase F.3)
  constant_optimization.jl # ConstantOptimization config (Phase F.3)
  nsga2.jl                 # NSGAII, ParsimonyEvaluator, NSGAIIResult, non-dominated sorting
  novelty.jl               # NoveltyArchive, NoveltySearchEvaluator (Phase F.4)
  map_elites.jl            # MAPElites, MAPElitesArchive, MAPElitesResult (Phase F.4)
  checkpoint.jl            # Checkpoint, save_checkpoint, load_checkpoint (Phase F.5)
  cmaes.jl                 # CMAES, flatten_weights, unflatten_weights! (Phase F.8)
ext/
  ArboristRecipesBaseExt.jl  # Plots.jl recipes via RecipesBase weakdep (Phase F.6)
```

## Key Conventions

- **Explicit RNG everywhere.** No global `rand()`.
- **Deterministic ordering.** All Dict/Set sampling uses `_sorted_*` helpers.
- **`parallel=true`** enables `Threads.@threads` evaluation. Set `false` for reproducibility.
- **Fitness sharing for minimization**: configurable via `sharing_formula` -- `:log2` (default), `:sqrt`, `:linear`, `:none`. Use `apply_sharing(raw, size, formula)`.
- **Output flushing**: `flush(stdout)` after progress output in long-running loops.
- **Innovation counter**: the single-objective, NSGA-II, and sequential IslandModel `solve` methods reset the process-global innovation counter on entry when `G === GraphGenome`. Distributed IslandModel uses `init_innovation_range!((island_id - 1) * INNOVATION_STRIDE)` per worker to keep innovation IDs disjoint across processes.
- **AST Sanitizer**: opt-in whitelist for @eval security. See `docs/src/security.md`.

## Known Limitations

- **@eval method table growth**: Every ExprGenome evaluation adds a method to Julia's method table. Long runs accumulate thousands of methods. TreeGenome avoids this via DynamicExpressions' compiled evaluation.
- **AntGenome not thread-safe**: Uses a module-level `Ref` for simulator state. Runtime error if `parallel=true`. Bin packing and sorting examples demonstrate the thread-local state workaround.
- **Distributed NEAT innovation matching is disjoint-range, not content-aware**: Distributed IslandModel gives each worker a unique innovation ID range (`[0, 10^9)`, `[10^9, 2·10^9)`, …) so IDs don't collide. The cost is that structurally identical mutations on different workers receive different IDs — treated as disjoint (non-matching) by NEAT crossover rather than aligned. Per-generation cross-worker innovation dedup is not implemented.
- **EpisodicEvaluator output decoding depends on output-node activation**: The `decode_action` callback receives raw network outputs in the range determined by the output node's activation function (default `:sigmoid` ∈ (0,1)). Threshold-based decoders must account for this — `y > 0` is always true with sigmoid, so thresholds should be at the activation midpoint (0.5 for sigmoid, 0 for tanh/identity). The Phase B control benchmarks all document this in their `_decode` helpers.
- **EpisodicEvaluator is not parallel-ready for stateful environments**: The declarative API is structurally thread-safe (no shared state across evaluations — all state lives in values passed between the callables). But `GraphGenome`'s solve method does not currently set `parallel=true` with episodic benchmarks; the existing `@threads` evaluate loop is fine for pure function callbacks. A benchmark whose dynamics closure captures mutable state would break under `parallel=true` — the planned `StatefulEpisodicEvaluator` (deferred) will address this when it lands.
- **ExprGenome serialize round-trip is partial**: `repr()` produces `Float32(literal)` forms that fail type-checking (~80% round-trip success rate).

## Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                              # fast (~1m44s)
ARBORIST_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'  # full (~27m23s)
```

The env var `GENPROG_RUN_BENCHMARKS` is also accepted for backward compatibility.

## Dependencies

Direct dependencies: CommonSolve.jl (solve verb), DynamicExpressions.jl (TreeGenome), Downloads.jl (LLM HTTP, stdlib), Distributed.jl (island model, stdlib), Random.jl (stdlib), Serialization.jl (checkpointing, stdlib), LinearAlgebra.jl (CMA-ES, stdlib).

Weak dependencies (extensions): RecipesBase.jl (`ext/ArboristRecipesBaseExt.jl` for plotting). Loaded automatically when both Arborist and any RecipesBase consumer (Plots.jl, Makie, etc.) are imported.

## Deferred Release-Readiness Items

The release-readiness plan at
`~/.claude/plans/great-review-thanks-please-smooth-parrot.md` enumerates 21
pre-registration cleanup items. Items deferred during execution are tracked
here so they are not lost between sessions.

- **Item 2d — bench/ directory cleanup (resolved 2026-05-02).** The
  `bench/` directory is an integration layer for the benchstone benchmark
  harness. benchstone was published to PyPI on 2026-05-02
  (https://pypi.org/project/benchstone/, source at
  https://github.com/CodeReclaimers/benchstone), so the Arborist `bench/`
  README and manifest were sanitized to reference the public PyPI/GitHub
  URLs instead of the local development path. No outstanding work.
