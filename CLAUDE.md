# Arborist.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Session Notes

Session updates for this project go to **refstore session notes**, not to
`progress-YYYYMMDD.md` files. Record via `mcp__refstore__write_session_note`
(project key `genprog_jl`, display name `Arborist.jl`) — content and cadence
rules in `CLAUDE.shared.md` still apply (record every commit, include
decisions + rejected alternatives, verification evidence, what was not done).

Existing `progress-YYYYMMDD.md` files at the repo root remain as historical
record — do not append to them. Read them when you need pre-migration
context; write new updates only via refstore.

## Project Status

All tests pass. Fast tier: 3391 tests, ~1m21s. Full benchmarks: 3726 tests, ~26m40s. The single "broken" reported in the test summary is `@test_skip "ANTHROPIC_API_KEY not set"` in `test/integration/test_llm_operator.jl` — `@test_skip` is reported in the same column as `@test_broken`.

**Core framework** (Phases 1-6): Complete. ExprGenome, TreeGenome, AntGenome, GraphGenome. LLM mutation operator. Island model (sequential, sync distributed, async distributed). Speciation (threshold, behavioral). AST sanitizer.

**GraphGenome framework integration (2026-04-20)**: GraphGenome is a first-class genome across the framework. Operator dispatch via `NEATDefaultMutation` + five individual mutation operators (`WeightPerturbMutation`, `WeightReplaceMutation`, `AddConnectionMutation`, `AddNodeMutation`, `ToggleConnectionMutation`) and `NEATCrossover`. Works with NSGA-II (Pareto of fitness vs enabled-connection count), sequential IslandModel, and distributed IslandModel (per-worker disjoint innovation ID ranges via `init_innovation_range!`). `GraphEvaluator` supports recurrent networks via `allow_recurrent=true` with `relaxation_passes=N`; samples are treated as a time sequence with persistent node state. Use `neat_defaults()` to get `(mutation_ops, crossover_ops)` for drop-in use. `_validate_ops` throws a clear `ArgumentError` when incompatible ops (e.g., default ExprGenome ops) are passed with a GraphGenome problem.

**EpisodicEvaluator (2026-04-21)**: Closed-loop control-task evaluator for GraphGenome. Declarative by design — user provides `(initial_state, dynamics, reward, done, observe, decode_action)` callables, rng is threaded through `initial_state(rng)` so reproducibility is structurally enforced, `n_episodes` rollouts are averaged per fitness evaluation. Returns `-mean_reward` (lower-is-better convention). Reuses the forward-pass helper `_apply_node_activations!` shared with `GraphEvaluator`. Works with the GraphGenome `solve` path (constraint widened from `E<:GraphEvaluator` to `E<:AbstractEvaluator`; n_in/n_out now derived from `input_signature`/`output_signature`). See test/benchmarks/cartpole_neat.jl, double_pole_neat.jl, mountain_car_neat.jl for concrete uses. The design choice (declarative default, with a `StatefulEpisodicEvaluator` escape hatch intentionally deferred) is documented in the project memory at `~/.claude/projects/-home-alan-GenProg-jl/memory/episodic_evaluator_design.md`.

**Phase A/B neuroevolution benchmarks (2026-04-21)**: XOR and sequence-memory are joined by 3-bit / 5-bit parity, two-spirals classification (+ NSGA-II variant), single-pole cart-pole, double-pole (Markovian), and Mountain Car. All benchstone-registered (parity-5 is test-only — too seed-sensitive for a 5-rep promotion gate). Plan document: `~/.claude/plans/yes-create-a-plan-smooth-wand.md`. Phase C (non-Markovian stress: double-pole no-velocity, T-Maze, variable-delay sequence recall) and Phase D (retina modularity, Mackey-Glass, NSGA-II retina) are landed.

**Phase E classical GP benchmarks (2026-04-21)**: Rounds out the benchmark suite with five additions selected to cover GP areas the neuroevolution phases leave thin — classical symbolic regression, Boolean scaling, and practical classification.
- **Nguyen-1..10** (`nguyen_regression.jl`): canonical modern SR suite per McDermott et al. 2012. TreeGenome + protected pdiv/plog/psqrt. All 10 equations gate 3/5 seeds at fitness < 0.01 (Nguyen-7 relaxed to 2/5; historically the hardest).
- **Keijzer-4, Keijzer-11** (`keijzer_extrapolation.jl`): extrapolation benchmarks. K-4 reports train fitness plus interior and out-of-range test RMSE as diagnostics; K-11 uses 20 sparse training points with 20×20 test grid. Gates on train fitness; extrapolation RMSE is displayed only.
- **6-bit / 11-bit multiplexer** (`multiplexer.jl`): Koza's canonical Boolean scaler. Uses Boolean-complete {AND, OR, NAND, NOR, XOR, NOT} because DynamicExpressions supports only unary/binary operators (Koza's IF primitive is ternary). The IF-free formulation is strictly harder and has a sharp local optimum at fitness=0.25 ("half-mux"); gates are calibrated accordingly — 6-bit requires 1/5 seeds perfect + 3/5 fitness < 0.3; 11-bit is forward-progress only. Promotion to tight gates would require a ternary primitive in DynamicExpressions.
- **UCI Iris** (`iris_classification.jl`): Fisher's canonical 3-class dataset, 150 rows embedded inline (no network dependency). One-vs-rest with three TreeGenome evolutions, argmax at classification. Gate 4/5 seeds ≥ 90% test accuracy on a fixed 80/20 stratified split. Empirical: 5/5 at 94% mean.
- **Acrobot swing-up** (`acrobot_neat.jl`): Sutton (1996) two-link under-actuated pendulum, RK4 integration at dt=0.2. Pattern mirrors mountain_car_neat.jl. Gate 3/5 seeds fitness < 150 (mean episode < 150 steps). Empirical: 5/5 at ~66 steps.

**NSGA-II multi-objective GP**: `NSGAII` algorithm with non-dominated sorting, crowding distance, and (mu+lambda) survivor selection. `ParsimonyEvaluator` wraps any single-objective evaluator into 2 objectives (fitness + complexity). Returns `NSGAIIResult` with Pareto front and hypervolume history. Example: `examples/nsga2_regression.jl`.

**NSGA-II bin packing (canonical)**: `BPMultiObjectiveEvaluator` in `examples/bin_packing.jl` uses 2 objectives — `(fitness, failed_placements)`. The 2026-04-14 ablation study (`progress-20260414.md`) showed that a third `success_rate` objective is redundant given `failed_placements` (matches best fitness within 10⁻⁴ while producing a noisier Pareto front and hypervolume stuck at 0.0). The historical 3-objective formulation is reproducible via the `three_objective` ablation in `examples/run_nsga2_ablations.jl`.

**Bin packing experiments** (2026-03-23 to 2026-03-24, LLM rerun 2026-04-09): Comprehensive experiments via `--experiment=` flags in `examples/bin_packing.jl`. Full results in `examples/bin_packing_overnight_results.md`.
- **LLM operator** (Qwen3-Coder 30B via Ollama): ~3x sample efficiency in generation count, ~12x wall-time overhead. Multi-seed mean test 1.0664 ± 0.0084 vs classical 1.0717 ± 0.0112. 3/5 seeds beat Best Fit (vs classical 2/5). Original 2026-03-24 LLM data was invalidated by parser + sanitizer bugs that silently fell back to classical ~99% of the time; corrected 2026-04-09.
- **Behavioral speciation**: Lowest variance across seeds (std 0.0027 vs 0.0112 classical). Most reliable but doesn't beat pure best-fit.
- **Best result**: seed 1337 classical GP, test 1.0527 (3.4% better than Best Fit); seed 1337 LLM, test 1.0531 (4.5% better than BF). Extended classical runs (300-800 gen) reach 1.0623.
- Experiment modes: `llm`, `bimodal`, `extended_classical`, `extended_llm`, `multiseed_classical`, `multiseed_behavioral`, `multiseed_llm`, `template_baseline`, `timenorm_classical`, `combine_results`.

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
    linear_genome.jl       # placeholder (empty)
  operators/
    mutation.jl            # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
    crossover.jl           # SubtreeCrossover
    selection.jl           # TournamentSelection
    neat_mutation.jl       # NEAT operators: WeightPerturb/Replace, AddConnection/Node, Toggle, NEATDefaultMutation, NEATCrossover, neat_defaults()
  evaluators.jl            # TableFitnessEvaluator
  algorithm.jl             # GeneticProgramming, IslandModel
  solve.jl                 # solve(), _parallel_evaluate!, _breed_next_generation!
  result.jl                # GPResult
  defaults.jl              # default_function_set(), boolean_function_set()
  speciation.jl            # NoSpeciation, ThresholdSpeciation, BehavioralSpeciation
  topology.jl              # RingTopology, CompleteTopology, RandomTopology
  migration.jl             # MigrantGenome, to_migrant/from_migrant
  distributed_island.jl    # Distributed.jl island model (sync + async)
  llm_operator.jl          # LLMMutationOperator (uses Downloads.jl stdlib)
  tree_genome.jl           # TreeGenome, TreeFitnessEvaluator, SymbolicRegressionEvaluator (DynamicExpressions.jl)
  nsga2.jl                 # NSGAII, ParsimonyEvaluator, NSGAIIResult, non-dominated sorting
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
- **GraphGenome.deserialize not implemented**: Returns `nothing` with a warning. LLM mutation of GraphGenome is non-functional.
- **Distributed NEAT innovation matching is disjoint-range, not content-aware**: Distributed IslandModel gives each worker a unique innovation ID range (`[0, 10^9)`, `[10^9, 2·10^9)`, …) so IDs don't collide. The cost is that structurally identical mutations on different workers receive different IDs — treated as disjoint (non-matching) by NEAT crossover rather than aligned. Per-generation cross-worker innovation dedup is not implemented.
- **EpisodicEvaluator output decoding depends on output-node activation**: The `decode_action` callback receives raw network outputs in the range determined by the output node's activation function (default `:sigmoid` ∈ (0,1)). Threshold-based decoders must account for this — `y > 0` is always true with sigmoid, so thresholds should be at the activation midpoint (0.5 for sigmoid, 0 for tanh/identity). The Phase B control benchmarks all document this in their `_decode` helpers.
- **EpisodicEvaluator is not parallel-ready for stateful environments**: The declarative API is structurally thread-safe (no shared state across evaluations — all state lives in values passed between the callables). But `GraphGenome`'s solve method does not currently set `parallel=true` with episodic benchmarks; the existing `@threads` evaluate loop is fine for pure function callbacks. A benchmark whose dynamics closure captures mutable state would break under `parallel=true` — the planned `StatefulEpisodicEvaluator` (deferred) will address this when it lands.
- **ExprGenome serialize round-trip is partial**: `repr()` produces `Float32(literal)` forms that fail type-checking (~80% round-trip success rate).

## Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                              # fast (~48s)
ARBORIST_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'  # full (~83s)
```

The env var `GENPROG_RUN_BENCHMARKS` is also accepted for backward compatibility.

## Dependencies

Direct dependencies: DynamicExpressions.jl (TreeGenome), Downloads.jl (LLM HTTP, stdlib), Distributed.jl (island model, stdlib), Random.jl (stdlib).
