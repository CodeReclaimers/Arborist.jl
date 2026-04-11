# Arborist.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

All tests pass. Fast tier: 3164 tests, ~1m08s. Full benchmarks: 3186 tests, ~8m53s. The single "broken" reported in the test summary is `@test_skip "ANTHROPIC_API_KEY not set"` in `test/integration/test_llm_operator.jl` — `@test_skip` is reported in the same column as `@test_broken`.

**Core framework** (Phases 1-6): Complete. ExprGenome, TreeGenome, AntGenome, GraphGenome. LLM mutation operator. Island model (sequential, sync distributed, async distributed). Speciation (threshold, behavioral). AST sanitizer.

**NSGA-II multi-objective GP**: `NSGAII` algorithm with non-dominated sorting, crowding distance, and (mu+lambda) survivor selection. `ParsimonyEvaluator` wraps any single-objective evaluator into 2 objectives (fitness + complexity). Returns `NSGAIIResult` with Pareto front and hypervolume history. Example: `examples/nsga2_regression.jl`.

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
    graph_genome.jl        # GraphGenome -- NEAT-style neural topology
    linear_genome.jl       # placeholder (empty)
  operators/
    mutation.jl            # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
    crossover.jl           # SubtreeCrossover
    selection.jl           # TournamentSelection
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
- **Innovation counter**: `reset_innovation_counter!()` before each GraphGenome solve.
- **AST Sanitizer**: opt-in whitelist for @eval security. See `docs/src/security.md`.

## Known Limitations

- **@eval method table growth**: Every ExprGenome evaluation adds a method to Julia's method table. Long runs accumulate thousands of methods. TreeGenome avoids this via DynamicExpressions' compiled evaluation.
- **AntGenome not thread-safe**: Uses a module-level `Ref` for simulator state. Runtime error if `parallel=true`. Bin packing and sorting examples demonstrate the thread-local state workaround.
- **GraphGenome.deserialize not implemented**: Returns `nothing` with a warning. LLM mutation of GraphGenome is non-functional.
- **Distributed NEAT innovation collisions**: Separate workers assign conflicting node IDs. Must be addressed before enabling GraphGenome with `distributed=true`.
- **ExprGenome serialize round-trip is partial**: `repr()` produces `Float32(literal)` forms that fail type-checking (~80% round-trip success rate).

## Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                              # fast (~48s)
ARBORIST_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'  # full (~83s)
```

The env var `GENPROG_RUN_BENCHMARKS` is also accepted for backward compatibility.

## Dependencies

Direct dependencies: DynamicExpressions.jl (TreeGenome), Downloads.jl (LLM HTTP, stdlib), Distributed.jl (island model, stdlib), Random.jl (stdlib).
