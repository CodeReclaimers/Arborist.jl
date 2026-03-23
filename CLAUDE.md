# GenProg.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

**Phase 1 — Core framework: COMPLETE** (2026-03-22)
**Phase 2 — Speciation and operators: COMPLETE** (2026-03-22)

All 862 tests pass. `using GenProg` loads cleanly (~340ms precompile).

Phase 3 (LLM operator extension) not yet started.

## Architecture

```
src/
  GenProg.jl              # module, exports, includes
  abstractions.jl         # abstract type hierarchy (8 abstract types)
  genome/
    codegen.jl            # Expr-tree code generation (FunctionDetails, FunctionSet, GenState)
    evolution.jl          # crossover, Individual, Population, evolve!
    expr_genome.jl        # ExprGenome <: AbstractGenome, GPProblem struct
  operators/
    mutation.jl           # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
    crossover.jl          # SubtreeCrossover
    selection.jl          # TournamentSelection
  evaluators.jl           # TableFitnessEvaluator <: AbstractEvaluator
  algorithm.jl            # GeneticProgramming, IslandModel
  solve.jl                # solve() entry points, _run_evolution!, island model solver
  result.jl               # GPResult
  defaults.jl             # default_function_set(), boolean_function_set(), gp_nand, gp_nor
  speciation.jl           # NoSpeciation, ThresholdSpeciation
ext/
  LLMOperatorExt.jl       # placeholder (Phase 3)
  DynExprExt.jl           # placeholder (Phase 4)
```

## Key Conventions

- **Explicit RNG everywhere.** `GenState` carries an `rng::AbstractRNG` field. Every `rand()`/`randn()` call uses `s.rng` or a passed `rng` parameter. No global RNG usage.
- **Deterministic collection ordering.** All sampling from `Dict`/`Set` goes through `_sorted_pairs()`, `_sorted_funcs()`, or `_sorted_types()` to ensure canonical iteration order. Julia's hash-based iteration order varies between processes; without sorting, the GP is non-reproducible even with explicit RNG.
- **Temp variable names are `__temp_$i`**, not `gensym()`. This keeps Dict key hashes deterministic across sessions.
- **`const FitnessEvaluator = AbstractEvaluator`** — compatibility alias so evolution.jl code referencing `FitnessEvaluator` works with the new type hierarchy.
- **`@eval` is used only for compiling evolved programs** (in `evaluate_genome` and `evaluate_individual!`), never for generating framework types or methods. This is the Wallace.jl lesson — see Section 2 of the plan.
- **`Base.invokelatest`** is required when calling `@eval`-defined functions to handle world-age issues. Do not remove it.
- **Loop safety** uses `LoopLimitExceeded` exception via `add_loop_checks()`, not time limits. Time limits in `TableFitnessEvaluator` should be generous (1s+) to avoid GC/JIT non-determinism.
- **Bloat penalty** is applied as `adjusted_fitness = raw_fitness + bloat_penalty * complexity(g)` after evaluation and before selection. Default `bloat_penalty=0.0` preserves existing behavior.
- **Speciation** runs after evaluation and before selection. `ThresholdSpeciation` applies fitness sharing (raw fitness / species size) for selection pressure. `NoSpeciation` leaves fitness unchanged.
- **IslandModel** runs islands sequentially (no threading). Migration uses ring topology every `migration_interval` generations.

## Running Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

862 tests pass. Benchmarks take ~74 minutes total due to @eval overhead in multi-seed convergence tests.

For faster iteration, run unit tests only:
```
julia --project=. -e 'using Test, GenProg, Random; @testset "unit" begin
    include("test/unit/test_evaluators.jl")
    include("test/unit/test_genome.jl")
    include("test/unit/test_operators.jl")
    include("test/unit/test_speciation.jl")
    include("test/unit/test_bloat_penalty.jl")
    include("test/unit/test_island_model.jl")
end'
```

Unit tests pass in ~8 seconds.

## Dependencies

Zero mandatory external dependencies. Only `Random` (stdlib). HTTP.jl and DynamicExpressions.jl are declared as weakdeps for future extensions.

## Plan Document

`genprog_jl_plan.md` in the project root contains the full architecture and development plan across all phases.
