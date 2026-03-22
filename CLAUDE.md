# GenProg.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

**Phase 1 — Core framework: COMPLETE** (2026-03-22)

All 695 tests pass. `using GenProg` loads cleanly. Both benchmarks (Max Ones, x² symbolic regression) converge.

Phase 2 (speciation, more operators, island model) not yet started.

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
    mutation.jl           # SubtreeMutation, PointMutation
    crossover.jl          # SubtreeCrossover
    selection.jl          # TournamentSelection
  evaluators.jl           # TableFitnessEvaluator <: AbstractEvaluator
  algorithm.jl            # GeneticProgramming <: AbstractEvolutionaryAlgorithm
  solve.jl                # solve() entry point, _run_evolution!
  result.jl               # GPResult
  defaults.jl             # default_function_set()
  speciation.jl           # NoSpeciation
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

## Running Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

All 695 tests should pass in ~54 seconds.

## Dependencies

Zero mandatory external dependencies. Only `Random` (stdlib). HTTP.jl and DynamicExpressions.jl are declared as weakdeps for future extensions.

## Plan Document

`genprog_jl_plan.md` in the project root contains the full architecture and development plan across all phases. Phase 1 scope is defined in Section 10.
