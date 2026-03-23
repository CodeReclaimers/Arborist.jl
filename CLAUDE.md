# GenProg.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

**Phase 1 — Core framework: COMPLETE** (2026-03-22)
**Phase 2 — Speciation and operators: COMPLETE** (2026-03-22)
**Phase 3 — LLM operator extension: COMPLETE** (2026-03-23)

All 932 tests pass (862 Phase 1-2 + 70 Phase 3). `using GenProg` loads cleanly without HTTP.jl. Extension loads automatically when HTTP.jl is present.

Phase 4 (DynamicExpressions extension, TreeGenome, registration) not yet started.

## Architecture

```
src/
  GenProg.jl              # module, exports, includes
  abstractions.jl         # abstract type hierarchy (8 abstract types)
  genome/
    codegen.jl            # Expr-tree code generation (FunctionDetails, FunctionSet, GenState)
    evolution.jl          # crossover, Individual, Population, evolve!
    expr_genome.jl        # ExprGenome <: AbstractGenome, GPProblem, serialize/deserialize
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
  LLMOperatorExt.jl       # LLMMutationOperator (weakdep: HTTP.jl)
  DynExprExt.jl           # placeholder (Phase 4)
docs/
  src/
    llm_operator.md       # FunSearch/AlphaEvolve connection, quick-start, Ollama guide
test/
  mocks/mock_http.jl      # Mock HTTP infrastructure for LLM operator testing
  integration/test_llm_operator.jl  # LLM operator integration tests
```

## Key Conventions

- **Explicit RNG everywhere.** `GenState` carries an `rng::AbstractRNG` field. Every `rand()`/`randn()` call uses `s.rng` or a passed `rng` parameter. No global RNG usage.
- **Deterministic collection ordering.** All sampling from `Dict`/`Set` goes through `_sorted_pairs()`, `_sorted_funcs()`, or `_sorted_types()` to ensure canonical iteration order.
- **Temp variable names are `__temp_$i`**, not `gensym()`. This keeps Dict key hashes deterministic across sessions.
- **`const FitnessEvaluator = AbstractEvaluator`** — compatibility alias so evolution.jl code referencing `FitnessEvaluator` works with the new type hierarchy.
- **`@eval` is used only for compiling evolved programs**, never for generating framework types or methods.
- **`Base.invokelatest`** is required when calling `@eval`-defined functions to handle world-age issues. Do not remove it.
- **Loop safety** uses `LoopLimitExceeded` exception via `add_loop_checks()`, not time limits. Time limits in `TableFitnessEvaluator` should be generous (1s+) to avoid GC/JIT non-determinism.
- **Bloat penalty** is applied as `adjusted_fitness = raw_fitness + bloat_penalty * complexity(g)` after evaluation and before selection.
- **Speciation** runs after evaluation and before selection. `ThresholdSpeciation` applies fitness sharing.
- **IslandModel** runs islands sequentially (no threading). Migration uses ring topology.
- **LLM operator** uses a `_http_post` Ref{Function} hook for testability. Tests replace it with a mock (see `test/mocks/mock_http.jl`). All LLM failures fall back to `fallback_op` silently.
- **serialize** uses `repr()` which produces `:()` wrapped output. **deserialize** unwraps QuoteNodes and type-checks assignments with partial recovery (invalid lines are skipped, not rejected wholesale).

## Accessing the LLM Extension

```julia
using GenProg
using HTTP  # triggers extension loading

LLMExt = Base.get_extension(GenProg, :LLMOperatorExt)
op = LLMExt.LLMMutationOperator()  # Anthropic API default
```

See `docs/src/llm_operator.md` for full examples including Ollama and OpenAI.

## Running Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

932 tests pass. Benchmarks take ~74 minutes due to @eval overhead.

For faster iteration, run unit + integration tests only (~8 seconds):
```julia
using Test, GenProg, Random, HTTP
@testset "quick" begin
    include("test/unit/test_evaluators.jl")
    include("test/unit/test_genome.jl")
    include("test/unit/test_operators.jl")
    include("test/unit/test_speciation.jl")
    include("test/unit/test_bloat_penalty.jl")
    include("test/unit/test_island_model.jl")
    include("test/integration/test_llm_operator.jl")
end
```

## Dependencies

Zero mandatory external dependencies. Only `Random` (stdlib). HTTP.jl and DynamicExpressions.jl are declared as weakdeps for optional extensions.

## Plan Document

`genprog_jl_plan.md` in the project root contains the full architecture and development plan across all phases.
