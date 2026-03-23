# GenProg.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

**Phase 1 — Core framework: COMPLETE** (2026-03-22)
**Phase 2 — Speciation and operators: COMPLETE** (2026-03-22)
**Phase 3 — LLM operator extension: COMPLETE** (2026-03-23)
**Phase 4 — DynamicExpressions extension: COMPLETE** (2026-03-23)

All 1045 tests pass (1044 + 1 @test_skip). Fast tier: 14s. Full benchmarks: 75s.

## Architecture

```
src/
  GenProg.jl              # module, exports, includes
  abstractions.jl         # abstract type hierarchy (8 abstract types)
  genome/
    codegen.jl            # Expr-tree code generation
    evolution.jl          # crossover, Individual, Population, evolve!
    expr_genome.jl        # ExprGenome, GPProblem, serialize/deserialize
  operators/
    mutation.jl           # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
    crossover.jl          # SubtreeCrossover
    selection.jl          # TournamentSelection
  evaluators.jl           # TableFitnessEvaluator
  algorithm.jl            # GeneticProgramming, IslandModel
  solve.jl                # solve() entry points
  result.jl               # GPResult
  defaults.jl             # default_function_set(), boolean_function_set()
  speciation.jl           # NoSpeciation, ThresholdSpeciation
ext/
  LLMOperatorExt.jl       # LLMMutationOperator (weakdep: HTTP.jl)
  DynExprExt.jl           # TreeGenome, TreeFitnessEvaluator (weakdep: DynamicExpressions.jl)
docs/src/
  llm_operator.md         # FunSearch/AlphaEvolve connection
  tree_genome.md          # TreeGenome vs ExprGenome guide
```

## Two Genome Types

- **TreeGenome** (DynamicExpressions.jl): fast vectorized evaluation for pure function approximation (symbolic regression, boolean parity). No @eval needed. 8x+ faster than ExprGenome.
- **ExprGenome** (@eval): arbitrary Julia programs with control flow, loops, side effects. Required for program synthesis (ant trail, routing programs).

## Key Conventions

- **Explicit RNG everywhere.** GenState carries `rng::AbstractRNG`. No global rand().
- **Deterministic collection ordering.** All sampling from Dict/Set uses `_sorted_*` helpers.
- **Loop safety** via `LoopLimitExceeded` + `add_loop_checks()`. Time limits should be generous (1s+).
- **LLM operator** uses `_http_post` Ref{Function} hook for testability. All failures fall back silently.
- **Tiered tests**: unit+integration run by default (<14s). Benchmarks opt-in via `GENPROG_RUN_BENCHMARKS=true`.

## Accessing Extension Types

```julia
using GenProg, DynamicExpressions
const DynExt = Base.get_extension(GenProg, :DynExprExt)
const TreeGenome = DynExt.TreeGenome
const TreeFitnessEvaluator = DynExt.TreeFitnessEvaluator

using GenProg, HTTP
const LLMExt = Base.get_extension(GenProg, :LLMOperatorExt)
const LLMMutationOperator = LLMExt.LLMMutationOperator
```

## Running Tests

```bash
# Fast tier (unit + integration, ~14s)
julia --project=. -e 'using Pkg; Pkg.test()'

# Full suite with benchmarks (~75s)
GENPROG_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'
```

## Dependencies

Zero mandatory external dependencies. HTTP.jl and DynamicExpressions.jl are weakdeps for optional extensions.
