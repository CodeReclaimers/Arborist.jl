# GenProg.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

**Phase 1 — Core framework: COMPLETE** (2026-03-22)
**Phase 2 — Speciation and operators: COMPLETE** (2026-03-22)
**Phase 3 — LLM operator extension: COMPLETE** (2026-03-23)
**Phase 4 — DynamicExpressions extension: COMPLETE** (2026-03-23)
**Phase 5 — Parallelism, sanitizer, new genomes: COMPLETE** (2026-03-23)

All 1135+ tests pass. Fast tier: ~17s. Full benchmarks: ~90s.

## Architecture

```
src/
  GenProg.jl              # module, exports, includes
  abstractions.jl         # 8 abstract types
  sanitizer.jl            # ASTSanitizer — function call whitelist for @eval safety
  genome/
    codegen.jl            # Expr-tree code generation
    evolution.jl          # crossover, Individual, Population, evolve!
    expr_genome.jl        # ExprGenome, GPProblem, serialize/deserialize
    ant_genome.jl         # AntGenome — side-effectful program synthesis
    graph_genome.jl       # GraphGenome — NEAT-style neural topology
  operators/
    mutation.jl           # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
    crossover.jl          # SubtreeCrossover
    selection.jl          # TournamentSelection
  evaluators.jl           # TableFitnessEvaluator
  algorithm.jl            # GeneticProgramming (with parallel field), IslandModel
  solve.jl                # solve(), _parallel_evaluate!
  result.jl               # GPResult
  defaults.jl             # default_function_set(), boolean_function_set()
  speciation.jl           # NoSpeciation, ThresholdSpeciation
ext/
  LLMOperatorExt.jl       # LLMMutationOperator (weakdep: HTTP.jl)
  DynExprExt.jl           # TreeGenome, TreeFitnessEvaluator, SymbolicRegressionEvaluator,
                           # prefix notation parser (weakdep: DynamicExpressions.jl)
```

## Four Genome Types

| Type | Backend | Use Case | Speed |
|---|---|---|---|
| `TreeGenome{T}` | DynamicExpressions.jl | Symbolic regression, function approximation | Fast |
| `ExprGenome` | @eval | General program synthesis with control flow | Slow |
| `AntGenome` | @eval | Agent control (ant trail, robotics) | Slow |
| `GraphGenome` | Custom | Neural topology (NEAT, XOR) | Medium |

## Key Conventions

- **Explicit RNG everywhere.** No global `rand()`.
- **Deterministic ordering.** All Dict/Set sampling uses `_sorted_*` helpers.
- **`parallel` field** on `GeneticProgramming` enables `Threads.@threads` evaluation. Default `true`. Set `false` for exact reproducibility.
- **AST Sanitizer.** `ASTSanitizer` whitelists function calls for @eval safety. See `docs/src/security.md`.
- **Output flushing.** Any benchmark or long-running loop must call `flush(stdout)` after each progress report.
- **Innovation counter.** `GraphGenome` uses a global thread-safe counter. Call `reset_innovation_counter!()` before each solve.

## Running Tests

```bash
# Fast tier (~17s)
julia --project=. -e 'using Pkg; Pkg.test()'

# Full benchmarks (~90s)
GENPROG_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'

# With parallelism
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

## Dependencies

Zero mandatory external dependencies. HTTP.jl and DynamicExpressions.jl are weakdeps.
