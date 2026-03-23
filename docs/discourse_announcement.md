# [ANN] Arborist.jl — Generic Genetic Programming for Julia

I'm pleased to announce the initial release of **Arborist.jl**, a generic genetic programming framework for Julia following the Problem/Algorithm/Solve pattern.

## Why now?

The Julia ecosystem has lacked a general-purpose GP framework since Wallace.jl died at Julia 0.3 in 2015. Existing packages (Metaheuristics.jl, Evolutionary.jl) target numerical optimization with vector genomes. Arborist.jl targets *genetic programming* — evolving trees, programs, and neural topologies — with explicit support for the LLM-as-mutation-operator pattern that FunSearch and AlphaEvolve have brought to mainstream attention.

## What does it do?

**Symbolic regression in two lines:**
```julia
using Arborist, DynamicExpressions
const DynExt = Base.get_extension(Arborist, :DynExprExt)

evaluator = DynExt.SymbolicRegressionEvaluator(
    x -> x^4 + x^3 + x^2 + x, domain=(-1f0, 1f0), points=20
)
result = solve(
    GPProblem(evaluator, DynExt.TreeGenome{Float32}; seed=42),
    GeneticProgramming(pop_size=100, generations=200)
)
```

**Four genome types** cover different problem classes:
- `TreeGenome` — DynamicExpressions.jl-backed, 8x faster, for symbolic regression
- `ExprGenome` — Julia AST compilation via @eval, for general program synthesis
- `GraphGenome` — NEAT-style neural topology with structural mutation
- `AntGenome` — side-effectful agent control programs

**NEAT-style neural topology evolution:**
```julia
using Arborist
reset_innovation_counter!()
X = Float64[0 0 1 1; 0 1 0 1]; y = Float64[0 1 1 0]
result = solve(
    GPProblem(GraphEvaluator(X, reshape(y,1,4)), GraphGenome; seed=42),
    GeneticProgramming(pop_size=150, generations=150,
                       speciation=ThresholdSpeciation(threshold=3.0))
)
```

**LLM mutation operator** (Anthropic, OpenAI, or local Ollama) — the FunSearch/AlphaEvolve pattern as a composable operator within a standard evolutionary loop. See the [documentation](https://github.com/CodeReclaimers/Arborist.jl/blob/master/docs/src/llm_operator.md) for details.

## Benchmarks

| Problem | Genome | Generations | Pop Size | Convergence |
|---|---|---|---|---|
| Koza-1 (x⁴+x³+x²+x) | TreeGenome | 300 | 100 | 5/5 seeds |
| Koza-2 (x⁵−2x³+x) | TreeGenome | 300 | 100 | 5/5 seeds |
| Koza-3 (x⁶−2x⁴+x²) | TreeGenome | 300 | 100 | 5/5 seeds |
| XOR (NEAT) | GraphGenome | 150 | 150 | 4/5 seeds |
| Max Ones | ExprGenome | 100 | 100 | 5/5 seeds |

## Related packages

Arborist.jl complements rather than replaces existing packages:
- [DynamicExpressions.jl](https://github.com/MilesCranmer/DynamicExpressions.jl) provides the TreeGenome evaluation backend
- [neat-python](https://github.com/CodeReclaimers/neat-python) is a NEAT implementation by the same author (Python)
- [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) is a specialized SR package; Arborist.jl is a general GP framework

## Try it

```julia
using Pkg; Pkg.add("Arborist")
```

- GitHub: https://github.com/CodeReclaimers/Arborist.jl
- Feedback welcome — issues and PRs are open

A paper describing Arborist.jl with the Koza and NEAT benchmarks is in preparation. FunSearch/AlphaEvolve comparison benchmarks are planned for the next release.
