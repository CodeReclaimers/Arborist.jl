# [ANN] Arborist.jl — Generic Genetic Programming for Julia

I'm pleased to announce the initial release of **Arborist.jl**, a generic genetic programming framework for Julia following the Problem/Algorithm/Solve pattern.

## Why now?

The Julia ecosystem has lacked a general-purpose GP framework since Wallace.jl died at Julia 0.3 in 2015. Existing packages (Metaheuristics.jl, Evolutionary.jl) target numerical optimization with vector genomes. Arborist.jl targets *genetic programming* — evolving trees, programs, and neural topologies — with explicit support for the LLM-as-mutation-operator pattern that FunSearch and AlphaEvolve have brought to mainstream attention.

## What does it do?

**Symbolic regression in two lines:**
```julia
using Arborist, DynamicExpressions

evaluator = SymbolicRegressionEvaluator(
    x -> x^4 + x^3 + x^2 + x, domain=(-1f0, 1f0), points=20
)
result = solve(
    GPProblem(evaluator, TreeGenome{Float32}; seed=42),
    GeneticProgramming(pop_size=100, generations=200)
)
```

**Five genome types** cover different problem classes:
- `TreeGenome` — DynamicExpressions.jl-backed, 8x faster, for symbolic regression
- `ExprGenome` — Julia AST compilation via @eval, for general program synthesis
- `GraphGenome` — NEAT-style neural topology with structural mutation
- `AntGenome` — side-effectful agent control programs
- `ADFGenome` — Koza-style Automatically Defined Functions on top of TreeGenome

**NEAT-style neural topology evolution:**
```julia
using Arborist
reset_innovation_counter!()
X = Float64[0 0 1 1; 0 1 0 1]; y = Float64[0 1 1 0]
ops = neat_defaults()
result = solve(
    GPProblem(GraphEvaluator(X, reshape(y,1,4)), GraphGenome; seed=42),
    GeneticProgramming(pop_size=150, generations=150,
                       mutation_ops=ops.mutation_ops,
                       crossover_ops=ops.crossover_ops,
                       speciation=ThresholdSpeciation(threshold=3.0))
)
```

**LLM mutation operator** (Anthropic, OpenAI, or local Ollama) — the FunSearch/AlphaEvolve pattern as a composable operator within a standard evolutionary loop. See the [documentation](https://github.com/CodeReclaimers/Arborist.jl/blob/master/docs/src/llm_operator.md) for details.

## Benchmarks

A representative slice of the benchmark suite (`ARBORIST_RUN_BENCHMARKS=true`,
~27 min wall time). Each gate is verified across 5 independent seeds.

| Problem | Genome | Gate | Passing |
|---|---|---|---|
| Koza-1 / Koza-2 / Koza-3 | TreeGenome | fitness < 0.1 | 3/5 each |
| Nguyen-1..6, -8..10 | TreeGenome | fitness < 0.01 | 3/5 |
| XOR | GraphGenome (NEAT) | fitness < 0.01 | 4/5 |
| UCI Iris (one-vs-rest) | TreeGenome | test acc ≥ 90% | 4/5 |
| Cart-pole | GraphGenome (NEAT) | ≥ 195 steps mean | 4/5 |
| Two-spirals (NSGA-II) | GraphGenome | best-front error < 1.0; HV > 0 | — |

See the [README](https://github.com/CodeReclaimers/Arborist.jl#benchmarks)
for the full set covering 25+ problems across symbolic regression, Boolean
synthesis, classification, control tasks, modularity, time series, and
multi-objective formulations.

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
