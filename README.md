<p align="center">
  <img src="docs/src/assets/logo.jpg" alt="Arborist.jl" width="600">
</p>

<p align="center">
  <a href="https://github.com/CodeReclaimers/Arborist.jl/actions"><img src="https://github.com/CodeReclaimers/Arborist.jl/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://julialang.org"><img src="https://img.shields.io/badge/Julia-1.10+-blue.svg" alt="Julia"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

## Installation

```julia
using Pkg
Pkg.add("Arborist")
```

For symbolic regression (recommended), also install DynamicExpressions.jl:
```julia
Pkg.add("DynamicExpressions")
```

## Quick Start

### Symbolic Regression

```julia
using Arborist, DynamicExpressions
const DynExt = Base.get_extension(Arborist, :DynExprExt)

evaluator = DynExt.SymbolicRegressionEvaluator(
    x -> x^4 + x^3 + x^2 + x,
    domain=(-1f0, 1f0), points=20
)

result = solve(
    GPProblem(evaluator, DynExt.TreeGenome{Float32}; seed=42),
    GeneticProgramming(pop_size=100, generations=200)
)
println("Best fitness: ", result.best_fitness)
println("Best expression: ", serialize(result.best_genome))
```

### Neural Topology Evolution (XOR with NEAT)

```julia
using Arborist
reset_innovation_counter!()

X = Float64[0 0 1 1; 0 1 0 1]
y = Float64[0 1 1 0]
evaluator = GraphEvaluator(X, reshape(y, 1, 4))

result = solve(
    GPProblem(evaluator, GraphGenome; seed=42),
    GeneticProgramming(
        pop_size=150, generations=150,
        mutation_rate=0.5, crossover_rate=0.3,
        speciation=ThresholdSpeciation(threshold=3.0)
    )
)
println("Best fitness: ", result.best_fitness)
```

### LLM-Enhanced GP

```julia
using Arborist, DynamicExpressions, HTTP
const DynExt = Base.get_extension(Arborist, :DynExprExt)
const LLMExt = Base.get_extension(Arborist, :LLMOperatorExt)

evaluator = DynExt.SymbolicRegressionEvaluator(
    x -> sin(x) * x, domain=(-3f0, 3f0), points=30
)
llm_op = LLMExt.LLMMutationOperator(api_key_env="ANTHROPIC_API_KEY")

result = solve(
    GPProblem(evaluator, DynExt.TreeGenome{Float32}; seed=42),
    GeneticProgramming(
        mutation_ops = [llm_op, SubtreeMutation(), PointMutation()]
    )
)
```

## Features

Arborist.jl follows the **Problem/Algorithm/Solve** pattern from the SciML ecosystem. You define a problem (what to optimize), pick an algorithm (how to optimize), and call `solve`. The framework handles population management, selection, speciation, and fitness history tracking.

Four genome types cover different problem classes. **TreeGenome** uses DynamicExpressions.jl for fast vectorized evaluation of mathematical expressions — 8x faster than compilation-based approaches. **ExprGenome** compiles arbitrary Julia ASTs via `@eval`, supporting loops, conditionals, and mutable state. **AntGenome** specializes in side-effectful agent control programs. **GraphGenome** implements NEAT-style neural topology evolution with innovation-number-aligned crossover and structural mutation.

The **LLM mutation operator** implements the FunSearch/AlphaEvolve pattern: serialize a genome to source, prompt an LLM (Anthropic, OpenAI, or local Ollama) to produce a semantically meaningful variation, parse and validate the response. All failures fall back silently to classical operators — the evolutionary loop is robust to 100% LLM failure rate.

An **island model** with ring-topology migration supports population diversity. **NEAT-style speciation** with fitness sharing protects structural innovations. Explicit **RNG seeding** ensures reproducible runs. An **AST sanitizer** provides defense-in-depth security for `@eval`-based genomes.

## Benchmarks

| Problem | Genome | Generations | Pop Size | Convergence |
|---|---|---|---|---|
| Koza-1 (x⁴+x³+x²+x) | TreeGenome | 300 | 100 | 5/5 seeds |
| Koza-2 (x⁵−2x³+x) | TreeGenome | 300 | 100 | 5/5 seeds |
| Koza-3 (x⁶−2x⁴+x²) | TreeGenome | 300 | 100 | 5/5 seeds |
| XOR (NEAT) | GraphGenome | 150 | 150 | 4/5 seeds |
| Max Ones | ExprGenome | 100 | 100 | 5/5 seeds |
| 4-bit Even Parity | TreeGenome | 500 | 200 | 1/5 seeds |

TreeGenome evaluates 8.4x faster than ExprGenome on the Koza suite (1000-point dataset).

## Related Work

Arborist.jl fills a gap left by [Wallace.jl](https://github.com/WallaceLab/Wallace.jl), which was the most ambitious Julia evolutionary computation framework (2014–2015) but died at Julia 0.3 due to reliance on runtime type generation via compiler internals. Arborist.jl avoids this by using only stable public APIs — no `Base.Compiler.*`, no runtime struct generation.

TreeGenome is backed by [DynamicExpressions.jl](https://github.com/MilesCranmer/DynamicExpressions.jl). The LLM mutation operator is inspired by [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) (Romera-Paredes et al., 2024) and [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind, 2025). GraphGenome follows the NEAT encoding from Stanley & Miikkulainen (2002). Benchmark problems follow [Koza (1992)](https://mitpress.mit.edu/9780262111706/genetic-programming/). The same author maintains [neat-python](https://github.com/CodeReclaimers/neat-python).

## License

MIT License. See [LICENSE](LICENSE) for details.

```bibtex
@software{arborist_jl,
  author = {CodeReclaimers LLC},
  title  = {Arborist.jl: Generic Genetic Programming for Julia},
  year   = {2026},
  url    = {https://github.com/CodeReclaimers/Arborist.jl}
}
```
