# Arborist.jl: Generic Genetic Programming in Julia

<p align="center">
  <a href="https://github.com/CodeReclaimers/Arborist.jl/actions"><img src="https://github.com/CodeReclaimers/Arborist.jl/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/CodeReclaimers/Arborist.jl"><img src="https://codecov.io/gh/CodeReclaimers/Arborist.jl/branch/master/graph/badge.svg" alt="Codecov"></a>
  <a href="https://codereclaimers.github.io/Arborist.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation"></a>
  <a href="https://julialang.org"><img src="https://img.shields.io/badge/Julia-1.10+-blue.svg" alt="Julia"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Arborist.jl is a generic, extensible genetic programming framework for
Julia from the maintainer of
[neat-python](https://github.com/CodeReclaimers/neat-python). It follows
the Problem/Algorithm/Solve pattern familiar from the SciML ecosystem
and offers first-class support for NEAT neural topology evolution,
LLM-as-operator, multi-objective optimization, and quality-diversity
search.

## Installation

```julia
using Pkg
Pkg.add("Arborist")
```

`DynamicExpressions.jl` is a direct dependency and is installed automatically —
it backs `TreeGenome` and the symbolic regression evaluator.

## Quick Start

### Symbolic Regression

```julia
using Arborist

evaluator = SymbolicRegressionEvaluator(
    x -> x^4 + x^3 + x^2 + x,
    domain=(-1f0, 1f0), points=20
)

result = solve(
    GPProblem(evaluator, TreeGenome{Float32}; seed=42),
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

### Multi-Objective GP (NSGA-II)

```julia
using Arborist

# Wrap any single-objective evaluator into (fitness, complexity).
inner = SymbolicRegressionEvaluator(
    x -> x^2 + x, domain=(-2f0, 2f0), points=30
)
evaluator = ParsimonyEvaluator(inner)

result = solve(
    GPProblem(evaluator, TreeGenome{Float32}; seed=42),
    NSGAII(pop_size=200, generations=100)
)

println("Pareto front size: ", length(result.pareto_front))
for (g, f) in zip(result.pareto_front, result.pareto_fitnesses)
    println("  mse=", f[1], "  complexity=", f[2], "  ", serialize(g))
end
```

### LLM-Enhanced GP

`LLMMutationOperator` plugs into the regular `mutation_ops` vector as a peer of
`SubtreeMutation`/`PointMutation`. It currently dispatches on `ExprGenome` only
(serialize → prompt → deserialize → type-check, with silent fallback to a
classical operator on any failure):

```julia
using Arborist

llm_op = LLMMutationOperator(
    endpoint    = "http://localhost:11434/v1/chat/completions",
    model       = "qwen2.5-coder:7b",  # lightweight default; runs on ~16 GB RAM.
                                        # The 2026-04-09 bin packing study used
                                        # "qwen3-coder:30b" — see
                                        # examples/bin_packing_overnight_results.md.
    api_key_env = "",                   # local Ollama: no Authorization header
)

algorithm = GeneticProgramming(
    pop_size     = 100,
    generations  = 50,
    mutation_ops = [llm_op, SubtreeMutation(), PointMutation()],
)
```

See `examples/bin_packing.jl` for a complete end-to-end run that uses the LLM
operator on `ExprGenome` with a custom evaluator and function set.

## Features

Arborist.jl follows the **Problem/Algorithm/Solve** pattern from the SciML ecosystem. You define a problem (what to optimize), pick an algorithm (how to optimize), and call `solve`. The framework handles population management, selection, speciation, and fitness history tracking.

Four genome types cover different problem classes. **TreeGenome** uses DynamicExpressions.jl for fast vectorized evaluation of mathematical expressions — roughly 8x faster than `@eval`-based compilation on the Koza symbolic regression suite. **ExprGenome** compiles arbitrary Julia ASTs via `@eval`, supporting loops, conditionals, and mutable state. **AntGenome** specializes in side-effectful agent control programs. **GraphGenome** implements NEAT-style neural topology evolution with innovation-number-aligned crossover and structural mutation.

The **LLM mutation operator** implements the FunSearch/AlphaEvolve pattern: serialize a genome to source, prompt an LLM (Anthropic, OpenAI, or local Ollama) to produce a semantically meaningful variation, parse and validate the response. All failures fall back silently to classical operators — the evolutionary loop is robust to 100% LLM failure rate.

**Multi-objective optimization** is provided via `NSGAII`, a full implementation of non-dominated sorting and crowding distance with (μ+λ) survivor selection. The included `ParsimonyEvaluator` wraps any single-objective evaluator into a two-objective (fitness, complexity) problem so the standard "accuracy vs. bloat" tradeoff is recovered as a real Pareto front instead of a single bloat-penalty compromise. Returned `NSGAIIResult` records the Pareto front (genomes and fitness vectors), the full final population, and per-generation hypervolume history. In 0.1.0, `NSGAII` dispatches on `ExprGenome` and `TreeGenome`; `AntGenome` and `GraphGenome` are not yet supported.

An **island model** with ring/complete/random migration topologies supports population diversity, with three execution backends: sequential (single-process), synchronous distributed, and asynchronous distributed (`Distributed.jl` workers). In 0.1.0, `IslandModel` dispatches on `ExprGenome` and `TreeGenome`; `AntGenome` and `GraphGenome` use their own single-population `solve` paths. **Speciation** is available in two flavors: `ThresholdSpeciation` (NEAT-style genomic-distance speciation with fitness sharing) protects structural innovations, while `BehavioralSpeciation` (probe-based fingerprints) groups syntactically distinct programs that make the same decisions — empirically the most reliable variant on the bin packing benchmark. Explicit **RNG seeding** ensures reproducible runs. An **AST sanitizer** with a configurable function-call whitelist provides defense-in-depth security for `@eval`-based genomes.

## Benchmarks

Convergence gates below are what CI verifies when the benchmark tier is
enabled (`ARBORIST_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg;
Pkg.test()'`; ~27 min wall time). Each gate is a minimum expected-pass
threshold across 5 independent seeds.

### Symbolic regression (TreeGenome)

| Problem | Gate | Passing |
|---|---|---|
| Koza-1 / Koza-2 / Koza-3 | fitness < 0.1 | 3/5 each |
| Nguyen-1..6, -8..10 | fitness < 0.01 | 3/5 |
| Nguyen-7 | fitness < 0.01 | 2/5 |
| Keijzer-4 | train MSE < 0.01 | 3/5 |
| Keijzer-11 | train fit | forward progress |
| Lorenz attractor (dx, dy, dz) | fitness < 0.1 | 3/5 each |

### Boolean synthesis

| Problem | Genome | Gate | Passing |
|---|---|---|---|
| Parity-3 | GraphGenome (NEAT) | fitness < 0.05 | 4/5 |
| 6-bit multiplexer | TreeGenome | perfect + fit < 0.3 | 1/5 + 3/5 |
| 11-bit multiplexer | TreeGenome | forward progress | — |

### Classification

| Problem | Genome | Gate | Passing |
|---|---|---|---|
| XOR | GraphGenome (NEAT) | fitness < 0.01 | 4/5 |
| UCI Iris (one-vs-rest) | TreeGenome | test acc ≥ 90% | 4/5 |
| Two-spirals | GraphGenome (NEAT) | fitness < 0.95 | 3/5 |

### Control tasks (GraphGenome + `EpisodicEvaluator`)

| Problem | Gate | Passing |
|---|---|---|
| Cart-pole | ≥ 195 steps mean | 4/5 |
| Double-pole (Markovian) | fitness < 150 | 3/5 |
| Mountain car | reaches goal | ≥ 2/5 |
| Acrobot swing-up | fitness < 150 | 3/5 |

### Modularity and time series (GraphGenome)

| Problem | Gate | Passing |
|---|---|---|
| Retina left-and-right | MSE < 0.055 | 3/5 |
| Mackey-Glass τ=17 | recurrent ≤ feedforward | smoke |

### Multi-objective (NSGA-II)

| Problem | Gate |
|---|---|
| Two-spirals | best-front error < 1.0; hypervolume > 0 |
| Retina | best-front error < 0.055; hypervolume > 0 |

TreeGenome evaluates 8.4× faster than ExprGenome on the Koza suite
(1000-point dataset).

## Documentation

Full documentation — quick start, user guide, API reference, and the security
model — is built with Documenter.jl and published at
[codereclaimers.github.io/Arborist.jl](https://codereclaimers.github.io/Arborist.jl/dev/).

End-to-end runnable scripts live in [`examples/`](examples/), including:

- `nsga2_regression.jl` — multi-objective symbolic regression
- `bin_packing.jl` — classical and LLM-driven heuristic discovery (see
  `bin_packing_overnight_results.md` for the experimental write-up)
- `sorting.jl` / `sorting_distributed.jl` — sorting-network evolution
- `feynman_regression.jl`, `lorenz_recovery.jl` — physics-flavored benchmarks

## Related Work

Arborist.jl fills a gap left by [Wallace.jl](https://github.com/WallaceLab/Wallace.jl), which was the most ambitious Julia evolutionary computation framework (2014–2015) but died at Julia 0.3 due to reliance on runtime type generation via compiler internals. Arborist.jl avoids this by using only stable public APIs — no `Base.Compiler.*`, no runtime struct generation.

TreeGenome is backed by [DynamicExpressions.jl](https://github.com/MilesCranmer/DynamicExpressions.jl). The LLM mutation operator is inspired by [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) (Romera-Paredes et al., 2024) and [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind, 2025). GraphGenome follows the NEAT encoding from Stanley & Miikkulainen (2002). Benchmark problems follow [Koza (1992)](https://mitpress.mit.edu/9780262111706/genetic-programming/).

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
