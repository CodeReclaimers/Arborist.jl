# GenProg.jl

**Generic genetic programming for Julia.**

GenProg.jl is an extensible GP framework built on the Problem/Algorithm/Solve pattern from the SciML ecosystem. It provides four genome types, composable genetic operators, LLM-as-mutation-operator support, and NEAT-style speciation — all with explicit RNG seeding for reproducible research.

## Who is this for?

- Researchers who need a typed, reproducible GP framework with multiple genome representations
- Practitioners exploring symbolic regression, neural topology search, or program synthesis
- Anyone interested in composing LLM operators with classical evolutionary operators (the FunSearch/AlphaEvolve pattern)

## How does it differ from other Julia EC packages?

- **Wallace.jl** (2014–2015) was the most ambitious Julia EC framework but died at Julia 0.3 due to runtime type generation via compiler internals. GenProg.jl uses only stable public APIs.
- **Metaheuristics.jl** and **Evolutionary.jl** target numerical optimization with vector genomes. GenProg.jl targets genetic *programming* with tree, graph, and AST genomes.
- **SymbolicRegression.jl** is a specialized symbolic regression package. GenProg.jl is a general GP framework that can do symbolic regression (via TreeGenome) but also program synthesis, neural topology evolution, and agent control.

## Benchmarks

| Problem | Genome | Generations | Pop Size | Convergence |
|---|---|---|---|---|
| Koza-1 (x⁴+x³+x²+x) | TreeGenome | 300 | 100 | 5/5 seeds |
| Koza-2 (x⁵−2x³+x) | TreeGenome | 300 | 100 | 5/5 seeds |
| Koza-3 (x⁶−2x⁴+x²) | TreeGenome | 300 | 100 | 5/5 seeds |
| XOR (NEAT) | GraphGenome | 150 | 150 | 4/5 seeds |
| Max Ones | ExprGenome | 100 | 100 | 5/5 seeds |

Get started with the [Quick Start guide](@ref).
