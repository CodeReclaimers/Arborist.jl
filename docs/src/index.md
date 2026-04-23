# Arborist.jl

**Generic genetic programming for Julia.**

Arborist.jl is an extensible GP framework from the maintainer of
[neat-python](https://github.com/CodeReclaimers/neat-python), built on
the Problem/Algorithm/Solve pattern familiar from the SciML ecosystem.
It provides five genome types (`ExprGenome`, `TreeGenome`, `GraphGenome`,
`AntGenome`, `ADFGenome`), composable genetic operators,
LLM-as-mutation-operator support, NEAT-style and behavioral speciation,
NSGA-II multi-objective optimization, quality-diversity search (Novelty
Search, MAP-Elites), CMA-ES, and a sequential/sync/async distributed
island model — all with explicit RNG seeding for reproducible research.

## Who is this for?

- Researchers who need a typed, reproducible GP framework with multiple genome representations
- Practitioners exploring symbolic regression, neural topology search, or program synthesis
- Anyone interested in composing LLM operators with classical evolutionary operators (the FunSearch/AlphaEvolve pattern)

## How does it differ from other Julia EC packages?

- **Wallace.jl** (2014–2015) was the most ambitious Julia EC framework but died at Julia 0.3 due to runtime type generation via compiler internals. Arborist.jl uses only stable public APIs.
- **Metaheuristics.jl** and **Evolutionary.jl** target numerical optimization with vector genomes. Arborist.jl targets genetic *programming* with tree, graph, and AST genomes.
- **SymbolicRegression.jl** is a specialized symbolic regression package. Arborist.jl is a general GP framework that can do symbolic regression (via TreeGenome) but also program synthesis, neural topology evolution, and agent control.

## Benchmarks

The `test/benchmarks/` tier (enabled via `ARBORIST_RUN_BENCHMARKS=true`)
exercises ~25 problems spanning:

- **Symbolic regression** — Koza-1/2/3, Nguyen-1..10, Keijzer-4/11,
  Lorenz attractor recovery.
- **Boolean synthesis** — parity-3 (NEAT), 6- and 11-bit multiplexer.
- **Classification** — XOR (NEAT), UCI Iris, two-spirals.
- **Control tasks** — cart-pole, double-pole (Markovian), mountain
  car, acrobot swing-up, all via `GraphGenome` + `EpisodicEvaluator`.
- **Modularity and time series** — retina left-and-right (NEAT),
  Mackey-Glass τ=17.
- **Multi-objective (NSGA-II)** — two-spirals and retina with
  hypervolume histories.

See [the README's Benchmarks section](https://github.com/CodeReclaimers/Arborist.jl#benchmarks)
for the per-problem convergence gates.

Get started with the [Quick Start](@ref).
