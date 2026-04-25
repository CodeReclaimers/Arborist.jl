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

Two tiers of benchmark surface ship with the package:

- **`Arborist.Benchmarks` submodule.** Canonical genetic-programming and
  neuroevolution problems as reusable data generators that user code can
  consume directly — `Benchmarks.nguyen(n)`, `Benchmarks.cartpole()`,
  `Benchmarks.iris()`, etc. Each generator returns a `NamedTuple`
  carrying the dataset (or environment callables), shape metadata,
  target descriptions, and sensible success thresholds. See the
  [Benchmarks API reference](@ref Arborist.Benchmarks) and the module
  docstring for the full coverage list and usage sketches.

  ```julia
  using Arborist, DynamicExpressions
  prob = Arborist.Benchmarks.nguyen(1)
  evaluator = TreeFitnessEvaluator(prob.X, prob.y,
      OperatorEnum(binary_operators=[+, -, *, /],
                   unary_operators=[sin, cos, exp]))
  result = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=42),
                 GeneticProgramming())
  ```

- **`test/benchmarks/` tier** (enabled via `ARBORIST_RUN_BENCHMARKS=true`)
  exercises ~25 problems against fixed convergence gates:
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

## Run utilities

Three cross-cutting helpers eliminate ad-hoc boilerplate in benchmark
and comparison scripts:

- `train_test_split(X, y; test_size, rng, stratify)` — deterministic
  train/test partition with optional class-stratified sampling.
- `summarize(xs)` — `(; mean, std, median, min, max, q25, q75, n)`
  named tuple over a vector of reals; non-finite entries are
  excluded so a single `Inf` fitness does not poison the report.
- `run_multi_seed(f, seeds; parallel=false)` — call `f(seed)` for each
  seed and return a vector of results, optionally parallel.

Typical pairing:

```julia
fitnesses = run_multi_seed([1, 2, 3, 4, 5]) do seed
    problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
    result  = solve(problem, alg)
    result.best_fitness
end
println(summarize(fitnesses))
```

Get started with the [Quick Start](@ref).
