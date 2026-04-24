# Symbolic Regression with TreeGenome

This tutorial walks through the canonical Koza-1 benchmark
(`y = x⁴ + x³ + x² + x`) end-to-end using `TreeGenome`, the
DynamicExpressions.jl-backed expression-tree genome. It exercises
the fastest evaluation path in Arborist — no `@eval` compilation,
vectorized evaluation over the whole dataset per generation.

## Setup

```julia
using Arborist
using DynamicExpressions: OperatorEnum
```

Pick the operator set. For pure symbolic regression over
`Float32`-valued inputs, the standard choice is the four basic
binary operators plus one unary:

```julia
operators = OperatorEnum(;
    binary_operators = [+, -, *, /],
    unary_operators  = [abs],
)
```

## Generate the training data

Koza-1 evaluates on 20 equispaced points in [−1, 1]:

```julia
xs = Float32.(range(-1, 1, length=20))
X  = reshape(xs, 1, :)          # n_features × n_samples (= 1 × 20)
y  = xs.^4 .+ xs.^3 .+ xs.^2 .+ xs
```

`X` must be a `Matrix{T}` of shape `n_features × n_samples` even
when there is only one input feature.

## Build the evaluator and problem

```julia
evaluator = TreeFitnessEvaluator(X, y, operators)
problem   = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
```

The fitness evaluator computes mean squared error over the dataset
every generation. Specifying `seed=42` makes the run reproducible —
Arborist threads an explicit `MersenneTwister` through the entire
evolution.

## Configure the algorithm

```julia
algorithm = GeneticProgramming(
    pop_size       = 100,
    generations    = 300,
    mutation_rate  = 0.4,
    crossover_rate = 0.2,
    elitism        = 2,
)
```

The defaults fit symbolic regression well: `TournamentSelection(3)` for
parent selection, `SubtreeMutation()` and `PointMutation()` for
mutation, and `SubtreeCrossover()` for recombination. All of these
dispatch on `TreeGenome{Float32}`.

## Solve

```julia
result = solve(problem, algorithm; verbose=true)
```

Verbose output prints the best and mean fitness per generation. With
`seed=42`, this run typically converges to near-zero fitness within
roughly 100 generations.

## Read the result

```julia
println("Best fitness: ", result.best_fitness)
println("Expression:  ", serialize(result.best_genome))
println("Wall time:   ", round(result.wall_time, digits=2), " s")
```

A representative run:

```
Best fitness: 6.83e-14
Expression:   (((x1 * x1) + x1) * (x1 + 1.0)) * x1 + x1
Wall time:    3.27 s
```

The serialized expression is a valid Julia expression reusing the
same variable names as the evaluator (`x1` for the first feature).
`result.best_fitness` is mean squared error; `result.fitness_history`
and `result.mean_history` are full per-generation trajectories that
plot nicely via the [Plots.jl recipes](../plotting.md).

## Next steps

- Extend to **Nguyen-1..10** by swapping the target function —
  `test/benchmarks/nguyen_regression.jl` shows the pattern with
  protected `pdiv / plog / psqrt` operators for functions that would
  otherwise domain-error.
- Add a **bloat penalty** via `GeneticProgramming(; bloat_penalty=1e-4)`
  or recover the accuracy-vs-parsimony Pareto front with
  [NSGA-II](nsga2_parsimony.md).
- Refine the constants found by GP using the optional BFGS pass:
  `GeneticProgramming(; constant_optimization=ConstantOptimization(frequency=25, top_k=5))`.
