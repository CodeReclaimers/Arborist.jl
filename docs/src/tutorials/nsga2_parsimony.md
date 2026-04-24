# Multi-Objective Symbolic Regression with NSGA-II

Classical symbolic regression trades off fit quality against
expression complexity via a `bloat_penalty` coefficient — a scalar
compromise. NSGA-II computes the full Pareto front in one run,
letting you inspect the accuracy-vs-parsimony tradeoff directly.

## The data and operators

```julia
using Arborist
using DynamicExpressions: OperatorEnum
using Printf

xs = Float32.(range(-2, 2, length=30))
X  = reshape(xs, 1, :)
y  = xs .^ 2 .+ xs

operators = OperatorEnum(;
    binary_operators = [+, -, *, /],
    unary_operators  = [sin, cos, abs],
)
```

## Wrap the evaluator for two objectives

`ParsimonyEvaluator` wraps any single-objective `AbstractEvaluator`
into a two-objective `(fitness, complexity)` problem. `complexity(g)`
counts nodes in the expression tree.

```julia
inner     = TreeFitnessEvaluator(X, y, operators)
evaluator = ParsimonyEvaluator(inner)

println("Objectives: ", objective_names(evaluator))
# Output: ["fitness", "complexity"]
```

## Configure NSGA-II

```julia
problem   = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
algorithm = NSGAII(
    pop_size       = 200,   # must be even and ≥ 4
    generations    = 100,
    mutation_rate  = 0.3,
    crossover_rate = 0.3,
    parallel       = false,
)

result = solve(problem, algorithm; verbose=true)
```

NSGA-II uses non-dominated sorting + crowding distance with (μ+λ)
survivor selection. Selection pressure emerges from the sorting — no
tournament / lexicase here — so `NSGAII` does not expose `elitism`
or `selection` kwargs.

## Read the Pareto front

```julia
# Deduplicate by (MSE rounded to 6 digits, complexity)
seen = Set{Tuple{Float64, Float64}}()
unique_indices = Int[]
for (i, fit) in enumerate(result.pareto_fitnesses)
    key = (round(fit[1], digits=6), fit[2])
    if key ∉ seen
        push!(seen, key)
        push!(unique_indices, i)
    end
end

println(lpad("MSE", 12), "  ", lpad("Nodes", 6), "  ", "Expression")
println("-" ^ 60)
for i in unique_indices
    fit = result.pareto_fitnesses[i]
    expr = serialize(result.pareto_front[i])
    mse_str = fit[1] < 0.001 ?
        @sprintf("%.2e", fit[1]) :
        @sprintf("%.6f", fit[1])
    println(lpad(mse_str, 12), "  ", lpad(Int(fit[2]), 6), "  ", expr)
end
```

A typical run produces a Pareto front with solutions ranging from
(MSE large, complexity 1) at one extreme to (MSE tiny, complexity 15+)
at the other. Somewhere around complexity 5 is the "right" answer
(`x^2 + x`, or equivalently `x*x + x`).

## Hypervolume and wall time

```julia
println("Wall time:          ", round(result.wall_time, digits=2), " s")
println("Final hypervolume:  ", round(result.hypervolume_history[end], digits=4))
```

`result.hypervolume_history` is a per-generation scalar — useful for
plotting (see [Plotting](../plotting.md)) and for comparing runs
across different algorithm configurations.

## Next steps

- Supported genomes: `NSGAII` currently dispatches on `ExprGenome` and
  `TreeGenome`. `GraphGenome` NSGA-II is supported via specialized
  benchmarks (`test/benchmarks/xor_nsga2_neat.jl`,
  `two_spirals_nsga2_neat.jl`, `retina_nsga2_neat.jl`) but not via
  the general `NSGAII` algorithm yet.
- Custom two-objective formulations (e.g. accuracy vs robustness)
  subtype `AbstractMultiObjectiveEvaluator` and implement
  `evaluate_multi(e, g) -> Vector{Float64}` plus `objective_names(e)`.
- Plot the front: `plot(result)` from the Plots.jl extension draws a
  2D scatter of the Pareto front; `plothypervolumetrajectory(result)`
  shows the hypervolume history.
