# Plotting Results

Arborist.jl ships a weakdep extension
(`ext/ArboristRecipesBaseExt.jl`) that registers Plots.jl recipes for
the main result types. The extension loads automatically when both
Arborist and any RecipesBase consumer (Plots.jl, Makie, etc.) are
imported — no explicit opt-in required.

```julia
using Plots
using Arborist
```

Zero runtime cost when Plots is not loaded.

## Fitness trajectory: `plot(::GPResult)`

```julia
result = solve(problem, algorithm)
plot(result)
```

Draws a two-line chart: the best fitness per generation (bottom
envelope) and the mean fitness per generation (upper line). Both
come from `result.fitness_history` and `result.mean_history`.

## Pareto front: `plot(::NSGAIIResult)`

For two-objective problems:

```julia
result = solve(problem, NSGAII(pop_size=200, generations=100))
plot(result)
```

Draws a scatter plot of the Pareto front. For three-objective
problems the recipe emits a 3D scatter; for ≥ 4 objectives the plot
falls back to a parallel-coordinates style view.

## Hypervolume trajectory: `plothypervolumetrajectory`

```julia
plothypervolumetrajectory(result)   # result::NSGAIIResult
```

Per-generation hypervolume from `result.hypervolume_history` (2D
objective problems only; 3+ objective problems record 0.0 and the
plot is flat).

## Structured run log: `plot(::RunLog)`

```julia
log = RunLog()
solve(problem, algorithm; log=log)
plot(log)
```

Renders three subplots: fitness trajectory (best / mean / worst),
species count and maximum species size over time, and unique
structure count (a hash-based diversity proxy).

## MAP-Elites archive: `plot(::MAPElitesResult)` and `plotarchive`

```julia
result = solve(problem, MAPElites(bins=(10,10), generations=100))

# Coverage + QD-score history as two subplots:
plot(result)

# The final archive state (bin fitness as a heatmap):
plotarchive(result.archive)
```

## Saving figures

All recipes return a `Plots.Plot` object — use the standard Plots API:

```julia
p = plot(result)
savefig(p, "run_fitness.pdf")
savefig(p, "run_fitness.png")
```

Or specify the backend up front:

```julia
using Plots
pyplot()       # Matplotlib backend, for publication figures
plot(result)
```

## Custom plots

When a recipe doesn't cover the specific view you want, every result
type exposes the underlying data directly:

- `result.fitness_history::Vector{Float64}` — best per generation.
- `result.mean_history::Vector{Float64}` — population mean per generation.
- `result.pareto_fitnesses::Vector{Vector{Float64}}` — Pareto front
  objective vectors (NSGA-II).
- `result.hypervolume_history::Vector{Float64}` — per-generation HV.
- `log.entries::Vector{GenerationLog}` — full structured log from
  `solve(...; log=RunLog())`, one entry per generation, containing
  operator-attempted / operator-success dicts, species sizes,
  structural diversity, and wall time.
