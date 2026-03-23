# Reproducibility

## Fully Reproducible Runs

Arborist.jl supports fully reproducible GP runs via explicit RNG seeding:

```julia
problem = GPProblem(evaluator, ExprGenome; seed=42)
algorithm = GeneticProgramming(parallel=false)  # disable threading
result = solve(problem, algorithm)
```

With `parallel=false` and a fixed seed:
- Population initialization is deterministic
- All random operations use the seeded RNG
- Collection ordering is canonicalized (Dict/Set items are sorted)
- Results are identical across runs and Julia sessions

## Parallel Evaluation and Reproducibility

When `parallel=true` (the default), population evaluation uses
`Threads.@threads`, which introduces non-deterministic evaluation order:

```julia
algorithm = GeneticProgramming(parallel=true)  # default
```

With parallel evaluation:
- Population initialization is still deterministic from the seed
- Evaluation ORDER within a generation is non-deterministic
- This means the exact fitness values may differ between runs if
  evaluation has side effects on shared state (rare for most problems)
- For ExprGenome, `@eval` acquires a global lock, so the actual
  speedup from parallel evaluation is limited

**Recommendation:** Use `parallel=false` for research requiring exact
reproducibility. Use `parallel=true` (default) for production runs
where speed matters more than exact reproducibility.

## TreeGenome Reproducibility

TreeGenome evaluation is fully deterministic (no `@eval`, no global
state). With `parallel=false` and a fixed seed, TreeGenome runs are
perfectly reproducible.

With `parallel=true`, TreeGenome evaluation IS thread-safe and
individual evaluations are independent, but thread scheduling order
is non-deterministic.

## GraphGenome Reproducibility

GraphGenome uses a global innovation counter for NEAT-style crossover.
The counter is reset at the start of each `solve()` call via
`reset_innovation_counter!()`. With `parallel=false` and a fixed seed,
GraphGenome runs are reproducible.

## Starting Julia with Multiple Threads

To benefit from parallel evaluation:
```bash
julia -t auto          # use all available cores
julia -t 4             # use exactly 4 threads
JULIA_NUM_THREADS=auto julia  # via environment variable
```
