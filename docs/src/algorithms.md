# Algorithms

## GeneticProgramming

The standard GP algorithm with tournament selection, elitism, and configurable genetic operators.

```julia
algorithm = GeneticProgramming(
    pop_size = 100,        # population size
    generations = 200,     # number of generations
    mutation_rate = 0.3,   # probability of mutation per offspring
    crossover_rate = 0.3,  # probability of crossover per pair
    elitism = 2,           # top individuals carried forward
    tournament_size = 3,   # tournament selection size
    bloat_penalty = 0.0,   # coefficient on complexity(g)
    parallel = true,       # threaded evaluation
    speciation = NoSpeciation(),
    mutation_ops = [SubtreeMutation(), PointMutation()],
    crossover_ops = [SubtreeCrossover()],
    selection = TournamentSelection(3),
)
```

## IslandModel

Multiple independent populations with periodic ring-topology migration.

```julia
algorithm = IslandModel(
    n_islands = 4,
    island_algorithm = GeneticProgramming(pop_size=50, generations=100),
    migration_interval = 10,  # generations between migrations
    migration_size = 2,       # individuals migrated per event
)
```

!!! note "Genome support in 0.1.0"
    `IslandModel` (sequential, synchronous distributed, and asynchronous
    distributed) dispatches on `ExprGenome` and `TreeGenome`. For
    `TreeGenome`, the evaluator must be a `TreeFitnessEvaluator`;
    migration transports `Node{T}` directly across workers so the op
    indices remain valid against every island's shared `OperatorEnum`.
    `AntGenome` and `GraphGenome` must use the single-population
    `GeneticProgramming` solver for now — the former is blocked by a
    thread-unsafe simulator, the latter by the process-local NEAT
    innovation counter. Extending `IslandModel` to those genome types
    is planned for a future release.

## NSGAII

Multi-objective GP using non-dominated sorting and crowding distance with
(μ+λ) survivor selection. Requires an evaluator that implements
`evaluate_multi(evaluator, genome) -> Vector{Float64}` (i.e., a subtype of
`AbstractMultiObjectiveEvaluator`). The included `ParsimonyEvaluator` wraps
any single-objective `AbstractEvaluator` into the standard
`(fitness, complexity)` two-objective problem.

```julia
using Arborist, DynamicExpressions

inner = SymbolicRegressionEvaluator(
    x -> x^2 + x, domain=(-2f0, 2f0), points=30
)
evaluator = ParsimonyEvaluator(inner)

algorithm = NSGAII(
    pop_size       = 200,    # must be even, ≥ 4
    generations    = 100,
    mutation_rate  = 0.3,
    crossover_rate = 0.3,
    parallel       = true,
    mutation_ops   = [SubtreeMutation(), PointMutation()],
    crossover_ops  = [SubtreeCrossover()],
)

result = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=42), algorithm)
```

`solve(::GPProblem, ::NSGAII)` returns an `NSGAIIResult` with these fields:

- `pareto_front::Vector{G}` — non-dominated genomes from the final population
- `pareto_fitnesses::Vector{Vector{Float64}}` — objective vectors for the
  Pareto-front genomes
- `population::Vector{G}` and `all_fitnesses::Vector{Vector{Float64}}` — full
  final population and its objective vectors
- `hypervolume_history::Vector{Float64}` — hypervolume of front 1 per
  generation (2D only; problems with three or more objectives return 0.0)
- `generations_run::Int`, `wall_time::Float64`, `objective_names::Vector{String}`

See [`examples/nsga2_regression.jl`](https://github.com/CodeReclaimers/Arborist.jl/blob/master/examples/nsga2_regression.jl)
for a complete runnable example.

!!! note "Genome support in 0.1.0"
    `NSGAII` dispatches on `ExprGenome` and `TreeGenome`. `AntGenome` and
    `GraphGenome` are not yet supported — attempting to `solve` an NSGA-II
    problem with those genome types raises a `MethodError` from
    `_nsga2_init_population`. Extending NSGA-II to the remaining genome
    types is planned for a future release.

## CMAES

Covariance Matrix Adaptation Evolution Strategy ((μ_w/μ, λ)-CMA-ES,
Hansen 2016) for continuous-parameter optimization. Used to refine
the weight vector of a genome whose topology is fixed — for example,
a `GraphGenome` whose connections you want to tune after a NEAT
search has discovered the structure. The algorithm itself does not
mutate topology; pair it with a topology search if both are needed.

```julia
algorithm = CMAES(
    generations = 200,
    pop_size    = 0,       # 0 → auto λ = 4 + ⌊3·ln(n)⌋ (Hansen 2016 default)
    sigma0      = 0.5,     # initial step size
    seed_genome = true,    # start the mean from the problem's initial genome
)
```

The genome must opt in by implementing `flatten_weights(g) ->
Vector{Float64}` and `unflatten_weights!(g, w) -> g`. The
`GraphGenome` adaptor sorts connections by innovation number for
deterministic packing; other genomes can implement these two methods
to participate. See the
[Infrastructure API reference](api/infrastructure.md) for details.

## MAPElites

Quality-Diversity search via grid-based behavioral archives (Mouret
& Clune, 2015). Replaces single-best convergence with a *coverage*
goal: fill as many distinct behavior cells as possible, keeping the
fittest representative in each. `MAPElites` runs its own solve loop
that samples parents from the filled archive — there is no notion of
"population" carried across generations.

```julia
algorithm = MAPElites(
    feature_fn     = g -> (my_behavior_x(g), my_behavior_y(g)),
    feature_bounds = [(0.0, 1.0), (0.0, 1.0)],
    n_bins         = [20, 20],
    generations    = 200,
    batch_size     = 50,    # children produced per generation
    n_init         = 200,   # seed pool of random genomes (gen 0)
    mutation_ops   = [SubtreeMutation()],
)

result = solve(GPProblem(evaluator, ExprGenome; ...), algorithm)
```

`feature_fn` returns an `NTuple{N, Float64}` (or any indexable
length-N collection of `Real`) describing where the genome lands in
behavior space. `feature_bounds` and `n_bins` discretize that space —
features outside `[lo, hi]` clamp to the extreme bin. `coverage` and
`qd_score` are queryable on the archive, and the result type is
`MAPElitesResult{G}` with `archive`, `coverage_history`, and
`qd_score_history`. The Plots.jl recipe `plotarchive(result)`
produces the canonical 2D heatmap (currently 2D archives only).

## Novelty Search

Replace the fitness signal with *behavioral novelty* — the negative
mean distance from the k nearest neighbors in a thread-safe bounded
archive of past behaviors. Useful on deceptive problems where the
fitness gradient points away from the global optimum.

```julia
archive  = NoveltyArchive(Vector{Float64};
                           max_size      = 2000,
                           add_threshold = 0.0)

evaluator = NoveltySearchEvaluator(
    g       -> behavioral_signature(g),    # fingerprint_fn :: genome → fingerprint
    (a, b)  -> norm(a - b),                # distance_fn   :: (fp, fp) → Float64
    archive;
    k       = 15,
)

algorithm = GeneticProgramming(
    pop_size      = 200,
    generations   = 200,
    mutation_ops  = [SubtreeMutation(), PointMutation()],
)

result = solve(GPProblem(evaluator, ExprGenome; ...), algorithm)
```

`NoveltySearchEvaluator` returns `-mean_knn_distance`, so the standard
minimization solve loop drives the population toward novelty. To
combine novelty with task fitness, build a multi-objective evaluator
and use `NSGAII` with both signals.

## Constant optimization

`GeneticProgramming` accepts an optional `constant_optimization` field
that runs a periodic BFGS pass over the numeric constants of the top-K
genomes:

```julia
algorithm = GeneticProgramming(
    pop_size = 100,
    generations = 200,
    constant_optimization = ConstantOptimization(
        frequency = 25,    # run every 25 generations
        top_k     = 5,     # refine the top 5 individuals
        max_iter  = 50,
        tol       = 1e-8,
        fd_step   = 1e-3,  # central finite-difference gradient step
    ),
)
```

Currently implemented for `TreeGenome` symbolic regression. The pass
is rollback-guarded: a refined genome's fitness can only improve. To
optimize a single genome's constants by hand, call
`optimize_constants!(g, evaluator; kwargs...)`.

## Checkpoint and resume

Long single-objective `GeneticProgramming` runs (on `ExprGenome` or
`TreeGenome`) can checkpoint to disk and resume:

```julia
result = solve(problem, algorithm;
    checkpoint_every = 10,                 # generations between writes
    checkpoint_path  = "run.checkpoint",
)

# Later, in a separate process:
result = solve(problem, algorithm;
    resume_from = "run.checkpoint",
)
```

`save_checkpoint` writes atomically (write-tmp + rename) and
`load_checkpoint` rejects mismatched format-version or Julia-version
stamps. By default, resuming refuses to load a checkpoint produced
by a different `_algorithm_signature` (different hyperparameters or
operator types); pass `allow_signature_mismatch=true` to bypass the
check. `IslandModel`, `NSGAII`, `MAPElites`, `AntGenome`, and
`GraphGenome` are not yet wired for checkpoint/resume.

## Structured run logs

For research-grade reproducibility every single-objective `solve` path
accepts an optional `RunLog`:

```julia
log = RunLog()
result = solve(problem, algorithm; log = log)

# log.entries :: Vector{GenerationLog}
# Each entry carries best/mean/median/worst fitness, species count and
# sizes, unique-structure hash diversity, per-operator attempt and
# success tallies, and wall time for that generation.
```

The Plots.jl recipe renders three stacked subplots — fitness
trajectory, species count, and unique-structure count — directly
from the log:

```julia
using Plots
plot(log)
```
