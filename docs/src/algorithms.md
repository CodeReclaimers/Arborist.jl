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

See `examples/nsga2_regression.jl` for a complete runnable example.
