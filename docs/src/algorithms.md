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
