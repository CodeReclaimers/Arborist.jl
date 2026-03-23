"""
    GeneticProgramming <: AbstractEvolutionaryAlgorithm

Standard genetic programming algorithm configuration. All fields are set
via keyword arguments with sensible defaults.

# Fields
- `pop_size::Int`: population size (default: 100)
- `generations::Int`: number of generations to run (default: 200)
- `mutation_rate::Float64`: probability of mutation per offspring (default: 0.3)
- `crossover_rate::Float64`: probability of crossover per offspring pair (default: 0.3)
- `elitism::Int`: number of top individuals carried forward unchanged (default: 2)
- `tournament_size::Int`: number of individuals in tournament selection (default: 3)
- `max_depth::Int`: maximum depth of generated expression trees (default: 8)
- `bloat_penalty::Float64`: coefficient on `complexity(g)` added to fitness (default: 0.0)
- `parallel::Bool`: enable threaded population evaluation (default: true)
- `speciation::AbstractSpeciation`: speciation strategy (default: `NoSpeciation()`)
- `mutation_ops::Vector{AbstractMutationOperator}`: mutation operators
- `crossover_ops::Vector{AbstractCrossoverOperator}`: crossover operators
- `selection::AbstractSelectionStrategy`: selection strategy
"""
struct GeneticProgramming <: AbstractEvolutionaryAlgorithm
    pop_size::Int
    generations::Int
    mutation_rate::Float64
    crossover_rate::Float64
    elitism::Int
    tournament_size::Int
    max_depth::Int
    bloat_penalty::Float64
    parallel::Bool
    speciation::AbstractSpeciation
    mutation_ops::Vector{AbstractMutationOperator}
    crossover_ops::Vector{AbstractCrossoverOperator}
    selection::AbstractSelectionStrategy
end

"""
    GeneticProgramming(; kwargs...) -> GeneticProgramming

Construct a `GeneticProgramming` algorithm with keyword arguments and sensible defaults.
"""
function GeneticProgramming(;
    pop_size::Int = 100,
    generations::Int = 200,
    mutation_rate::Float64 = 0.3,
    crossover_rate::Float64 = 0.3,
    elitism::Int = 2,
    tournament_size::Int = 3,
    max_depth::Int = 8,
    bloat_penalty::Float64 = 0.0,
    parallel::Bool = true,
    speciation::AbstractSpeciation = NoSpeciation(),
    mutation_ops::Vector{<:AbstractMutationOperator} = AbstractMutationOperator[SubtreeMutation(), PointMutation()],
    crossover_ops::Vector{<:AbstractCrossoverOperator} = AbstractCrossoverOperator[SubtreeCrossover()],
    selection::AbstractSelectionStrategy = TournamentSelection(3)
)
    GeneticProgramming(
        pop_size, generations, mutation_rate, crossover_rate,
        elitism, tournament_size, max_depth, bloat_penalty,
        parallel, speciation,
        convert(Vector{AbstractMutationOperator}, mutation_ops),
        convert(Vector{AbstractCrossoverOperator}, crossover_ops),
        selection
    )
end

"""
    IslandModel <: AbstractEvolutionaryAlgorithm

Island model that runs multiple independent populations (islands) with periodic
migration. Migration uses a ring topology: island i sends its best individuals
to island i+1. Islands can run in parallel when `island_algorithm.parallel=true`
and multiple threads are available.

# Fields
- `n_islands::Int`: number of islands (default: 4)
- `island_algorithm::GeneticProgramming`: algorithm for each island
- `migration_interval::Int`: generations between migrations (default: 10)
- `migration_size::Int`: number of individuals to migrate per event (default: 2)
"""
struct IslandModel <: AbstractEvolutionaryAlgorithm
    n_islands::Int
    island_algorithm::GeneticProgramming
    migration_interval::Int
    migration_size::Int
end

"""
    IslandModel(; n_islands=4, island_algorithm=GeneticProgramming(),
                  migration_interval=10, migration_size=2)

Construct an `IslandModel` with keyword arguments and sensible defaults.
"""
function IslandModel(;
    n_islands::Int = 4,
    island_algorithm::GeneticProgramming = GeneticProgramming(),
    migration_interval::Int = 10,
    migration_size::Int = 2
)
    IslandModel(n_islands, island_algorithm, migration_interval, migration_size)
end
