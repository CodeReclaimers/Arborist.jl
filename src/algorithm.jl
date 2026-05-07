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
- `bloat_penalty::Float64`: coefficient on `complexity(g)` added to fitness (default: 0.0)
- `parallel::Bool`: enable threaded population evaluation (default: true)
- `speciation::AbstractSpeciation`: speciation strategy (default: `NoSpeciation()`)
- `mutation_ops::Vector{AbstractMutationOperator}`: mutation operators
- `crossover_ops::Vector{AbstractCrossoverOperator}`: crossover operators
- `selection::AbstractSelectionStrategy`: selection strategy
- `convergence_threshold::Float64`: best fitness below this value sets `converged=true` in result (default: `Inf`, meaning never converged)
- `constant_sampler::Union{Function, Nothing}`: callable `(rng) -> T` used by `TreeGenome` when a new numeric-literal leaf is created (Koza-style Ephemeral Random Constants). `nothing` preserves the default `T(randn(rng))`. See `erc_uniform` for a helper that builds the canonical uniform-over-range sampler.
"""
struct GeneticProgramming <: AbstractEvolutionaryAlgorithm
    pop_size::Int
    generations::Int
    mutation_rate::Float64
    crossover_rate::Float64
    elitism::Int
    bloat_penalty::Float64
    parallel::Bool
    speciation::AbstractSpeciation
    mutation_ops::Vector{AbstractMutationOperator}
    crossover_ops::Vector{AbstractCrossoverOperator}
    selection::AbstractSelectionStrategy
    convergence_threshold::Float64
    constant_optimization::Union{Nothing, ConstantOptimization}
    constant_sampler::Union{Function, Nothing}
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
    bloat_penalty::Float64 = 0.0,
    parallel::Bool = true,
    speciation::AbstractSpeciation = NoSpeciation(),
    mutation_ops::Vector{<:AbstractMutationOperator} = AbstractMutationOperator[SubtreeMutation(), PointMutation()],
    crossover_ops::Vector{<:AbstractCrossoverOperator} = AbstractCrossoverOperator[SubtreeCrossover()],
    selection::AbstractSelectionStrategy = TournamentSelection(3),
    convergence_threshold::Float64 = Inf,
    constant_optimization::Union{Nothing, ConstantOptimization} = nothing,
    constant_sampler::Union{Function, Nothing} = nothing,
)
    pop_size > 0 || throw(ArgumentError("pop_size must be > 0, got $pop_size"))
    generations >= 0 || throw(ArgumentError("generations must be >= 0, got $generations"))
    elitism >= 0 || throw(ArgumentError("elitism must be >= 0, got $elitism"))
    elitism <= pop_size || throw(ArgumentError(
        "elitism ($elitism) must be <= pop_size ($pop_size)"))
    isfinite(mutation_rate) || throw(ArgumentError("mutation_rate must be finite, got $mutation_rate"))
    isfinite(crossover_rate) || throw(ArgumentError("crossover_rate must be finite, got $crossover_rate"))
    0.0 <= mutation_rate <= 1.0 || throw(ArgumentError(
        "mutation_rate must be in [0, 1], got $mutation_rate"))
    0.0 <= crossover_rate <= 1.0 || throw(ArgumentError(
        "crossover_rate must be in [0, 1], got $crossover_rate"))
    bloat_penalty >= 0.0 || throw(ArgumentError(
        "bloat_penalty must be >= 0, got $bloat_penalty"))
    if crossover_rate + mutation_rate > 1.0
        throw(ArgumentError(
            "crossover_rate ($crossover_rate) + mutation_rate ($mutation_rate) = " *
            "$(crossover_rate + mutation_rate) exceeds 1.0. " *
            "These rates partition [0, 1): crossover, mutation, and reproduction (the remainder)."))
    end
    GeneticProgramming(
        pop_size, generations, mutation_rate, crossover_rate,
        elitism, bloat_penalty,
        parallel, speciation,
        convert(Vector{AbstractMutationOperator}, mutation_ops),
        convert(Vector{AbstractCrossoverOperator}, crossover_ops),
        selection, convergence_threshold,
        constant_optimization,
        constant_sampler,
    )
end

"""
    IslandModel <: AbstractEvolutionaryAlgorithm

Island model that runs multiple independent populations (islands) with periodic
migration. Supports pluggable topologies and optional distributed execution
where each island runs in its own worker process.

# Fields
- `n_islands::Int`: number of islands (default: 4)
- `island_algorithm::GeneticProgramming`: algorithm for each island
- `migration_interval::Int`: generations between migrations (default: 10)
- `migration_size::Int`: number of individuals to migrate per event (default: 2)
- `topology::AbstractTopology`: migration topology (default: `RingTopology()`)
- `distributed::Bool`: run each island in a separate worker process (default: false)
- `async::Bool`: asynchronous evolution — islands evolve independently (default: false, requires distributed=true)
"""
struct IslandModel <: AbstractEvolutionaryAlgorithm
    n_islands::Int
    island_algorithm::GeneticProgramming
    migration_interval::Int
    migration_size::Int
    topology::AbstractTopology
    distributed::Bool
    async::Bool
end

"""
    IslandModel(; n_islands=4, island_algorithm=GeneticProgramming(),
                  migration_interval=10, migration_size=2,
                  topology=RingTopology(), distributed=false, async=false)

Construct an `IslandModel` with keyword arguments and sensible defaults.
When `distributed=true`, each island runs in its own worker process via
`Distributed.jl`, eliminating `@eval` contention between islands.
"""
function IslandModel(;
    n_islands::Int = 4,
    island_algorithm::GeneticProgramming = GeneticProgramming(),
    migration_interval::Int = 10,
    migration_size::Int = 2,
    topology::AbstractTopology = RingTopology(),
    distributed::Bool = false,
    async::Bool = false
)
    n_islands > 0 || throw(ArgumentError("n_islands must be > 0, got $n_islands"))
    migration_interval > 0 || throw(ArgumentError(
        "migration_interval must be > 0, got $migration_interval"))
    migration_size >= 0 || throw(ArgumentError(
        "migration_size must be >= 0, got $migration_size"))
    if async && !distributed
        throw(ArgumentError("async=true requires distributed=true"))
    end
    IslandModel(n_islands, island_algorithm, migration_interval, migration_size,
                topology, distributed, async)
end
