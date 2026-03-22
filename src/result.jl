"""
    GPResult{G<:AbstractGenome} <: AbstractEvolutionResult

Result returned by `solve`. Contains the best genome found, fitness history,
and metadata about the evolutionary run.

# Fields
- `best_genome::G`: the genome with the best fitness found during the run
- `best_fitness::Float64`: fitness of the best genome (lower is better)
- `population::Vector{G}`: final population sorted by fitness
- `fitness_history::Vector{Float64}`: best fitness per generation
- `mean_history::Vector{Float64}`: mean finite fitness per generation
- `generations_run::Int`: number of generations completed
- `wall_time::Float64`: elapsed wall-clock time in seconds
- `converged::Bool`: whether the run met the convergence criterion
"""
mutable struct GPResult{G<:AbstractGenome} <: AbstractEvolutionResult
    best_genome::G
    best_fitness::Float64
    population::Vector{G}
    fitness_history::Vector{Float64}
    mean_history::Vector{Float64}
    generations_run::Int
    wall_time::Float64
    converged::Bool
end
