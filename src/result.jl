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
- `hall_of_fame::Union{Nothing, HallOfFame{G}}`: top-K archive across all
  generations when `solve(... ; hall_of_fame_size=K)` was passed with
  `K > 0`. `nothing` otherwise.
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
    hall_of_fame::Union{Nothing, Any}  # HallOfFame{G} when populated; loose
                                       # type here because HallOfFame is
                                       # defined in a file loaded after result.jl.
end

# Backward-compatible constructor: older call sites that positionally passed
# 8 args (no hall_of_fame) continue to work with a nil archive.
function GPResult{G}(best_genome::G, best_fitness::Float64,
                     population::Vector{G},
                     fitness_history::Vector{Float64},
                     mean_history::Vector{Float64},
                     generations_run::Int, wall_time::Float64,
                     converged::Bool) where {G<:AbstractGenome}
    return GPResult{G}(best_genome, best_fitness, population,
                       fitness_history, mean_history,
                       generations_run, wall_time, converged, nothing)
end

_converged(best_fitness::Real, threshold::Real) =
    isfinite(threshold) && best_fitness < threshold

# --- Display ---------------------------------------------------------------

function Base.show(io::IO, r::GPResult{G}) where G
    print(io, "GPResult{", G, "}(gens=", r.generations_run,
              ", best=", _fmt_fitness(r.best_fitness), ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::GPResult{G}) where G
    println(io, "GPResult{", G, "}")
    println(io, "  generations run: ", r.generations_run)
    println(io, "  wall time:       ", _fmt_wall(r.wall_time))
    println(io, "  best fitness:    ", _fmt_fitness(r.best_fitness))
    println(io, "  converged:       ", r.converged)
    println(io, "  population size: ", length(r.population))
    print(io,   "  best genome:     ")
    show(io, r.best_genome)
end

"""
    _fmt_fitness(f) -> String

Format a fitness scalar with 4 significant digits for compact display.
`Inf`, `NaN`, and negative values are handled as-is. Used by the `Base.show`
methods on result and log types so every fitness readout is uniform.
"""
_fmt_fitness(f::Float64) = isfinite(f) ? string(round(f; sigdigits=4)) : string(f)
_fmt_fitness(f::Real) = _fmt_fitness(Float64(f))

"""
    _fmt_wall(seconds) -> String

Format a wall-clock duration with units chosen by magnitude: seconds (under
60), minutes:seconds otherwise. Used by result / log show methods.
"""
function _fmt_wall(s::Real)
    s < 60     && return string(round(s; digits=2), "s")
    m, r = divrem(s, 60)
    return string(Int(m), "m", round(r; digits=1), "s")
end
