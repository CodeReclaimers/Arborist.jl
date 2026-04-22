# run_log.jl — structured per-generation metrics stream.
#
# Callers opt in by passing a `RunLog` to the `log` kwarg of `solve`. When
# non-nothing, each generation populates one `GenerationLog` entry with
# aggregate fitness stats, speciation snapshot (if speciation is active),
# structural diversity, and wall-time. When nothing (the default), solve
# incurs zero overhead.
#
# Per-operator success/attempted counters are initialized as empty Dicts;
# Phase F.5 will populate them via operator-layer instrumentation.

"""
    GenerationLog

One record per generation in a `RunLog`. All fields are populated by
`record!`. Fields intended for population in a later phase (Phase F.5)
are left as empty containers by F.0.

# Fields
- `generation::Int`: 1-based generation number.
- `best_fitness::Float64`: minimum finite fitness in the generation.
  `Inf` if no individual evaluated successfully.
- `mean_fitness::Float64`: mean of finite fitnesses.
- `median_fitness::Float64`: median of finite fitnesses.
- `worst_fitness::Float64`: maximum finite fitness. `Inf` if no finite
  fitnesses at all.
- `n_species::Int`: number of active species. `1` when speciation is
  `NoSpeciation` or speciation state is unavailable.
- `species_sizes::Vector{Int}`: member counts per species, in the same
  order as the speciation state's internal list. Empty when speciation is
  inactive or the solve path does not thread a `SpeciationSnapshot`.
- `operator_success::Dict{Symbol,Int}`: count of offspring per mutation/
  crossover operator that beat the parent's fitness. Populated in F.5.
- `operator_attempted::Dict{Symbol,Int}`: count of times each operator was
  invoked. Populated in F.5.
- `unique_structures::Int`: number of distinct genomes in the generation
  by `serialize`-hash. Coarse genotypic diversity proxy.
- `wall_time::Float64`: cumulative wall-clock seconds elapsed since
  `t0` (run start).
"""
mutable struct GenerationLog
    generation::Int
    best_fitness::Float64
    mean_fitness::Float64
    median_fitness::Float64
    worst_fitness::Float64
    n_species::Int
    species_sizes::Vector{Int}
    operator_success::Dict{Symbol,Int}
    operator_attempted::Dict{Symbol,Int}
    unique_structures::Int
    wall_time::Float64
end

"""
    RunLog

A vector-like container of `GenerationLog` entries. Callers construct one
as `RunLog()` and pass it to `solve(...; log=log)`. Iteration and indexed
access are supported via `entries(log)`, `length(log)`, and `log[i]`.

`RunLog` is mutable: `record!` appends one entry per generation.
"""
struct RunLog
    entries::Vector{GenerationLog}
end

RunLog() = RunLog(GenerationLog[])

"""
    entries(log::RunLog) -> Vector{GenerationLog}

Return the vector of `GenerationLog` entries recorded so far.
"""
entries(log::RunLog) = log.entries

Base.length(log::RunLog) = length(log.entries)
Base.getindex(log::RunLog, i::Integer) = log.entries[i]
Base.lastindex(log::RunLog) = lastindex(log.entries)
Base.iterate(log::RunLog, state...) = iterate(log.entries, state...)
Base.isempty(log::RunLog) = isempty(log.entries)

"""
    SpeciationSnapshot

Mutable carrier passed as a kwarg to `_apply_speciation!`. Populated with
the post-culling species count and per-species member sizes so that
`record!` can record them without the solve path re-computing speciation.

Constructed fresh per generation and discarded; not part of the public API.
"""
mutable struct SpeciationSnapshot
    n_species::Int
    sizes::Vector{Int}
end

SpeciationSnapshot() = SpeciationSnapshot(0, Int[])

# ---------------------------------------------------------------------------
# record! — append one entry to a RunLog.
# ---------------------------------------------------------------------------

"""
    record!(log::RunLog, gen, fitnesses, genomes, wall_time;
            snapshot=nothing)

Append one `GenerationLog` to `log` with aggregate fitness statistics,
optional speciation snapshot, structural diversity, and wall-clock time.

- `gen::Integer`: 1-based generation index.
- `fitnesses::AbstractVector`: raw fitness per individual (may contain
  `Inf` for failed evaluations).
- `genomes::AbstractVector`: population genomes, parallel to `fitnesses`.
- `wall_time::Real`: seconds since run start.
- `snapshot::Union{Nothing, SpeciationSnapshot}`: if provided, its
  `n_species` / `sizes` fields are copied into the entry. If `nothing`,
  the entry records `n_species=1` and `species_sizes=[length(genomes)]`
  (the NoSpeciation case).
"""
function record!(log::RunLog, gen::Integer,
                 fitnesses::AbstractVector, genomes::AbstractVector,
                 wall_time::Real;
                 snapshot::Union{Nothing, SpeciationSnapshot} = nothing)
    finite = Float64[f for f in fitnesses if isfinite(f)]
    if isempty(finite)
        best = Inf
        mean_f = Inf
        median_f = Inf
        worst = Inf
    else
        best = minimum(finite)
        mean_f = sum(finite) / length(finite)
        median_f = _median_sorted(sort(finite))
        worst = maximum(finite)
    end

    n_species, sizes = if snapshot === nothing
        (1, [length(genomes)])
    else
        (snapshot.n_species, copy(snapshot.sizes))
    end

    unique_n = _count_unique_structures(genomes)

    push!(log.entries, GenerationLog(
        Int(gen), best, mean_f, median_f, worst,
        n_species, sizes,
        Dict{Symbol,Int}(), Dict{Symbol,Int}(),
        unique_n, Float64(wall_time),
    ))
    return log
end

# In-tree median to avoid a Statistics.jl dependency (see Phase E housekeeping).
function _median_sorted(sorted::Vector{Float64})
    n = length(sorted)
    n == 0 && return Inf
    isodd(n) ? sorted[(n + 1) ÷ 2] :
               0.5 * (sorted[n ÷ 2] + sorted[n ÷ 2 + 1])
end

function _count_unique_structures(genomes::AbstractVector)
    seen = Set{UInt}()
    for g in genomes
        h = try
            hash(serialize(g))
        catch err
            err isa InterruptException && rethrow()
            hash(objectid(g))
        end
        push!(seen, h)
    end
    return length(seen)
end
