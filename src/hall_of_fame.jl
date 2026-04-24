"""
    HallOfFame{G<:AbstractGenome}

Bounded top-K archive of the best genomes a `solve()` has ever seen,
across all generations. Maintained in ascending fitness order (best
first). Opt-in via the `hall_of_fame_size::Int` kwarg on `solve()`
— `size == 0` is the default and produces `nothing` in
`GPResult.hall_of_fame`.

# Fields
- `capacity::Int`: maximum number of distinct entries retained
- `genomes::Vector{G}`: genome list, best-first
- `fitnesses::Vector{Float64}`: matching fitness list

# Dedup

`push!(hof, g, f)` treats two fitnesses as duplicates when they are
within `1e-12` of each other. This cheap filter catches structurally-
equivalent solutions (identical fitness) without the cost of walking
genomes for structural equality. It will occasionally merge two
semantically-distinct genomes that happen to produce the same exact
fitness; acceptable for a Hall-of-Fame, which is best-effort rather
than canonical.
"""
mutable struct HallOfFame{G<:AbstractGenome}
    capacity::Int
    genomes::Vector{G}
    fitnesses::Vector{Float64}
end

"""
    HallOfFame{G}(capacity::Int) -> HallOfFame{G}
    HallOfFame{G}(; capacity::Int=20) -> HallOfFame{G}
"""
function HallOfFame{G}(capacity::Int) where {G<:AbstractGenome}
    capacity >= 1 || throw(ArgumentError(
        "HallOfFame: capacity must be >= 1, got $capacity"))
    return HallOfFame{G}(capacity, G[], Float64[])
end

HallOfFame{G}(; capacity::Int=20) where {G<:AbstractGenome} = HallOfFame{G}(capacity)

Base.length(hof::HallOfFame) = length(hof.genomes)
Base.isempty(hof::HallOfFame) = isempty(hof.genomes)
Base.getindex(hof::HallOfFame, i::Int) = hof.genomes[i]
Base.iterate(hof::HallOfFame, state=1) =
    state > length(hof.genomes) ? nothing : (hof.genomes[state], state + 1)

"""
    push!(hof::HallOfFame{G}, genome::G, fitness::Real)

Insert `(genome, fitness)` into the hall if it qualifies. Non-finite
fitnesses are rejected. Duplicates (fitness within `1e-12`) are
rejected. When the archive is at capacity, the worst-fitness entry
is evicted if the candidate is strictly better.
"""
function Base.push!(hof::HallOfFame{G}, genome::G, fitness::Real) where {G}
    f = Float64(fitness)
    isfinite(f) || return hof

    # Dedup: any existing entry with |f_existing - f| < tol → skip.
    tol = 1e-12
    for existing in hof.fitnesses
        if abs(existing - f) < tol
            return hof
        end
    end

    # Find insertion point (sorted ascending).
    pos = searchsortedfirst(hof.fitnesses, f)

    if length(hof) < hof.capacity
        insert!(hof.genomes, pos, deepcopy(genome))
        insert!(hof.fitnesses, pos, f)
    else
        # Full: only insert if better than current worst (last entry).
        f < hof.fitnesses[end] || return hof
        # Drop worst, then insert.
        pop!(hof.genomes)
        pop!(hof.fitnesses)
        insert!(hof.genomes, pos, deepcopy(genome))
        insert!(hof.fitnesses, pos, f)
    end
    return hof
end

"""
    fitnesses(hof::HallOfFame) -> Vector{Float64}

Return the archive's fitness list in order (best first). Alias for
`hof.fitnesses` — kept as a function for API stability if the internal
representation changes.
"""
fitnesses(hof::HallOfFame) = copy(hof.fitnesses)

# --- Display ---------------------------------------------------------------

function Base.show(io::IO, hof::HallOfFame{G}) where {G}
    print(io, "HallOfFame{", G, "}(", length(hof), "/", hof.capacity, ")")
end

function Base.show(io::IO, ::MIME"text/plain", hof::HallOfFame{G}) where {G}
    println(io, "HallOfFame{", G, "} — top-", hof.capacity,
               " archive (", length(hof), " filled)")
    n_shown = min(5, length(hof))
    for i in 1:n_shown
        println(io, "  [", i, "] fitness=", _fmt_fitness(hof.fitnesses[i]),
                    "  ", sprint(show, hof.genomes[i]))
    end
    if length(hof) > n_shown
        print(io, "  … (", length(hof) - n_shown, " more)")
    end
end
