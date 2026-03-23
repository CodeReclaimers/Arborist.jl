"""
    NoSpeciation <: AbstractSpeciation

Trivial speciation strategy: all individuals belong to a single species.
"""
struct NoSpeciation <: AbstractSpeciation end

"""
    ThresholdSpeciation <: AbstractSpeciation

Species individuals by compatibility distance. Individuals are assigned to the
first species whose representative has `distance(g1, g2) <= threshold`. Tracks
stagnation per species and culls stagnant species below `min_species_size`.

The default threshold of 10.0 is appropriate for `ExprGenome` where `distance`
is a symmetric-difference node count.

Fitness sharing penalizes members of large species to protect structural
innovations in small species.

# Fields
- `threshold::Float64`: compatibility distance cutoff (default: 10.0)
- `min_species_size::Int`: minimum species size to avoid culling (default: 2)
- `stagnation_limit::Int`: generations without improvement before culling (default: 15)
- `sharing_formula::Symbol`: fitness sharing strength — `:linear`, `:sqrt`, `:log2`, or `:none` (default: `:log2`)
"""
struct ThresholdSpeciation <: AbstractSpeciation
    threshold::Float64
    min_species_size::Int
    stagnation_limit::Int
    sharing_formula::Symbol
end

"""
    ThresholdSpeciation(; threshold=10.0, min_species_size=2, stagnation_limit=15, sharing_formula=:log2)

Construct a `ThresholdSpeciation` with keyword arguments and sensible defaults.
"""
function ThresholdSpeciation(; threshold::Float64=10.0, min_species_size::Int=2,
                              stagnation_limit::Int=15, sharing_formula::Symbol=:log2)
    ThresholdSpeciation(threshold, min_species_size, stagnation_limit, sharing_formula)
end

"""
    BehavioralSpeciation <: AbstractSpeciation

Speciation based on behavioral similarity rather than genomic distance.
Two individuals are in the same species if their behavioral fingerprints
are within `threshold` of each other under the provided distance function.

This is useful for behavioral optimization problems (e.g., bin packing,
game playing) where multiple syntactically distinct programs can implement
the same strategy. Genomic distance treats them as different; behavioral
distance correctly collapses them.

# Fields
- `fingerprint_fn::Function`: `(genome) -> fingerprint` — computes a behavioral
  fingerprint. The return type can be anything that `distance_fn` can compare.
- `distance_fn::Function`: `(a, b) -> Float64` — behavioral distance between
  two fingerprints. Should return 0.0 for identical behavior.
- `threshold::Float64`: behavioral distance below which individuals are
  same-species (default: 0.15)
- `min_species_size::Int`: minimum species size before culling (default: 2)
- `stagnation_limit::Int`: generations without improvement before culling (default: 15)
- `sharing_formula::Symbol`: fitness sharing strength — `:linear`, `:sqrt`, `:log2`, or `:none` (default: `:sqrt`)
"""
struct BehavioralSpeciation <: AbstractSpeciation
    fingerprint_fn::Function
    distance_fn::Function
    threshold::Float64
    min_species_size::Int
    stagnation_limit::Int
    sharing_formula::Symbol
end

function BehavioralSpeciation(;
    fingerprint_fn::Function,
    distance_fn::Function,
    threshold::Float64 = 0.15,
    min_species_size::Int = 2,
    stagnation_limit::Int = 15,
    sharing_formula::Symbol = :sqrt
)
    BehavioralSpeciation(fingerprint_fn, distance_fn, threshold,
                          min_species_size, stagnation_limit, sharing_formula)
end

# =============================================================================
# Fitness sharing helper
# =============================================================================

"""
    apply_sharing(raw_fitness::Float64, species_size::Int, formula::Symbol) -> Float64

Apply fitness sharing to penalize members of large species (for minimization:
lower = better). A larger penalty means more diversity pressure.

Formula options:
- `:none`   — no sharing (raw fitness unchanged)
- `:linear` — multiply by species_size (strong; NEAT default for topology protection)
- `:sqrt`   — multiply by sqrt(species_size) (moderate; good for behavioral optimization)
- `:log2`   — multiply by (1 + log2(species_size)) (gentle; singletons unpenalized)
"""
function apply_sharing(raw_fitness::Float64, species_size::Int, formula::Symbol)::Float64
    species_size <= 1 && return raw_fitness
    formula == :none   && return raw_fitness
    formula == :linear && return raw_fitness * species_size
    formula == :sqrt   && return raw_fitness * sqrt(species_size)
    formula == :log2   && return raw_fitness * (1.0 + log2(species_size))
    throw(ArgumentError("Unknown sharing formula: $formula. Use :none, :linear, :sqrt, or :log2"))
end

# =============================================================================
# Internal species tracking
# =============================================================================

"""
Internal struct tracking per-species state across generations.
"""
mutable struct _SpeciesInfo
    representative::Any        # genome or fingerprint (kept as Any to avoid parametric complexity)
    best_fitness::Float64
    stagnation::Int
end

# =============================================================================
# Speciation dispatch — init
# =============================================================================

"""
    _init_species_state(::NoSpeciation)

Return `nothing` — no species tracking needed.
"""
_init_species_state(::NoSpeciation) = nothing

"""
    _init_species_state(::ThresholdSpeciation)

Return an empty species list for ThresholdSpeciation.
"""
_init_species_state(::ThresholdSpeciation) = _SpeciesInfo[]

"""
    _init_species_state(::BehavioralSpeciation)

Return an empty species list for BehavioralSpeciation.
"""
_init_species_state(::BehavioralSpeciation) = _SpeciesInfo[]

# =============================================================================
# Speciation dispatch — apply
# =============================================================================

"""
    _apply_speciation!(genomes, fitnesses, ::NoSpeciation, ::Nothing, rng)

No-op for NoSpeciation: return raw fitnesses unchanged.
"""
_apply_speciation!(genomes, fitnesses, ::NoSpeciation, ::Nothing, rng) = fitnesses

"""
    _apply_speciation!(genomes, fitnesses, spec::ThresholdSpeciation, species_list, rng)

Assign individuals to species by genomic distance, track stagnation,
cull stagnant species, and return shared fitnesses for selection.
"""
function _apply_speciation!(genomes::Vector{G}, fitnesses::Vector{Float64},
                            spec::ThresholdSpeciation,
                            species_list::Vector{_SpeciesInfo},
                            rng::AbstractRNG) where {G}
    n = length(genomes)

    # Build member lists for each existing species.
    n_species = length(species_list)
    member_lists = [Int[] for _ in 1:n_species]

    # Assign each individual to a species (individuals are already sorted by fitness).
    assignments = zeros(Int, n)
    for i in 1:n
        assigned = false
        for si in 1:length(species_list)
            if distance(genomes[i], species_list[si].representative) <= spec.threshold
                push!(member_lists[si], i)
                assignments[i] = si
                assigned = true
                break
            end
        end
        if !assigned
            push!(species_list, _SpeciesInfo(deepcopy(genomes[i]), fitnesses[i], 0))
            push!(member_lists, [i])
            assignments[i] = length(species_list)
        end
    end

    _update_stagnation!(species_list, member_lists, fitnesses, genomes)
    global_best_species = assignments[1]
    _cull_species!(species_list, member_lists, assignments, genomes, fitnesses,
                   spec.stagnation_limit, spec.min_species_size, global_best_species, n)
    _remove_empty_species!(species_list, member_lists)

    return _compute_shared_fitnesses(fitnesses, member_lists, spec.sharing_formula)
end

"""
    _apply_speciation!(genomes, fitnesses, spec::BehavioralSpeciation, species_list, rng)

Assign individuals to species by behavioral fingerprint distance, track
stagnation, cull stagnant species, and return shared fitnesses for selection.

Unlike ThresholdSpeciation, the representative stored is a *fingerprint*
(computed via `spec.fingerprint_fn`), not a genome. Distance is computed via
`spec.distance_fn` on fingerprints.
"""
function _apply_speciation!(genomes::Vector{G}, fitnesses::Vector{Float64},
                            spec::BehavioralSpeciation,
                            species_list::Vector{_SpeciesInfo},
                            rng::AbstractRNG) where {G}
    n = length(genomes)

    # Compute fingerprints for all individuals.
    fingerprints = Vector{Any}(undef, n)
    for i in 1:n
        fingerprints[i] = spec.fingerprint_fn(genomes[i])
    end

    # Build member lists for each existing species.
    n_species = length(species_list)
    member_lists = [Int[] for _ in 1:n_species]

    # Assign each individual to a species by behavioral distance.
    assignments = zeros(Int, n)
    for i in 1:n
        assigned = false
        for si in 1:length(species_list)
            if spec.distance_fn(fingerprints[i], species_list[si].representative) <= spec.threshold
                push!(member_lists[si], i)
                assignments[i] = si
                assigned = true
                break
            end
        end
        if !assigned
            push!(species_list, _SpeciesInfo(fingerprints[i], fitnesses[i], 0))
            push!(member_lists, [i])
            assignments[i] = length(species_list)
        end
    end

    # Update stagnation and representatives (using fingerprints, not genomes).
    for si in 1:length(species_list)
        members = member_lists[si]
        if isempty(members)
            species_list[si].stagnation += 1
            continue
        end
        species_best = minimum(fitnesses[j] for j in members)
        if species_list[si].best_fitness - species_best >= 1e-6
            species_list[si].best_fitness = species_best
            species_list[si].stagnation = 0
        else
            species_list[si].stagnation += 1
        end
        best_member = members[argmin([fitnesses[j] for j in members])]
        species_list[si].representative = fingerprints[best_member]
    end

    global_best_species = assignments[1]

    # Cull stagnant small species (redistribute orphans to nearest by behavioral distance).
    to_cull = Int[]
    for si in 1:length(species_list)
        si == global_best_species && continue
        if species_list[si].stagnation > spec.stagnation_limit &&
           length(member_lists[si]) < spec.min_species_size
            push!(to_cull, si)
        end
    end

    if !isempty(to_cull)
        orphans = Int[]
        for si in to_cull
            append!(orphans, member_lists[si])
        end
        remaining = [si for si in 1:length(species_list) if si ∉ to_cull]
        for oi in orphans
            best_si_idx = 1
            best_dist = Inf
            for (ri, si) in enumerate(remaining)
                d = spec.distance_fn(fingerprints[oi], species_list[si].representative)
                if d < best_dist
                    best_dist = d
                    best_si_idx = ri
                end
            end
            push!(member_lists[remaining[best_si_idx]], oi)
            assignments[oi] = remaining[best_si_idx]
        end
        sort!(to_cull, rev=true)
        for si in to_cull
            deleteat!(species_list, si)
            deleteat!(member_lists, si)
        end
    end

    _remove_empty_species!(species_list, member_lists)

    return _compute_shared_fitnesses(fitnesses, member_lists, spec.sharing_formula)
end

# =============================================================================
# Shared helpers for speciation implementations
# =============================================================================

"""Update stagnation counters and representatives for ThresholdSpeciation."""
function _update_stagnation!(species_list, member_lists, fitnesses, genomes)
    for si in 1:length(species_list)
        members = member_lists[si]
        if isempty(members)
            species_list[si].stagnation += 1
            continue
        end
        species_best = minimum(fitnesses[j] for j in members)
        if species_list[si].best_fitness - species_best >= 1e-6
            species_list[si].best_fitness = species_best
            species_list[si].stagnation = 0
        else
            species_list[si].stagnation += 1
        end
        best_member = members[argmin([fitnesses[j] for j in members])]
        species_list[si].representative = deepcopy(genomes[best_member])
    end
end

"""Cull stagnant small species and redistribute orphans (genomic distance)."""
function _cull_species!(species_list, member_lists, assignments, genomes, fitnesses,
                        stagnation_limit, min_species_size, global_best_species, n)
    to_cull = Int[]
    for si in 1:length(species_list)
        si == global_best_species && continue
        if species_list[si].stagnation > stagnation_limit &&
           length(member_lists[si]) < min_species_size
            push!(to_cull, si)
        end
    end
    isempty(to_cull) && return

    orphans = Int[]
    for si in to_cull
        append!(orphans, member_lists[si])
    end
    remaining = [si for si in 1:length(species_list) if si ∉ to_cull]
    for oi in orphans
        best_si_idx = 1
        best_dist = Inf
        for (ri, si) in enumerate(remaining)
            d = distance(genomes[oi], species_list[si].representative)
            if d < best_dist
                best_dist = d
                best_si_idx = ri
            end
        end
        push!(member_lists[remaining[best_si_idx]], oi)
        assignments[oi] = remaining[best_si_idx]
    end
    sort!(to_cull, rev=true)
    for si in to_cull
        deleteat!(species_list, si)
        deleteat!(member_lists, si)
    end
    # Note: assignment index remapping not needed — shared_fitnesses
    # are computed from member_lists, not assignments.
end

"""Remove species with no members assigned this generation."""
function _remove_empty_species!(species_list, member_lists)
    empty_species = [si for si in 1:length(species_list) if isempty(member_lists[si])]
    isempty(empty_species) && return
    sort!(empty_species, rev=true)
    for si in empty_species
        deleteat!(species_list, si)
        deleteat!(member_lists, si)
    end
end

"""Compute shared fitnesses from member lists using the specified sharing formula."""
function _compute_shared_fitnesses(fitnesses, member_lists, formula::Symbol)
    shared_fitnesses = copy(fitnesses)
    for (si, members) in enumerate(member_lists)
        species_size = length(members)
        for j in members
            shared_fitnesses[j] = apply_sharing(fitnesses[j], species_size, formula)
        end
    end
    return shared_fitnesses
end
