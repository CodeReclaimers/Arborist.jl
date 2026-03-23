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

Fitness sharing divides each individual's raw fitness by its species size,
applying selective pressure within species (standard NEAT formula).

# Fields
- `threshold::Float64`: compatibility distance cutoff (default: 10.0)
- `min_species_size::Int`: minimum species size to avoid culling (default: 2)
- `stagnation_limit::Int`: generations without improvement before culling (default: 15)
"""
struct ThresholdSpeciation <: AbstractSpeciation
    threshold::Float64
    min_species_size::Int
    stagnation_limit::Int
end

"""
    ThresholdSpeciation(; threshold=10.0, min_species_size=2, stagnation_limit=15)

Construct a `ThresholdSpeciation` with keyword arguments and sensible defaults.
"""
function ThresholdSpeciation(; threshold::Float64=10.0, min_species_size::Int=2,
                              stagnation_limit::Int=15)
    ThresholdSpeciation(threshold, min_species_size, stagnation_limit)
end

# --- Internal species tracking ---

"""
Internal struct tracking per-species state across generations.
"""
mutable struct _SpeciesInfo
    representative::Any        # genome (kept as Any to avoid parametric complexity)
    best_fitness::Float64
    stagnation::Int
end

# --- Speciation dispatch ---

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
    _apply_speciation!(genomes, fitnesses, ::NoSpeciation, ::Nothing, rng)

No-op for NoSpeciation: return raw fitnesses unchanged.
"""
_apply_speciation!(genomes, fitnesses, ::NoSpeciation, ::Nothing, rng) = fitnesses

"""
    _apply_speciation!(genomes, fitnesses, spec::ThresholdSpeciation, species_list, rng)

Assign individuals to species, track stagnation, cull stagnant species,
and return shared fitnesses for selection.

Algorithm:
1. Assign each individual (sorted by fitness) to the first species whose
   representative has distance <= threshold, or create a new species.
2. Update stagnation counters and representatives.
3. Cull species that are stagnant AND below min_species_size, except the
   species containing the global best individual.
4. Redistribute orphaned individuals to the nearest remaining species.
5. Compute shared fitnesses: raw_fitness / species_size.
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
            # Create new species with this individual as representative.
            push!(species_list, _SpeciesInfo(deepcopy(genomes[i]), fitnesses[i], 0))
            push!(member_lists, [i])
            assignments[i] = length(species_list)
        end
    end

    # Update stagnation counters and representatives.
    for si in 1:length(species_list)
        members = member_lists[si]
        if isempty(members)
            species_list[si].stagnation += 1
            continue
        end
        species_best = minimum(fitnesses[j] for j in members)
        if species_list[si].best_fitness - species_best >= 1e-6
            # Improvement detected.
            species_list[si].best_fitness = species_best
            species_list[si].stagnation = 0
        else
            species_list[si].stagnation += 1
        end
        # Update representative to best individual in this species.
        best_member = members[argmin([fitnesses[j] for j in members])]
        species_list[si].representative = deepcopy(genomes[best_member])
    end

    # Find global best individual's species (never cull it).
    global_best_species = assignments[1]  # genomes are sorted, index 1 is best

    # Identify species to cull: stagnant AND below min_species_size.
    to_cull = Int[]
    for si in 1:length(species_list)
        if si == global_best_species
            continue
        end
        if species_list[si].stagnation > spec.stagnation_limit &&
           length(member_lists[si]) < spec.min_species_size
            push!(to_cull, si)
        end
    end

    # Cull and redistribute orphans.
    if !isempty(to_cull)
        orphans = Int[]
        for si in to_cull
            append!(orphans, member_lists[si])
        end

        # Build list of remaining species indices.
        remaining = [si for si in 1:length(species_list) if si ∉ to_cull]

        # Redistribute orphans to nearest remaining species.
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

        # Remove culled species (reverse order to preserve indices).
        sort!(to_cull, rev=true)
        for si in to_cull
            deleteat!(species_list, si)
            deleteat!(member_lists, si)
        end

        # Remap assignments to new species indices.
        old_to_new = Dict{Int,Int}()
        new_remaining = [si for si in 1:(length(species_list) + length(to_cull)) if si ∉ sort(to_cull)]
        for (new_si, old_si) in enumerate(new_remaining)
            old_to_new[old_si] = new_si
        end
        for i in 1:n
            if haskey(old_to_new, assignments[i])
                assignments[i] = old_to_new[assignments[i]]
            end
        end
    end

    # Remove empty species (no members assigned this generation).
    empty_species = [si for si in 1:length(species_list) if isempty(member_lists[si])]
    if !isempty(empty_species)
        sort!(empty_species, rev=true)
        for si in empty_species
            deleteat!(species_list, si)
            deleteat!(member_lists, si)
        end
    end

    # Compute shared fitnesses for minimization (lower = better).
    # Standard NEAT uses fi' = fi / |S| for maximization (higher = better).
    # For minimization, we MULTIPLY by species size so members of large
    # species appear worse (higher), protecting structural innovations
    # in small species.
    shared_fitnesses = copy(fitnesses)
    for (si, members) in enumerate(member_lists)
        species_size = length(members)
        if species_size > 1
            for j in members
                shared_fitnesses[j] = fitnesses[j] * species_size
            end
        end
    end

    return shared_fitnesses
end
