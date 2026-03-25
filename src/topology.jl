# topology.jl — Migration topology types for island models.

"""
    RingTopology <: AbstractTopology

Ring migration: island i sends to island (i % n) + 1.
This is the default and most common topology in the literature.
Sparse connectivity preserves diversity across islands.
"""
struct RingTopology <: AbstractTopology end

"""
    CompleteTopology <: AbstractTopology

All-to-all migration: island i sends to every other island.
Fast dissemination of good solutions but rapidly reduces diversity.
"""
struct CompleteTopology <: AbstractTopology end

"""
    RandomTopology <: AbstractTopology

Random migration: island i sends to `n_targets` randomly chosen islands.
Good pairing with asynchronous evolution where the randomness of timing
provides a natural diversity mechanism.

# Fields
- `n_targets::Int`: number of destination islands per migration event
"""
struct RandomTopology <: AbstractTopology
    n_targets::Int
end

"""
    migration_targets(topology, i, n_islands, rng) -> Vector{Int}

Return destination island indices for emigrants from island `i`.
The `rng` argument is required by the interface but ignored by
deterministic topologies (Ring, Complete).
"""
function migration_targets(::RingTopology, i::Int, n_islands::Int, ::AbstractRNG)
    return [(i % n_islands) + 1]
end

function migration_targets(::CompleteTopology, i::Int, n_islands::Int, ::AbstractRNG)
    return [j for j in 1:n_islands if j != i]
end

function migration_targets(t::RandomTopology, i::Int, n_islands::Int, rng::AbstractRNG)
    others = [j for j in 1:n_islands if j != i]
    k = min(t.n_targets, length(others))
    return others[randperm(rng, length(others))[1:k]]
end
