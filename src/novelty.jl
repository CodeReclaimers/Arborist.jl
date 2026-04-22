# novelty.jl — Novelty Search (Lehman & Stanley 2008/2011) + behavioral archive.
#
# Novelty Search replaces fitness-based selection with behavior-based selection.
# Instead of selecting individuals that score well on the objective, it selects
# individuals whose *behavior* differs from previously-seen behaviors. Empirically
# escapes deceptive local optima in domains where the fitness gradient misleads
# (mazes, robot locomotion, modal control tasks).
#
# Design (per Phase F D4):
# - `NoveltyArchive{B}` is a thread-safe collection of behavioral fingerprints.
#   Mutated under a ReentrantLock during evaluation so `parallel=true` is safe.
# - `NoveltySearchEvaluator{F,D,B}` wraps a fingerprint function and distance
#   function. It is an `AbstractEvaluator` so the existing `solve(GP, ...)`
#   path works unchanged — selection just happens to be on novelty rather than
#   problem fitness.
# - The score returned by `evaluate_genome` is `-mean_knn_distance` so that the
#   framework's lower-is-better convention picks more novel individuals first.

"""
    NoveltyArchive{B}

A thread-safe, append-only collection of behavioral fingerprints used by
`NoveltySearchEvaluator` for k-nearest-neighbor novelty scoring.

# Fields
- `entries::Vector{B}`: stored fingerprints, in insertion order.
- `max_size::Int`: cap on the number of entries. When the archive is full,
  insertion is silently dropped (oldest-out eviction would change novelty
  scores for already-evaluated genomes; bounded growth is the standard
  Lehman-Stanley behavior).
- `add_threshold::Float64`: minimum novelty (mean k-NN distance) at which a
  new fingerprint is added. Setting to `0.0` means add every evaluated
  fingerprint (saturates fast); setting too high prevents archive growth
  and starves later evaluations of references.
- `lock::ReentrantLock`: guards `entries` for `parallel=true` evaluation.
"""
mutable struct NoveltyArchive{B}
    entries::Vector{B}
    max_size::Int
    add_threshold::Float64
    lock::ReentrantLock
end

function NoveltyArchive(::Type{B}; max_size::Int = 1000,
                        add_threshold::Float64 = 0.0) where B
    max_size >= 1 || throw(ArgumentError("max_size must be >= 1 (got $max_size)"))
    add_threshold >= 0.0 || throw(ArgumentError("add_threshold must be >= 0 (got $add_threshold)"))
    NoveltyArchive{B}(B[], max_size, add_threshold, ReentrantLock())
end

Base.length(a::NoveltyArchive) = length(a.entries)
Base.isempty(a::NoveltyArchive) = isempty(a.entries)

"""
    NoveltySearchEvaluator{F,D,B} <: AbstractEvaluator

Behavior-based evaluator. Returns the negative mean of the k nearest
neighbor distances between the current genome's fingerprint and the
archive — lower is better, matching the framework's convention.

# Type parameters
- `F`: type of the fingerprint function `genome -> B`.
- `D`: type of the distance function `(B, B) -> Float64`.
- `B`: type of a single behavioral fingerprint.

# Fields
- `fingerprint_fn::F`: extracts a behavioral descriptor from a genome.
  Typically performs the same rollout the base evaluator would, but
  records a behavior summary rather than a fitness scalar.
- `distance_fn::D`: distance metric over fingerprints. Should return 0.0
  for identical behaviors and positive for different ones.
- `archive::NoveltyArchive{B}`: behavioral memory.
- `k::Int`: number of nearest neighbors used in the novelty score.
"""
struct NoveltySearchEvaluator{F, D, B} <: AbstractEvaluator
    fingerprint_fn::F
    distance_fn::D
    archive::NoveltyArchive{B}
    k::Int
end

function NoveltySearchEvaluator(fingerprint_fn::F, distance_fn::D,
                                archive::NoveltyArchive{B};
                                k::Int = 15) where {F, D, B}
    k >= 1 || throw(ArgumentError("k must be >= 1 (got $k)"))
    NoveltySearchEvaluator{F, D, B}(fingerprint_fn, distance_fn, archive, k)
end

# AbstractEvaluator interface — opaque signatures since fingerprint can be anything.
input_signature(::NoveltySearchEvaluator) = Dict{Symbol, DataType}()
output_signature(::NoveltySearchEvaluator) = Dict{Symbol, DataType}()

# A scalar-fitness evaluate(e, f::Function) doesn't apply: novelty is genome-based.
# Solve paths use evaluate_genome(g, e); we rely on that.

"""
    evaluate_genome(g::AbstractGenome, e::NoveltySearchEvaluator) -> Float64

Compute the novelty score: take the genome's behavioral fingerprint, find
the k nearest fingerprints in the archive, return the negative mean of
those distances. The fingerprint may be added to the archive (under lock)
when its novelty exceeds `archive.add_threshold` and the archive isn't full.

Returns `Inf` if `fingerprint_fn` raises. Returns `0.0` when the archive
is empty (first genome of the run) — there's nothing to be novel against
yet, but adding to the archive seeds it for subsequent calls.
"""
function evaluate_genome(g::AbstractGenome, e::NoveltySearchEvaluator)
    fp = try
        e.fingerprint_fn(g)
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end

    novelty = _novelty_score(fp, e.archive, e.k, e.distance_fn)
    _maybe_add_to_archive!(e.archive, fp, novelty)
    # Framework convention: lower = better, so negate.
    return -novelty
end

# Mean of the k smallest distances from `fp` to entries in `archive`. Reads
# entries under the archive's lock so a concurrent insert can't see a partial
# vector. Returns 0.0 when the archive is empty.
function _novelty_score(fp, archive::NoveltyArchive, k::Int, distance_fn)
    entries_snapshot = lock(archive.lock) do
        copy(archive.entries)
    end
    isempty(entries_snapshot) && return 0.0
    distances = Float64[distance_fn(fp, ref) for ref in entries_snapshot]
    sort!(distances)
    upto = min(k, length(distances))
    return sum(@view distances[1:upto]) / upto
end

function _maybe_add_to_archive!(archive::NoveltyArchive{B}, fp::B,
                                 score::Float64) where B
    score >= archive.add_threshold || return false
    lock(archive.lock) do
        if length(archive.entries) < archive.max_size
            push!(archive.entries, fp)
            return true
        end
        return false
    end
end
