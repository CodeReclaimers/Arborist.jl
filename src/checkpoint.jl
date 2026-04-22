# checkpoint.jl — harness-level save/load of mid-run evolution state.
#
# Format: Julia's `Serialization` stdlib. No new dependencies. The known
# caveat is that checkpoints are not portable across Julia minor versions —
# a stamp is written into the checkpoint and `load_checkpoint` refuses to
# load a file whose Julia version differs, giving the user a clear error
# rather than a silent type-deserialization failure.
#
# Atomicity: saves write to `path * ".tmp"` then `mv(...; force=true)`. A
# crash mid-save leaves either the previous checkpoint or the incoming one,
# never a partial file.

import Serialization

const CHECKPOINT_FORMAT_VERSION = 1

"""
    Checkpoint{G}

Opaque snapshot of a mid-run single-objective GP evolution. Holds everything
`_run_evolution!` needs to resume: population, fitnesses, generation counter,
RNG state, fitness/mean histories, wall-time, and a signature derived from
the algorithm config.

Not meant for direct construction — `save_checkpoint` is invoked internally
by `solve(...; checkpoint_every, checkpoint_path)`. Users construct one only
when implementing a custom solve loop.

# Fields
- `format_version::Int`: internal version of the `Checkpoint` layout
  (bumped on incompatible field changes).
- `arborist_version::VersionNumber`: project version from Project.toml.
- `julia_version::VersionNumber`: `VERSION` at save time.
- `generation::Int`: generation just completed. Resume starts at `generation + 1`.
- `population::Vector{G}`: final population of the completed generation,
  sorted best-first.
- `fitnesses::Vector{Float64}`: aligned with `population`.
- `rng_state::Any`: `copy(rng)` at save time, so resumed runs draw the same
  random stream the interrupted one would have.
- `best_genome::G`: the single best across the whole run so far (may be
  different from `population[1]` if elitism lost it through breeding).
- `best_fitness::Float64`: paired with `best_genome`.
- `fitness_history::Vector{Float64}`: per-generation best-fitness trajectory.
- `mean_history::Vector{Float64}`: per-generation mean-finite-fitness trajectory.
- `wall_time::Float64`: cumulative seconds elapsed (does not include
  pre-resume idle time).
- `algorithm_signature::UInt64`: hash of the algorithm config (see
  `_algorithm_signature`) — checked on resume so the user can't hot-swap
  hyperparameters silently.
"""
struct Checkpoint{G}
    format_version::Int
    arborist_version::VersionNumber
    julia_version::VersionNumber
    generation::Int
    population::Vector{G}
    fitnesses::Vector{Float64}
    rng_state::Any
    best_genome::G
    best_fitness::Float64
    fitness_history::Vector{Float64}
    mean_history::Vector{Float64}
    wall_time::Float64
    algorithm_signature::UInt64
end

"""
    save_checkpoint(ckpt::Checkpoint, path::AbstractString)

Atomically write `ckpt` to `path`. Uses Julia's `Serialization` stdlib.
Writes to `path * ".tmp"` then renames, so a partial file never clobbers
an older good checkpoint.
"""
function save_checkpoint(ckpt::Checkpoint, path::AbstractString)
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        Serialization.serialize(io, ckpt)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    load_checkpoint(path::AbstractString) -> Checkpoint

Load a checkpoint previously written by `save_checkpoint`. Raises
`ArgumentError` if the file's Julia version or checkpoint format version
differs from the current process — Julia's `Serialization` format is not
stable across minor versions.
"""
function load_checkpoint(path::AbstractString)
    isfile(path) || throw(ArgumentError("checkpoint file not found: $path"))
    ckpt = open(path, "r") do io
        Serialization.deserialize(io)
    end
    ckpt isa Checkpoint || throw(ArgumentError(
        "file $path did not deserialize to a Checkpoint (got $(typeof(ckpt)))"))
    ckpt.format_version == CHECKPOINT_FORMAT_VERSION || throw(ArgumentError(
        "checkpoint format version mismatch: file has v$(ckpt.format_version), " *
        "runtime expects v$CHECKPOINT_FORMAT_VERSION"))
    ckpt.julia_version == VERSION || throw(ArgumentError(
        "checkpoint Julia version mismatch: file was saved under $(ckpt.julia_version), " *
        "current Julia is $VERSION. Serialization format is not stable across minor versions."))
    return ckpt
end

# Internal: derive a stable 64-bit signature for an algorithm config so
# resume can detect mismatched hyperparameters. We avoid `hash(alg)` because
# some fields hold closures (LLM operator `_http_post` hook, custom fingerprint
# functions) whose hash depends on object identity rather than contents.
function _algorithm_signature(alg::GeneticProgramming)
    h = hash((alg.pop_size, alg.generations,
              alg.mutation_rate, alg.crossover_rate,
              alg.elitism, alg.bloat_penalty,
              alg.parallel,
              length(alg.mutation_ops),
              length(alg.crossover_ops),
              typeof(alg.selection),
              typeof(alg.speciation),
              alg.convergence_threshold))
    # Include operator-type identities so adding/removing an operator changes the signature.
    for op in alg.mutation_ops
        h = hash(typeof(op), h)
    end
    for op in alg.crossover_ops
        h = hash(typeof(op), h)
    end
    return h
end
