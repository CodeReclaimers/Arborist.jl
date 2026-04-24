# Sequence / memory benchmark generators.
#
# These problems require recurrence (or some form of state) to solve;
# purely feedforward networks can't store information across time steps.

"""
    sequence_memory(; length::Int=10, T=Float64) -> NamedTuple

Temporal recall: at each time step the network sees a single bit; after
a total of `length` time steps the network must output the *first* bit
it saw. Requires memory of at least one time step, growing with
`length`.

Returns the raw per-step observation sequences as a matrix. Users
typically feed these through a `GraphEvaluator` with
`allow_recurrent=true`.

Returns `(; sequences, targets, length, name, target_expr)` where
`sequences::Vector{Vector{T}}` each contain `length` bits, and
`targets::Vector{T}` is each sequence's first bit.
"""
function sequence_memory(; length::Int=10, T::Type=Float64)
    length >= 2 || throw(ArgumentError(
        "sequence_memory: length must be >= 2, got $length"))
    # 2^length patterns; cap at 32 sequences for small test sets.
    n_patterns = min(2^length, 32)
    sequences = Vector{Vector{T}}(undef, n_patterns)
    targets = Vector{T}(undef, n_patterns)
    # Deterministic patterns: first `n_patterns` integers.
    for p in 0:(n_patterns - 1)
        seq = Vector{T}(undef, length)
        for i in 0:(length - 1)
            seq[i + 1] = T((p >> i) & 1)
        end
        sequences[p + 1] = seq
        targets[p + 1] = seq[1]  # first-bit target
    end
    return (; sequences=sequences, targets=targets, length=length,
              name="Sequence-Memory-$length",
              target_expr="output bit 1 after $(length) time steps")
end

"""
    sequence_recall(; delay::Int=5, T=Float64) -> NamedTuple

Variable-delay sequence recall: at step 0 the network sees the cue bit;
at step `delay` the network must output that cue. Other steps contain
noise bits. Tests genuine long-range memory (not just last-step
association).

Returns `(; sequences, targets, delay, name, target_expr)`.
"""
function sequence_recall(; delay::Int=5, T::Type=Float64)
    delay >= 1 || throw(ArgumentError(
        "sequence_recall: delay must be >= 1, got $delay"))
    seq_length = delay + 1
    n_patterns = 16
    rng = MersenneTwister(2026)
    sequences = Vector{Vector{T}}(undef, n_patterns)
    targets = Vector{T}(undef, n_patterns)
    for p in 1:n_patterns
        seq = T.(rand(rng, 0:1, seq_length))
        cue = seq[1]
        sequences[p] = seq
        targets[p] = cue
    end
    return (; sequences=sequences, targets=targets, delay=delay,
              name="Sequence-Recall-delay-$delay",
              target_expr="output the cue bit after $(delay) noise steps")
end
