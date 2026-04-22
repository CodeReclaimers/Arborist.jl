"""
    TournamentSelection <: AbstractSelectionStrategy

Selection strategy that picks `tournament_size` individuals at random
and returns the one with the best (lowest) fitness.

# Fields
- `tournament_size::Int`: number of individuals competing in each tournament
"""
struct TournamentSelection <: AbstractSelectionStrategy
    tournament_size::Int
end

"""
    LexicaseSelection <: AbstractSelectionStrategy

Lexicase selection (Spector 2012; Helmuth, Spector, Matheson 2014):

1. All individuals start as candidates.
2. Shuffle test cases into a random order.
3. For each case in order, narrow candidates to those with the best (lowest)
   loss on that case. If only one remains, select it.
4. If a full pass exhausts cases with multiple candidates remaining, pick
   uniformly at random.

Lexicase preserves individuals that excel on some cases even if their
aggregate fitness is poor, making it effective on modal / deceptive
fitness landscapes where averaged objectives mask specialist solutions.

Requires the evaluator to implement `evaluate_cases`. The solve loop
materializes the per-individual per-case matrix automatically when
this strategy is passed to `GeneticProgramming(selection=...)`.
"""
struct LexicaseSelection <: AbstractSelectionStrategy end

"""
    EpsilonLexicaseSelection(; epsilon=0.0) <: AbstractSelectionStrategy

Epsilon-lexicase (La Cava, Spector, Danai 2016) — a relaxed form of
`LexicaseSelection` for continuous targets. At each case, individuals
are kept if their loss on that case is within `epsilon` of the best
candidate's loss.

- `epsilon > 0.0`: fixed scalar tolerance applied to every case.
- `epsilon == 0.0` (default): auto-epsilon — per-case tolerance
  computed as the median absolute deviation (MAD) of the case's
  population fitness distribution, following La Cava et al.

MAD-based auto-epsilon is the standard in the symbolic-regression
lexicase literature and avoids the manual-tuning pitfall that plagues
fixed-epsilon variants.
"""
struct EpsilonLexicaseSelection <: AbstractSelectionStrategy
    epsilon::Float64
end

EpsilonLexicaseSelection(; epsilon::Float64 = 0.0) = EpsilonLexicaseSelection(epsilon)

# ---------------------------------------------------------------------------
# needs_cases trait
# ---------------------------------------------------------------------------

needs_cases(::TournamentSelection) = false
needs_cases(::LexicaseSelection) = true
needs_cases(::EpsilonLexicaseSelection) = true

# ---------------------------------------------------------------------------
# select_parent dispatch
# ---------------------------------------------------------------------------

"""
    _tournament_select(fitnesses, tournament_size, rng) -> Int

Perform tournament selection. Returns the index of the selected individual.
"""
function _tournament_select(fitnesses::Vector{Float64}, tournament_size::Int, rng::AbstractRNG)
    n = length(fitnesses)
    best_idx = rand(rng, 1:n)
    for _ in 2:tournament_size
        idx = rand(rng, 1:n)
        if fitnesses[idx] < fitnesses[best_idx]
            best_idx = idx
        end
    end
    return best_idx
end

function select_parent(s::TournamentSelection,
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses,
                       rng::AbstractRNG)
    _tournament_select(selection_fitnesses, s.tournament_size, rng)
end

function select_parent(s::LexicaseSelection,
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses::Vector{Vector{Float64}},
                       rng::AbstractRNG)
    n_inds = length(case_fitnesses)
    n_inds == 0 && throw(ArgumentError("LexicaseSelection requires at least one individual"))
    n_cases = length(case_fitnesses[1])
    n_cases == 0 && return rand(rng, 1:n_inds)

    case_order = Random.shuffle(rng, collect(1:n_cases))
    candidates = collect(1:n_inds)

    for case_idx in case_order
        # Compute best loss on this case among current candidates.
        best = Inf
        for i in candidates
            v = case_fitnesses[i][case_idx]
            if v < best
                best = v
            end
        end
        # Filter candidates. Use exact equality for pure lexicase; NaN never
        # satisfies `v == best`, so NaN losses correctly drop out.
        new_candidates = Int[]
        for i in candidates
            if case_fitnesses[i][case_idx] == best
                push!(new_candidates, i)
            end
        end
        if isempty(new_candidates)
            # All candidates had non-finite losses on this case; skip.
            continue
        end
        candidates = new_candidates
        length(candidates) == 1 && return candidates[1]
    end

    return rand(rng, candidates)
end

function select_parent(s::EpsilonLexicaseSelection,
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses::Vector{Vector{Float64}},
                       rng::AbstractRNG)
    n_inds = length(case_fitnesses)
    n_inds == 0 && throw(ArgumentError("EpsilonLexicaseSelection requires at least one individual"))
    n_cases = length(case_fitnesses[1])
    n_cases == 0 && return rand(rng, 1:n_inds)

    # Precompute per-case auto-epsilon (MAD over the full population) when
    # s.epsilon == 0.0. Fixed epsilon uses the scalar directly.
    auto = s.epsilon <= 0.0
    per_case_eps = if auto
        eps_vec = zeros(Float64, n_cases)
        for c in 1:n_cases
            eps_vec[c] = _case_mad(case_fitnesses, c, n_inds)
        end
        eps_vec
    else
        fill(s.epsilon, n_cases)
    end

    case_order = Random.shuffle(rng, collect(1:n_cases))
    candidates = collect(1:n_inds)

    for case_idx in case_order
        best = Inf
        for i in candidates
            v = case_fitnesses[i][case_idx]
            if v < best
                best = v
            end
        end
        threshold = best + per_case_eps[case_idx]
        new_candidates = Int[]
        for i in candidates
            v = case_fitnesses[i][case_idx]
            if v <= threshold
                push!(new_candidates, i)
            end
        end
        if isempty(new_candidates)
            continue
        end
        candidates = new_candidates
        length(candidates) == 1 && return candidates[1]
    end

    return rand(rng, candidates)
end

# Clean MethodError when lexicase is paired with an evaluator that doesn't
# provide per-case fitness (the solve loop will never construct such a call
# — but user-written custom loops might).
function select_parent(s::Union{LexicaseSelection, EpsilonLexicaseSelection},
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses::Nothing,
                       rng::AbstractRNG)
    throw(ArgumentError(
        "$(typeof(s).name.name) requires per-case fitnesses, but case_fitnesses is `nothing`. " *
        "Ensure the evaluator implements `evaluate_cases`; the solve loop passes the matrix " *
        "automatically when `needs_cases(alg.selection) == true`."))
end

# ---------------------------------------------------------------------------
# Internal: per-case median-absolute-deviation for auto-epsilon lexicase.
# ---------------------------------------------------------------------------

function _case_mad(case_fitnesses::Vector{Vector{Float64}}, case_idx::Int, n_inds::Int)
    # Collect finite losses on this case. MAD over the full population,
    # not just a candidate subset (matches La Cava's formulation).
    vals = Float64[]
    sizehint!(vals, n_inds)
    for i in 1:n_inds
        v = case_fitnesses[i][case_idx]
        isfinite(v) && push!(vals, v)
    end
    isempty(vals) && return 0.0
    sorted = sort(vals)
    n = length(sorted)
    med = isodd(n) ? sorted[(n + 1) ÷ 2] : 0.5 * (sorted[n ÷ 2] + sorted[n ÷ 2 + 1])
    devs = [abs(v - med) for v in vals]
    sort!(devs)
    m = length(devs)
    return isodd(m) ? devs[(m + 1) ÷ 2] : 0.5 * (devs[m ÷ 2] + devs[m ÷ 2 + 1])
end
