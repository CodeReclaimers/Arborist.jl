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

"""
    FitnessProportionateSelection <: AbstractSelectionStrategy

Roulette-wheel selection. Under the library's minimization convention,
individual `i` is selected with probability proportional to
`1 / (eps + f[i] - f_min)`, so lower-fitness individuals get higher
selection weight. Edge cases:

- All fitnesses equal → uniform sampling over the population.
- Any `Inf` fitness → zero probability of selection (finite individuals
  absorb the whole distribution). All-`Inf` → uniform fallback.

Proportionate selection is the textbook baseline but is rarely optimal
in practice (premature convergence on multi-modal fitness). Included
for reproducibility of pre-2005 GA/GP literature.
"""
struct FitnessProportionateSelection <: AbstractSelectionStrategy end

"""
    RankSelection(; selection_pressure::Float64=1.5) <: AbstractSelectionStrategy

Linear-rank selection (Baker 1985), adapted for Arborist's minimization
convention. Individuals are sorted ascending by fitness (best first);
rank `r ∈ 1..N` gets weight

    w(r) = s - 2 * (s - 1) * (r - 1) / (N - 1)

so at `s == 1.0` selection is uniform, and at `s == 2.0` the best
individual (rank 1) has exactly twice the weight of the worst
(rank N). Values outside `[1.0, 2.0]` are rejected.

Less sensitive to fitness scaling than `FitnessProportionateSelection`
and often a better default when fitnesses span many orders of
magnitude. The sign is flipped from the textbook maximization form so
that — under the minimization convention — rank 1 is the *best* and
gets the highest weight.

# Fields
- `selection_pressure::Float64`: typical values in `[1.1, 2.0]` (default 1.5).
"""
struct RankSelection <: AbstractSelectionStrategy
    selection_pressure::Float64
end

function RankSelection(; selection_pressure::Float64 = 1.5)
    1.0 <= selection_pressure <= 2.0 || throw(ArgumentError(
        "RankSelection: selection_pressure must be in [1.0, 2.0], got $selection_pressure"))
    return RankSelection(selection_pressure)
end

"""
    TruncationSelection(; ratio::Float64=0.5) <: AbstractSelectionStrategy

Pick uniformly at random from the top `ceil(ratio * N)` individuals
(sorted ascending by fitness — best first). `ratio=1.0` degenerates to
uniform selection over the whole population; `ratio=1/N` selects only
the best.

Aggressive but predictable. Commonly used inside evolution-strategy
variants where intense exploitation is desired.

# Fields
- `ratio::Float64`: fraction of the population to keep (default 0.5).
"""
struct TruncationSelection <: AbstractSelectionStrategy
    ratio::Float64
end

function TruncationSelection(; ratio::Float64 = 0.5)
    0.0 < ratio <= 1.0 || throw(ArgumentError(
        "TruncationSelection: ratio must be in (0, 1], got $ratio"))
    return TruncationSelection(ratio)
end

# All three are scalar-fitness strategies — no case matrix needed.
needs_cases(::FitnessProportionateSelection) = false
needs_cases(::RankSelection) = false
needs_cases(::TruncationSelection) = false

# Proportionate selection: inverted-minimization weights, cumulative sample.
function select_parent(::FitnessProportionateSelection,
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses,
                       rng::AbstractRNG)
    n = length(selection_fitnesses)
    n == 0 && throw(ArgumentError(
        "FitnessProportionateSelection requires at least one individual"))

    finite_mask = isfinite.(selection_fitnesses)
    n_finite = count(finite_mask)
    n_finite == 0 && return rand(rng, 1:n)  # all-Inf: uniform fallback

    f_min = Inf
    for i in 1:n
        finite_mask[i] || continue
        selection_fitnesses[i] < f_min && (f_min = selection_fitnesses[i])
    end

    eps = 1e-12
    weights = Vector{Float64}(undef, n)
    total = 0.0
    @inbounds for i in 1:n
        w = finite_mask[i] ?
            1.0 / (eps + selection_fitnesses[i] - f_min) :
            0.0
        weights[i] = w
        total += w
    end

    # All-identical finite fitness → uniform over finite individuals.
    if total <= 0.0
        idx = rand(rng, 1:n_finite)
        k = 0
        for i in 1:n
            if finite_mask[i]
                k += 1
                k == idx && return i
            end
        end
        return n  # unreachable, appease the compiler
    end

    u = rand(rng) * total
    acc = 0.0
    @inbounds for i in 1:n
        acc += weights[i]
        acc >= u && return i
    end
    return n
end

# Rank selection: precompute normalized linear-rank weights once per call.
function select_parent(s::RankSelection,
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses,
                       rng::AbstractRNG)
    n = length(selection_fitnesses)
    n == 0 && throw(ArgumentError(
        "RankSelection requires at least one individual"))
    n == 1 && return 1

    order = sortperm(selection_fitnesses)  # ascending: best first
    sp = s.selection_pressure
    # Weights for rank r ∈ 1..n: w(r) = sp - 2*(sp - 1)*(r - 1)/(n - 1)
    # BEST (r=1) gets weight sp; WORST (r=n) gets weight 2 - sp. With
    # sp=2.0 the ratio is 2:0 (worst never picked); sp=1.0 is uniform.
    # Total weight sums exactly to n regardless of sp, so u ∈ [0, n).
    u = rand(rng) * n
    acc = 0.0
    @inbounds for r in 1:n
        w = sp - 2.0 * (sp - 1.0) * (r - 1) / (n - 1)
        acc += w
        acc >= u && return order[r]
    end
    return order[n]
end

# Truncation selection: uniform over top-k (sorted ascending = best-first).
function select_parent(s::TruncationSelection,
                       selection_fitnesses::Vector{Float64},
                       case_fitnesses,
                       rng::AbstractRNG)
    n = length(selection_fitnesses)
    n == 0 && throw(ArgumentError(
        "TruncationSelection requires at least one individual"))
    k = max(1, ceil(Int, s.ratio * n))
    order = sortperm(selection_fitnesses)
    return order[rand(rng, 1:k)]
end

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
