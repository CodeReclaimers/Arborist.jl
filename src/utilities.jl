# Cross-cutting utilities: train/test split, summary statistics,
# multi-seed runner. These cover needs that come up in every serious
# benchmark / comparison script; keeping them in one place eliminates
# ad-hoc reimplementations.

"""
    train_test_split(X::AbstractMatrix, y::AbstractVector;
                     test_size=0.2,
                     rng=Random.default_rng(),
                     stratify=nothing
                    ) -> (X_train, y_train, X_test, y_test)

Split a feature matrix `X` (features × samples) and matching target
vector `y` into train and test partitions.

# Arguments

- `X::AbstractMatrix`: features-by-samples (columns are samples).
- `y::AbstractVector`: one target per column of `X`.

# Keyword arguments

- `test_size::Real` (default 0.2): fraction of samples to place in the
  test set. Must be in `(0, 1)`.
- `rng::AbstractRNG` (default `Random.default_rng()`): RNG used for
  the permutation. Pass a `MersenneTwister(seed)` for reproducibility.
- `stratify::Union{Nothing, AbstractVector}` (default `nothing`): when
  supplied, a class-label vector of the same length as `y`. Sampling
  is done class-wise so the class proportions in both partitions
  match the input distribution as closely as integer rounding allows.

Returns a 4-tuple `(X_train, y_train, X_test, y_test)` of sub-matrices
and sub-vectors.
"""
function train_test_split(X::AbstractMatrix, y::AbstractVector;
                          test_size::Real=0.2,
                          rng::AbstractRNG=Random.default_rng(),
                          stratify::Union{Nothing, AbstractVector}=nothing)
    n = length(y)
    size(X, 2) == n || throw(ArgumentError(
        "train_test_split: X has $(size(X, 2)) columns but y has $n elements"))
    0 < test_size < 1 || throw(ArgumentError(
        "train_test_split: test_size must be in (0, 1), got $test_size"))

    if stratify === nothing
        n_test = round(Int, test_size * n)
        n_test = clamp(n_test, 1, n - 1)
        perm = randperm(rng, n)
        test_idx  = perm[1:n_test]
        train_idx = perm[(n_test + 1):end]
    else
        length(stratify) == n || throw(ArgumentError(
            "train_test_split: stratify length ($(length(stratify))) must match y length ($n)"))
        train_idx = Int[]
        test_idx  = Int[]
        for cls in unique(stratify)
            cls_rows = findall(==(cls), stratify)
            nc = length(cls_rows)
            nc_test = round(Int, test_size * nc)
            nc_test = clamp(nc_test, nc >= 2 ? 1 : 0, nc - 1)
            perm = cls_rows[randperm(rng, nc)]
            append!(test_idx,  perm[1:nc_test])
            append!(train_idx, perm[(nc_test + 1):end])
        end
    end

    X_train = X[:, train_idx]
    X_test  = X[:, test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]
    return (X_train, y_train, X_test, y_test)
end

"""
    summarize(xs::AbstractVector{<:Real}) -> NamedTuple

Return `(; mean, std, median, min, max, q25, q75, n)` for a vector of
real values. Non-finite entries are *excluded* from every statistic so
a single `Inf` fitness does not poison the summary. `n` reports the
number of finite entries used.

Matches the shape most benchmark reporting expects: mean ± std for a
quick headline, quartiles for distribution shape. No weak
dependency on Statistics (so `Pkg.test` in a sandboxed environment
works).
"""
function summarize(xs::AbstractVector{<:Real})
    finite = [Float64(x) for x in xs if isfinite(x)]
    n = length(finite)
    n == 0 && return (; mean=NaN, std=NaN, median=NaN,
                        min=NaN, max=NaN, q25=NaN, q75=NaN, n=0)

    m = sum(finite) / n
    var = n > 1 ? sum((x - m)^2 for x in finite) / (n - 1) : 0.0
    s = sqrt(var)

    sorted = sort(finite)
    return (; mean=m, std=s,
              median=_quantile_sorted(sorted, 0.5),
              min=sorted[1], max=sorted[end],
              q25=_quantile_sorted(sorted, 0.25),
              q75=_quantile_sorted(sorted, 0.75),
              n=n)
end

# Linear-interpolation quantile on a pre-sorted vector. Matches
# Julia's Statistics.quantile behavior with default linear interpolation.
function _quantile_sorted(sorted::Vector{Float64}, p::Float64)
    n = length(sorted)
    n == 0 && return NaN
    n == 1 && return sorted[1]
    idx = p * (n - 1) + 1
    lo = floor(Int, idx)
    hi = ceil(Int, idx)
    lo == hi && return sorted[lo]
    frac = idx - lo
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
end

"""
    run_multi_seed(f, seeds::AbstractVector{Int}; parallel=false) -> Vector

Call `f(seed)` for each integer in `seeds` and return a vector of the
results. When `parallel=true` and `Threads.nthreads() > 1`, runs
concurrently via `Threads.@threads` — callers must ensure `f` is
thread-safe (no shared mutable state without locking).

Typical use:
```julia
fitnesses = run_multi_seed([1, 2, 3, 4, 5]) do seed
    problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
    result = solve(problem, alg)
    result.best_fitness
end
println(summarize(fitnesses))
```
"""
function run_multi_seed(f, seeds::AbstractVector{Int}; parallel::Bool=false)
    n = length(seeds)
    n == 0 && return Any[]
    out = Vector{Any}(undef, n)
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            out[i] = f(seeds[i])
        end
    else
        for i in 1:n
            out[i] = f(seeds[i])
        end
    end
    return out
end
