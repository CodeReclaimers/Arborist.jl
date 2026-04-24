# Classification benchmark generators.

"""
Fisher (1936) Iris dataset embedded verbatim. Order is setosa (1-50),
versicolor (51-100), virginica (101-150); each row is
`(sepal_length, sepal_width, petal_length, petal_width, class_index)`.
Public domain.
"""
const _IRIS_DATA = Tuple{Float32, Float32, Float32, Float32, Int}[
    # --- setosa (class 1) ---
    (5.1f0, 3.5f0, 1.4f0, 0.2f0, 1), (4.9f0, 3.0f0, 1.4f0, 0.2f0, 1),
    (4.7f0, 3.2f0, 1.3f0, 0.2f0, 1), (4.6f0, 3.1f0, 1.5f0, 0.2f0, 1),
    (5.0f0, 3.6f0, 1.4f0, 0.2f0, 1), (5.4f0, 3.9f0, 1.7f0, 0.4f0, 1),
    (4.6f0, 3.4f0, 1.4f0, 0.3f0, 1), (5.0f0, 3.4f0, 1.5f0, 0.2f0, 1),
    (4.4f0, 2.9f0, 1.4f0, 0.2f0, 1), (4.9f0, 3.1f0, 1.5f0, 0.1f0, 1),
    (5.4f0, 3.7f0, 1.5f0, 0.2f0, 1), (4.8f0, 3.4f0, 1.6f0, 0.2f0, 1),
    (4.8f0, 3.0f0, 1.4f0, 0.1f0, 1), (4.3f0, 3.0f0, 1.1f0, 0.1f0, 1),
    (5.8f0, 4.0f0, 1.2f0, 0.2f0, 1), (5.7f0, 4.4f0, 1.5f0, 0.4f0, 1),
    (5.4f0, 3.9f0, 1.3f0, 0.4f0, 1), (5.1f0, 3.5f0, 1.4f0, 0.3f0, 1),
    (5.7f0, 3.8f0, 1.7f0, 0.3f0, 1), (5.1f0, 3.8f0, 1.5f0, 0.3f0, 1),
    (5.4f0, 3.4f0, 1.7f0, 0.2f0, 1), (5.1f0, 3.7f0, 1.5f0, 0.4f0, 1),
    (4.6f0, 3.6f0, 1.0f0, 0.2f0, 1), (5.1f0, 3.3f0, 1.7f0, 0.5f0, 1),
    (4.8f0, 3.4f0, 1.9f0, 0.2f0, 1), (5.0f0, 3.0f0, 1.6f0, 0.2f0, 1),
    (5.0f0, 3.4f0, 1.6f0, 0.4f0, 1), (5.2f0, 3.5f0, 1.5f0, 0.2f0, 1),
    (5.2f0, 3.4f0, 1.4f0, 0.2f0, 1), (4.7f0, 3.2f0, 1.6f0, 0.2f0, 1),
    (4.8f0, 3.1f0, 1.6f0, 0.2f0, 1), (5.4f0, 3.4f0, 1.5f0, 0.4f0, 1),
    (5.2f0, 4.1f0, 1.5f0, 0.1f0, 1), (5.5f0, 4.2f0, 1.4f0, 0.2f0, 1),
    (4.9f0, 3.1f0, 1.5f0, 0.2f0, 1), (5.0f0, 3.2f0, 1.2f0, 0.2f0, 1),
    (5.5f0, 3.5f0, 1.3f0, 0.2f0, 1), (4.9f0, 3.6f0, 1.4f0, 0.1f0, 1),
    (4.4f0, 3.0f0, 1.3f0, 0.2f0, 1), (5.1f0, 3.4f0, 1.5f0, 0.2f0, 1),
    (5.0f0, 3.5f0, 1.3f0, 0.3f0, 1), (4.5f0, 2.3f0, 1.3f0, 0.3f0, 1),
    (4.4f0, 3.2f0, 1.3f0, 0.2f0, 1), (5.0f0, 3.5f0, 1.6f0, 0.6f0, 1),
    (5.1f0, 3.8f0, 1.9f0, 0.4f0, 1), (4.8f0, 3.0f0, 1.4f0, 0.3f0, 1),
    (5.1f0, 3.8f0, 1.6f0, 0.2f0, 1), (4.6f0, 3.2f0, 1.4f0, 0.2f0, 1),
    (5.3f0, 3.7f0, 1.5f0, 0.2f0, 1), (5.0f0, 3.3f0, 1.4f0, 0.2f0, 1),
    # --- versicolor (class 2) ---
    (7.0f0, 3.2f0, 4.7f0, 1.4f0, 2), (6.4f0, 3.2f0, 4.5f0, 1.5f0, 2),
    (6.9f0, 3.1f0, 4.9f0, 1.5f0, 2), (5.5f0, 2.3f0, 4.0f0, 1.3f0, 2),
    (6.5f0, 2.8f0, 4.6f0, 1.5f0, 2), (5.7f0, 2.8f0, 4.5f0, 1.3f0, 2),
    (6.3f0, 3.3f0, 4.7f0, 1.6f0, 2), (4.9f0, 2.4f0, 3.3f0, 1.0f0, 2),
    (6.6f0, 2.9f0, 4.6f0, 1.3f0, 2), (5.2f0, 2.7f0, 3.9f0, 1.4f0, 2),
    (5.0f0, 2.0f0, 3.5f0, 1.0f0, 2), (5.9f0, 3.0f0, 4.2f0, 1.5f0, 2),
    (6.0f0, 2.2f0, 4.0f0, 1.0f0, 2), (6.1f0, 2.9f0, 4.7f0, 1.4f0, 2),
    (5.6f0, 2.9f0, 3.6f0, 1.3f0, 2), (6.7f0, 3.1f0, 4.4f0, 1.4f0, 2),
    (5.6f0, 3.0f0, 4.5f0, 1.5f0, 2), (5.8f0, 2.7f0, 4.1f0, 1.0f0, 2),
    (6.2f0, 2.2f0, 4.5f0, 1.5f0, 2), (5.6f0, 2.5f0, 3.9f0, 1.1f0, 2),
    (5.9f0, 3.2f0, 4.8f0, 1.8f0, 2), (6.1f0, 2.8f0, 4.0f0, 1.3f0, 2),
    (6.3f0, 2.5f0, 4.9f0, 1.5f0, 2), (6.1f0, 2.8f0, 4.7f0, 1.2f0, 2),
    (6.4f0, 2.9f0, 4.3f0, 1.3f0, 2), (6.6f0, 3.0f0, 4.4f0, 1.4f0, 2),
    (6.8f0, 2.8f0, 4.8f0, 1.4f0, 2), (6.7f0, 3.0f0, 5.0f0, 1.7f0, 2),
    (6.0f0, 2.9f0, 4.5f0, 1.5f0, 2), (5.7f0, 2.6f0, 3.5f0, 1.0f0, 2),
    (5.5f0, 2.4f0, 3.8f0, 1.1f0, 2), (5.5f0, 2.4f0, 3.7f0, 1.0f0, 2),
    (5.8f0, 2.7f0, 3.9f0, 1.2f0, 2), (6.0f0, 2.7f0, 5.1f0, 1.6f0, 2),
    (5.4f0, 3.0f0, 4.5f0, 1.5f0, 2), (6.0f0, 3.4f0, 4.5f0, 1.6f0, 2),
    (6.7f0, 3.1f0, 4.7f0, 1.5f0, 2), (6.3f0, 2.3f0, 4.4f0, 1.3f0, 2),
    (5.6f0, 3.0f0, 4.1f0, 1.3f0, 2), (5.5f0, 2.5f0, 4.0f0, 1.3f0, 2),
    (5.5f0, 2.6f0, 4.4f0, 1.2f0, 2), (6.1f0, 3.0f0, 4.6f0, 1.4f0, 2),
    (5.8f0, 2.6f0, 4.0f0, 1.2f0, 2), (5.0f0, 2.3f0, 3.3f0, 1.0f0, 2),
    (5.6f0, 2.7f0, 4.2f0, 1.3f0, 2), (5.7f0, 3.0f0, 4.2f0, 1.2f0, 2),
    (5.7f0, 2.9f0, 4.2f0, 1.3f0, 2), (6.2f0, 2.9f0, 4.3f0, 1.3f0, 2),
    (5.1f0, 2.5f0, 3.0f0, 1.1f0, 2), (5.7f0, 2.8f0, 4.1f0, 1.3f0, 2),
    # --- virginica (class 3) ---
    (6.3f0, 3.3f0, 6.0f0, 2.5f0, 3), (5.8f0, 2.7f0, 5.1f0, 1.9f0, 3),
    (7.1f0, 3.0f0, 5.9f0, 2.1f0, 3), (6.3f0, 2.9f0, 5.6f0, 1.8f0, 3),
    (6.5f0, 3.0f0, 5.8f0, 2.2f0, 3), (7.6f0, 3.0f0, 6.6f0, 2.1f0, 3),
    (4.9f0, 2.5f0, 4.5f0, 1.7f0, 3), (7.3f0, 2.9f0, 6.3f0, 1.8f0, 3),
    (6.7f0, 2.5f0, 5.8f0, 1.8f0, 3), (7.2f0, 3.6f0, 6.1f0, 2.5f0, 3),
    (6.5f0, 3.2f0, 5.1f0, 2.0f0, 3), (6.4f0, 2.7f0, 5.3f0, 1.9f0, 3),
    (6.8f0, 3.0f0, 5.5f0, 2.1f0, 3), (5.7f0, 2.5f0, 5.0f0, 2.0f0, 3),
    (5.8f0, 2.8f0, 5.1f0, 2.4f0, 3), (6.4f0, 3.2f0, 5.3f0, 2.3f0, 3),
    (6.5f0, 3.0f0, 5.5f0, 1.8f0, 3), (7.7f0, 3.8f0, 6.7f0, 2.2f0, 3),
    (7.7f0, 2.6f0, 6.9f0, 2.3f0, 3), (6.0f0, 2.2f0, 5.0f0, 1.5f0, 3),
    (6.9f0, 3.2f0, 5.7f0, 2.3f0, 3), (5.6f0, 2.8f0, 4.9f0, 2.0f0, 3),
    (7.7f0, 2.8f0, 6.7f0, 2.0f0, 3), (6.3f0, 2.7f0, 4.9f0, 1.8f0, 3),
    (6.7f0, 3.3f0, 5.7f0, 2.1f0, 3), (7.2f0, 3.2f0, 6.0f0, 1.8f0, 3),
    (6.2f0, 2.8f0, 4.8f0, 1.8f0, 3), (6.1f0, 3.0f0, 4.9f0, 1.8f0, 3),
    (6.4f0, 2.8f0, 5.6f0, 2.1f0, 3), (7.2f0, 3.0f0, 5.8f0, 1.6f0, 3),
    (7.4f0, 2.8f0, 6.1f0, 1.9f0, 3), (7.9f0, 3.8f0, 6.4f0, 2.0f0, 3),
    (6.4f0, 2.8f0, 5.6f0, 2.2f0, 3), (6.3f0, 2.8f0, 5.1f0, 1.5f0, 3),
    (6.1f0, 2.6f0, 5.6f0, 1.4f0, 3), (7.7f0, 3.0f0, 6.1f0, 2.3f0, 3),
    (6.3f0, 3.4f0, 5.6f0, 2.4f0, 3), (6.4f0, 3.1f0, 5.5f0, 1.8f0, 3),
    (6.0f0, 3.0f0, 4.8f0, 1.8f0, 3), (6.9f0, 3.1f0, 5.4f0, 2.1f0, 3),
    (6.7f0, 3.1f0, 5.6f0, 2.4f0, 3), (6.9f0, 3.1f0, 5.1f0, 2.3f0, 3),
    (5.8f0, 2.7f0, 5.1f0, 1.9f0, 3), (6.8f0, 3.2f0, 5.9f0, 2.3f0, 3),
    (6.7f0, 3.3f0, 5.7f0, 2.5f0, 3), (6.7f0, 3.0f0, 5.2f0, 2.3f0, 3),
    (6.3f0, 2.5f0, 5.0f0, 1.9f0, 3), (6.5f0, 3.0f0, 5.2f0, 2.0f0, 3),
    (6.2f0, 3.4f0, 5.4f0, 2.3f0, 3), (5.9f0, 3.0f0, 5.1f0, 1.8f0, 3),
]

"""
    iris(; rng=MersenneTwister(20260421), test_ratio=0.2, T=Float32) -> NamedTuple

UCI Iris dataset (Fisher 1936), 150 rows × 4 features × 3 classes,
embedded inline (no network dependency). Stratified split keyed by
`rng`: default `MersenneTwister(20260421)` reproduces the split used
in `test/benchmarks/iris_classification.jl` prior to the Benchmarks
module refactor.

Returns `(; X_train, y_train, X_test, y_test, feature_names, class_names,
           n_classes, target_expr, name)` where `y_train`, `y_test` are
`Vector{Int}` of class indices (1-3).
"""
function iris(; rng::AbstractRNG=MersenneTwister(20260421),
                test_ratio::Real=0.2,
                T::Type=Float32)
    0 < test_ratio < 1 || throw(ArgumentError(
        "iris: test_ratio must be in (0, 1), got $test_ratio"))

    # Materialize the full X, y from the inline tuple data.
    n = length(_IRIS_DATA)
    X = Matrix{T}(undef, 4, n)
    y = Vector{Int}(undef, n)
    for (col, r) in enumerate(_IRIS_DATA)
        X[1, col] = T(r[1]); X[2, col] = T(r[2])
        X[3, col] = T(r[3]); X[4, col] = T(r[4])
        y[col] = r[5]
    end

    # Stratify by class label via the shared train_test_split utility.
    X_train, y_train, X_test, y_test = train_test_split(
        X, y; test_size=test_ratio, rng=rng, stratify=y)

    return (;
        X_train = X_train,
        y_train = y_train,
        X_test  = X_test,
        y_test  = y_test,
        feature_names = ["sepal_length", "sepal_width",
                         "petal_length", "petal_width"],
        class_names = ["setosa", "versicolor", "virginica"],
        n_classes = 3,
        target_expr = "one-vs-rest 3-class classification",
        name = "Iris")
end

"""
    two_spirals(; T=Float64) -> NamedTuple

Lang & Witbrock (1989) two-spirals dataset. 194 points = 97 per spiral,
interleaved in columns (spiral 1 at odd columns, spiral 2 at even).
Hard-for-polynomial-boundaries; canonical NEAT stress test.

Returns `(; X, y, name, target_expr)` where `X::Matrix{T}(2, 194)` and
`y::Matrix{T}(1, 194)` with values `±1` matching the standard tanh-
compatible encoding.
"""
function two_spirals(; T::Type=Float64)
    n_per_spiral = 97
    n_total = 2 * n_per_spiral

    X = zeros(T, 2, n_total)
    y = zeros(T, 1, n_total)
    for k in 0:(n_per_spiral - 1)
        angle = k * T(π) / T(16)
        radius = T(6.5) * T(104 - k) / T(104)
        x = radius * sin(angle)
        yv = radius * cos(angle)
        # Spiral 1 at odd column, spiral 2 (negated) at even column.
        X[1, 2k + 1] =  x;  X[2, 2k + 1] =  yv
        X[1, 2k + 2] = -x;  X[2, 2k + 2] = -yv
        y[1, 2k + 1] =  one(T)
        y[1, 2k + 2] = -one(T)
    end
    return (; X=X, y=y,
              name="Two-Spirals",
              target_expr="binary classification, ±1 encoding")
end
