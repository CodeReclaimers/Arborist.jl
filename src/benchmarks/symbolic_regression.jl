# Symbolic regression benchmark generators.
#
# All generators return a NamedTuple:
#   (; X::Matrix{T}, y::Vector{T}, target_expr::String, name::String)
# For generators with a held-out test set (Keijzer), additional fields
# `X_test, y_test` are included.

"""
    nguyen(n::Int; T=Float32) -> NamedTuple

Nguyen symbolic regression benchmarks (Uytterhoeven 2008, codified in
McDermott et al. 2012). Valid `n`: 1 through 10. Univariate for 1–8
(20 evenly spaced points), bivariate for 9–10 (100 random points in
`[-1, 1]²` with fixed seed 1234 for reproducibility).

Returns `(; X, y, target_expr, name, domain_description)`.

Targets:
- 1: `x³ + x² + x`, x ∈ [-1, 1]
- 2: `x⁴ + x³ + x² + x`, x ∈ [-1, 1]
- 3: `x⁵ + x⁴ + x³ + x² + x`, x ∈ [-1, 1]
- 4: `x⁶ + x⁵ + x⁴ + x³ + x² + x`, x ∈ [-1, 1]
- 5: `sin(x²) * cos(x) - 1`, x ∈ [-1, 1]
- 6: `sin(x) + sin(x + x²)`, x ∈ [-1, 1]
- 7: `log(x+1) + log(x²+1)`, x ∈ [0, 2]
- 8: `sqrt(x)`, x ∈ [0, 4]
- 9: `sin(x₁) + sin(x₂²)`, (x₁, x₂) ∈ [-1, 1]²
- 10: `2 * sin(x₁) * cos(x₂)`, (x₁, x₂) ∈ [-1, 1]²
"""
function nguyen(n::Int; T::Type=Float32)
    1 <= n <= 10 || throw(ArgumentError("nguyen: n must be in 1..10, got $n"))

    if n == 7
        xs = T.(range(0, 2, length=20))
        X  = reshape(xs, 1, :)
        y  = log.(xs .+ one(T)) .+ log.(xs .^ 2 .+ one(T))
        return (; X=Matrix(X), y=Vector(y),
                  target_expr="log(x+1) + log(x²+1)",
                  name="Nguyen-7",
                  domain_description="x ∈ [0, 2]")
    elseif n == 8
        xs = T.(range(0, 4, length=20))
        X  = reshape(xs, 1, :)
        y  = sqrt.(xs)
        return (; X=Matrix(X), y=Vector(y),
                  target_expr="sqrt(x)",
                  name="Nguyen-8",
                  domain_description="x ∈ [0, 4]")
    elseif 1 <= n <= 6
        xs = T.(range(-1, 1, length=20))
        X  = reshape(xs, 1, :)
        target_expr, y = if n == 1
            "x³ + x² + x", xs .^ 3 .+ xs .^ 2 .+ xs
        elseif n == 2
            "x⁴ + x³ + x² + x", xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs
        elseif n == 3
            "x⁵ + x⁴ + x³ + x² + x", xs .^ 5 .+ xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs
        elseif n == 4
            "x⁶ + x⁵ + x⁴ + x³ + x² + x",
                xs .^ 6 .+ xs .^ 5 .+ xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs
        elseif n == 5
            "sin(x²) * cos(x) - 1", sin.(xs .^ 2) .* cos.(xs) .- one(T)
        else  # n == 6
            "sin(x) + sin(x + x²)", sin.(xs) .+ sin.(xs .+ xs .^ 2)
        end
        return (; X=Matrix(X), y=Vector(y),
                  target_expr=target_expr,
                  name="Nguyen-$n",
                  domain_description="x ∈ [-1, 1]")
    else  # n == 9 or 10
        rng = MersenneTwister(1234)
        n_pts = 100
        X = T.(2 .* rand(rng, 2, n_pts) .- 1)
        xs2 = view(X, 1, :)
        ys2 = view(X, 2, :)
        target_expr, y = if n == 9
            "sin(x₁) + sin(x₂²)", sin.(xs2) .+ sin.(ys2 .^ 2)
        else  # 10
            "2 * sin(x₁) * cos(x₂)", T(2) .* sin.(xs2) .* cos.(ys2)
        end
        return (; X=Matrix(X), y=Vector(y),
                  target_expr=target_expr,
                  name="Nguyen-$n",
                  domain_description="(x₁, x₂) ∈ [-1, 1]²")
    end
end

"""
    keijzer(variant::Symbol; T=Float32) -> NamedTuple

Keijzer extrapolation benchmarks (Keijzer 2003, McDermott et al. 2012).
Unlike Nguyen, these evaluate a trained model on held-out *test* points
to measure extrapolation, not just interpolation.

Returned `NamedTuple` includes `X, y` for training, `X_test_interior`
and `X_test_extrapolation` (with `y_test_interior`, `y_test_extrapolation`)
for test evaluation.

Valid variants:
- `:k4`: `x * y * z`, 20 random training points in `[-1, 1]³`, interior
  test on `[-1, 1]³` grid, extrapolation test on `[-2, 2]³ ∖ [-1, 1]³`.
- `:k11`: `x * y + sin((x-1) * (y-1))`, 20 training points random in
  `[-3, 3]²`, test on 20×20 grid in the same domain (sparse training is
  the whole point).
"""
function keijzer(variant::Symbol; T::Type=Float32)
    if variant === :k4
        rng = MersenneTwister(2003)
        n_train = 20
        X_tr = T.(2 .* rand(rng, 3, n_train) .- 1)
        y_tr = vec(X_tr[1, :] .* X_tr[2, :] .* X_tr[3, :])

        # Interior test grid (5³ = 125 points).
        pts = T.(range(-1, 1, length=5))
        X_int = Matrix{T}(undef, 3, 125)
        idx = 1
        for a in pts, b in pts, c in pts
            X_int[:, idx] = [a, b, c]; idx += 1
        end
        y_int = vec(X_int[1, :] .* X_int[2, :] .* X_int[3, :])

        # Extrapolation test: 125 points in [-2, 2]³ outside [-1, 1]³.
        n_ex = 125
        X_ex = Matrix{T}(undef, 3, n_ex)
        filled = 0
        while filled < n_ex
            p = T.(4 .* rand(rng, 3) .- 2)
            if maximum(abs.(p)) > 1
                filled += 1
                X_ex[:, filled] = p
            end
        end
        y_ex = vec(X_ex[1, :] .* X_ex[2, :] .* X_ex[3, :])

        return (; X=X_tr, y=y_tr,
                  X_test_interior=X_int, y_test_interior=y_int,
                  X_test_extrapolation=X_ex, y_test_extrapolation=y_ex,
                  target_expr="x * y * z",
                  name="Keijzer-4",
                  domain_description="train: 20 random in [-1,1]³; test interior grid, extrapolation [-2,2]³\\[-1,1]³")
    elseif variant === :k11
        rng = MersenneTwister(2011)
        n_train = 20
        X_tr = T.(6 .* rand(rng, 2, n_train) .- 3)
        y_tr = vec(X_tr[1, :] .* X_tr[2, :] .+ sin.((X_tr[1, :] .- 1) .* (X_tr[2, :] .- 1)))

        # 20×20 test grid.
        pts = T.(range(-3, 3, length=20))
        X_te = Matrix{T}(undef, 2, 400)
        idx = 1
        for a in pts, b in pts
            X_te[:, idx] = [a, b]; idx += 1
        end
        y_te = vec(X_te[1, :] .* X_te[2, :] .+ sin.((X_te[1, :] .- 1) .* (X_te[2, :] .- 1)))

        return (; X=X_tr, y=y_tr,
                  X_test_interior=X_te, y_test_interior=y_te,
                  target_expr="x * y + sin((x-1)*(y-1))",
                  name="Keijzer-11",
                  domain_description="train: 20 random in [-3,3]²; test 20×20 grid same domain")
    else
        throw(ArgumentError("keijzer: variant must be :k4 or :k11, got $(repr(variant))"))
    end
end

"""
    koza(name::Symbol; T=Float32) -> NamedTuple

Koza's original SR benchmarks (Koza 1992). Valid names:
- `:quartic`: `x⁴ + x³ + x² + x`, x ∈ [-1, 1], 20 points. Same target as
  Nguyen-2 but this is the historical reference.
- `:septic`: `x⁷ + x⁶ + x⁵ + x⁴ + x³ + x² + x`, x ∈ [-1, 1], 20 points.
- `:nonic`: `x⁹ + x⁸ + ... + x`, x ∈ [-1, 1], 20 points.
"""
function koza(name::Symbol; T::Type=Float32)
    xs = T.(range(-1, 1, length=20))
    X  = reshape(xs, 1, :)
    if name === :quartic
        y = xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs
        expr = "x⁴ + x³ + x² + x"
    elseif name === :septic
        y = sum(xs .^ k for k in 1:7)
        expr = "x + x² + x³ + x⁴ + x⁵ + x⁶ + x⁷"
    elseif name === :nonic
        y = sum(xs .^ k for k in 1:9)
        expr = "x + x² + ... + x⁹"
    else
        throw(ArgumentError("koza: name must be :quartic, :septic, or :nonic, got $(repr(name))"))
    end
    return (; X=Matrix(X), y=Vector(y),
              target_expr=expr,
              name="Koza-$(name)",
              domain_description="x ∈ [-1, 1]")
end

"""
    pagie(; T=Float32) -> NamedTuple

Pagie-1 benchmark: `1/(1 + x⁻⁴) + 1/(1 + y⁻⁴)`, 676 points on 26×26
grid over `[-5, 5]²`. Hard because of the near-singular inverse terms.
"""
function pagie(; T::Type=Float32)
    pts = T.(range(-5, 5, length=26))
    X = Matrix{T}(undef, 2, 676)
    idx = 1
    for a in pts, b in pts
        X[:, idx] = [a, b]; idx += 1
    end
    x1 = view(X, 1, :)
    x2 = view(X, 2, :)
    # Protected inverse at x=0 to avoid Inf in the dataset itself.
    safe_inv4(z) = let z4 = z^4; z4 < T(1e-6) ? T(1e6) : one(T) / z4; end
    y = one(T) ./ (one(T) .+ safe_inv4.(x1)) .+ one(T) ./ (one(T) .+ safe_inv4.(x2))

    return (; X=X, y=vec(y),
              target_expr="1/(1 + x⁻⁴) + 1/(1 + y⁻⁴)",
              name="Pagie-1",
              domain_description="(x, y) ∈ [-5, 5]² on 26×26 grid")
end
