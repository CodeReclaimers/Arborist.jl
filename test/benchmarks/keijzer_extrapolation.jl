# Keijzer extrapolation symbolic regression benchmarks.
#
# Keijzer (2003) "Improving Symbolic Regression with Interval Arithmetic
# and Linear Scaling." The Keijzer suite's signature contribution over
# Koza and Nguyen is explicit train/test separation — training samples
# live in one interval and the test set probes the model's behavior
# _outside_ that interval, exposing GP's tendency to fit polynomial
# tails that diverge on unseen data.
#
# We implement K-4 (univariate, dense training grid) and K-11
# (bivariate, sparse random training) as representative points in the
# extrapolation-difficulty spectrum.

using DynamicExpressions
using Random
using Statistics: mean

# Reuse the protected operators from nguyen_regression if already loaded,
# else define locally.  Guard with `@isdefined` so the file can also run
# standalone without depending on include-order.
if !@isdefined(_pdiv)
    _pdiv(a::Float32, b::Float32) = abs(b) < 1f-6 ? 1f0 : a / b
end
if !@isdefined(_plog)
    _plog(x::Float32) = x > 0f0 ? log(x) : 0f0
end
if !@isdefined(_psqrt)
    _psqrt(x::Float32) = sqrt(abs(x))
end

@testset "Keijzer extrapolation benchmarks (TreeGenome)" begin
    operators = OperatorEnum(;
        binary_operators=[+, -, *, _pdiv],
        unary_operators=[sin, cos, exp, _plog, _psqrt]
    )

    algorithm = GeneticProgramming(
        pop_size=200,
        generations=400,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    # --- Keijzer-4: f(x) = x^3 * exp(-x) * cos(x) * sin(x) * (sin(x)^2 * cos(x) - 1) ---
    # Canonical protocol: train x ∈ [0, 10] step 0.05 (200 pts),
    # test x ∈ [0.05, 10.05] step 0.05 (interior held-out).
    # We additionally compute an out-of-range test on x ∈ [10.05, 12] as
    # the true extrapolation signal — reported but not gated (Keijzer-4
    # extrapolation is expected to fail for most GP runs; the gate lives
    # on train fitness).
    @testset "Keijzer-4: x^3 * exp(-x) * cos(x) * sin(x) * (sin^2(x) * cos(x) - 1)" begin
        k4(x) = x^3 * exp(-x) * cos(x) * sin(x) * (sin(x)^2 * cos(x) - 1)

        xs_train = Float32.(collect(0.0:0.05:10.0))
        xs_test_interior  = Float32.(collect(0.05:0.05:10.05))
        xs_test_extrap    = Float32.(collect(10.05:0.05:12.0))

        X_train  = reshape(xs_train, 1, :)
        y_train  = Float32.(k4.(xs_train))
        X_tint   = reshape(xs_test_interior, 1, :)
        y_tint   = Float32.(k4.(xs_test_interior))
        X_textr  = reshape(xs_test_extrap, 1, :)
        y_textr  = Float32.(k4.(xs_test_extrap))

        evaluator = TreeFitnessEvaluator(X_train, y_train, operators)

        # Keijzer-4 target function has small amplitude (~10^-3 peak) after
        # the x^3*exp(-x) envelope decays.  Gate is train MSE < 0.01 for
        # 3/5 seeds — in-distribution fit.  Extrapolation (x ∈ [10, 12]) is
        # the diagnostic signal, reported per-seed but not gated: the whole
        # point of this benchmark is to expose that GP does not generalize
        # beyond the training range, so a tight extrapolation gate would
        # just flicker.
        train_successes = 0
        for seed in 1:5
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)

            best = result.best_genome
            pred_tint  = best.tree(X_tint,  operators)
            pred_textr = best.tree(X_textr, operators)
            rmse_tint  = sqrt(mean((Float64.(pred_tint)  .- Float64.(y_tint))  .^ 2))
            rmse_textr = sqrt(mean((Float64.(pred_textr) .- Float64.(y_textr)) .^ 2))

            println("    Keijzer-4 seed=$seed: train_fit=$(round(result.best_fitness, sigdigits=3)), " *
                    "test_rmse_interior=$(round(rmse_tint, sigdigits=3)), " *
                    "test_rmse_extrap=$(round(rmse_textr, sigdigits=3))")
            flush(stdout)
            if isfinite(result.best_fitness) && result.best_fitness < 0.01
                train_successes += 1
            end
        end

        println("  Keijzer-4: $train_successes/5 seeds reached train MSE < 0.01")
        flush(stdout)
        @test train_successes >= 3
    end

    # --- Keijzer-11: f(x, y) = x*y + sin((x-1)*(y-1)) ---
    # Canonical protocol: 20 random training points uniform in [-3, 3]^2,
    # test on a 20×20 grid in [-3, 3]^2.  Both train and test lie in the
    # same region, but the sparse training set exposes overfitting.
    @testset "Keijzer-11: x*y + sin((x-1)*(y-1))" begin
        k11(x, y) = x*y + sin((x-1)*(y-1))

        # Fixed-seed training sample.
        rng = MersenneTwister(42)
        n_train = 20
        X_train = Float32.(6 .* rand(rng, 2, n_train) .- 3f0)  # [-3, 3]^2
        y_train = Float32[k11(X_train[1, i], X_train[2, i]) for i in 1:n_train]

        # Test grid.
        gxs = Float32.(collect(-3.0:(6/19):3.0))
        gys = Float32.(collect(-3.0:(6/19):3.0))
        X_test = zeros(Float32, 2, length(gxs) * length(gys))
        y_test = zeros(Float32, length(gxs) * length(gys))
        idx = 0
        for gx in gxs, gy in gys
            idx += 1
            X_test[1, idx] = gx
            X_test[2, idx] = gy
            y_test[idx]    = k11(gx, gy)
        end

        evaluator = TreeFitnessEvaluator(X_train, y_train, operators)

        # Report train fitness + test RMSE per seed; gate: 2/5 seeds
        # achieve train MSE < 1.0 (loose — train has only 20 points,
        # variance is high).
        train_successes = 0
        for seed in 1:5
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)

            best = result.best_genome
            pred_test = best.tree(X_test, operators)
            rmse_test = sqrt(mean((Float64.(pred_test) .- Float64.(y_test)) .^ 2))

            println("    Keijzer-11 seed=$seed: train_fit=$(round(result.best_fitness, sigdigits=3)), " *
                    "test_rmse=$(round(rmse_test, sigdigits=3))")
            flush(stdout)
            if isfinite(result.best_fitness) && result.best_fitness < 1.0
                train_successes += 1
            end
        end

        println("  Keijzer-11: $train_successes/5 seeds reached train MSE < 1.0")
        flush(stdout)
        @test train_successes >= 2
    end
end
