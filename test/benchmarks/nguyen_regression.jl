# Nguyen symbolic regression benchmark suite — Nguyen-1 through Nguyen-10.
#
# Uytterhoeven (2008) / Nguyen et al. benchmarks as codified in
# McDermott et al. (2012) "Genetic Programming Needs Better Benchmarks."
# These are the de-facto standard beyond Koza for symbolic regression
# comparisons in modern GP papers.
#
# Uses TreeGenome (DynamicExpressions.jl) for vectorized evaluation.
#
# Operator set: {+, -, *, pdiv, sin, cos, exp, plog, psqrt}, where
# protected variants (pdiv, plog, psqrt) keep evolution well-behaved
# rather than relying on NaN→Inf penalization alone.

using DynamicExpressions
using Random

# Protected operators. Return sentinel values rather than NaN/Inf so that
# partial solutions can still accumulate gradient-style signal through
# selection pressure.
_pdiv(a::Float32, b::Float32) = abs(b) < 1f-6 ? 1f0 : a / b
_plog(x::Float32) = x > 0f0 ? log(x) : 0f0
_psqrt(x::Float32) = sqrt(abs(x))

@testset "Nguyen symbolic regression suite (TreeGenome)" begin
    operators = OperatorEnum(;
        binary_operators=[+, -, *, _pdiv],
        unary_operators=[sin, cos, exp, _plog, _psqrt]
    )

    algorithm = GeneticProgramming(
        pop_size=100,
        generations=300,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    # run_one: standard 5-seed eval with a common gate.  Returns n_success.
    # gate_fitness defaults to 0.01 (canonical Nguyen success threshold,
    # stricter than Koza's 0.1 because the target functions here are
    # smaller in magnitude).
    function run_one(name::AbstractString, X::Matrix{Float32}, y::Vector{Float32};
                     gate_fitness::Float64 = 0.01,
                     n_seeds_required::Int = 3)
        evaluator = TreeFitnessEvaluator(X, y, operators)

        successes = map(1:5) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            result.best_fitness < gate_fitness
        end

        n_success = count(successes)
        println("  $name: $n_success/5 seeds converged (fitness < $gate_fitness)")
        flush(stdout)
        @test n_success >= n_seeds_required
    end

    # --- Univariate Nguyen (1-8) ---
    xs = Float32.(range(-1, 1, length=20))
    X1 = reshape(xs, 1, :)

    # x ∈ [0, 2] for Nguyen-7 (log(x+1), needs x > -1)
    xs_07 = Float32.(range(0, 2, length=20))
    X1_07 = reshape(xs_07, 1, :)

    # x ∈ [0, 4] for Nguyen-8 (sqrt, needs x ≥ 0)
    xs_08 = Float32.(range(0, 4, length=20))
    X1_08 = reshape(xs_08, 1, :)

    @testset "Nguyen-1: x^3 + x^2 + x" begin
        run_one("Nguyen-1", X1, xs .^ 3 .+ xs .^ 2 .+ xs)
    end

    @testset "Nguyen-2: x^4 + x^3 + x^2 + x" begin
        run_one("Nguyen-2", X1, xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs)
    end

    @testset "Nguyen-3: x^5 + x^4 + x^3 + x^2 + x" begin
        run_one("Nguyen-3", X1, xs .^ 5 .+ xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs)
    end

    @testset "Nguyen-4: x^6 + x^5 + x^4 + x^3 + x^2 + x" begin
        run_one("Nguyen-4", X1,
                xs .^ 6 .+ xs .^ 5 .+ xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs)
    end

    @testset "Nguyen-5: sin(x^2) * cos(x) - 1" begin
        run_one("Nguyen-5", X1, sin.(xs .^ 2) .* cos.(xs) .- 1f0)
    end

    @testset "Nguyen-6: sin(x) + sin(x + x^2)" begin
        run_one("Nguyen-6", X1, sin.(xs) .+ sin.(xs .+ xs .^ 2))
    end

    @testset "Nguyen-7: log(x+1) + log(x^2+1)" begin
        # Notoriously discriminating — relaxed gate (2/5 seeds).
        run_one("Nguyen-7", X1_07,
                log.(xs_07 .+ 1f0) .+ log.(xs_07 .^ 2 .+ 1f0);
                n_seeds_required=2)
    end

    @testset "Nguyen-8: sqrt(x)" begin
        run_one("Nguyen-8", X1_08, sqrt.(xs_08))
    end

    # --- Bivariate Nguyen (9, 10) ---
    # Canonical protocol samples 100 random points uniformly from [-1, 1]^2
    # with a fixed RNG so the benchmark is reproducible.
    rng = MersenneTwister(1234)
    n_2d = 100
    X2 = Float32.(2 .* rand(rng, 2, n_2d) .- 1)  # [-1, 1]^2
    xs2 = view(X2, 1, :)
    ys2 = view(X2, 2, :)

    @testset "Nguyen-9: sin(x) + sin(y^2)" begin
        run_one("Nguyen-9", X2, sin.(xs2) .+ sin.(ys2 .^ 2))
    end

    @testset "Nguyen-10: 2 * sin(x) * cos(y)" begin
        run_one("Nguyen-10", X2, 2f0 .* sin.(xs2) .* cos.(ys2))
    end
end
