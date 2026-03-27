# Lorenz attractor recovery — symbolic regression on a chaotic dynamical system.
#
# Given trajectory data from the Lorenz system, recover the derivative
# expressions from numerical data alone. Uses TreeGenome (DynamicExpressions.jl)
# for fast vectorized evaluation.
#
# The Lorenz system:
#   dx/dt = σ(y - x)         = 10y - 10x       (linear, easiest)
#   dy/dt = x(ρ - z) - y     = 28x - xz - y    (cross-term xz)
#   dz/dt = xy - βz          = xy - 2.667z      (cross-term xy)
#
# Standard parameters: σ=10, ρ=28, β=8/3

using DynamicExpressions
_std(v) = sqrt(sum((v .- sum(v)/length(v)).^2) / (length(v) - 1))

"""
Generate Lorenz trajectory data via RK4 integration.
Returns (X, dxdt, dydt, dzdt) where X is 3 × n_points and
dxdt/dydt/dzdt are normalized numerical derivatives (divided by their
standard deviation so that MSE of predicting zero ≈ 1.0).
"""
function _generate_lorenz_data(;
    σ::Float64=10.0, ρ::Float64=28.0, β::Float64=8/3,
    dt::Float64=0.005, n_steps::Int=2000, subsample::Int=20,
    x0::Float64=1.0, y0::Float64=1.0, z0::Float64=1.0)

    # Lorenz right-hand side
    function lorenz_rhs(x, y, z)
        (σ * (y - x), x * (ρ - z) - y, x * y - β * z)
    end

    # RK4 integration
    xs = Vector{Float64}(undef, n_steps)
    ys = Vector{Float64}(undef, n_steps)
    zs = Vector{Float64}(undef, n_steps)
    xs[1], ys[1], zs[1] = x0, y0, z0

    for i in 1:(n_steps - 1)
        x, y, z = xs[i], ys[i], zs[i]

        k1x, k1y, k1z = lorenz_rhs(x, y, z)
        k2x, k2y, k2z = lorenz_rhs(x + 0.5dt*k1x, y + 0.5dt*k1y, z + 0.5dt*k1z)
        k3x, k3y, k3z = lorenz_rhs(x + 0.5dt*k2x, y + 0.5dt*k2y, z + 0.5dt*k2z)
        k4x, k4y, k4z = lorenz_rhs(x + dt*k3x, y + dt*k3y, z + dt*k3z)

        xs[i+1] = x + (dt/6) * (k1x + 2k2x + 2k3x + k4x)
        ys[i+1] = y + (dt/6) * (k1y + 2k2y + 2k3y + k4y)
        zs[i+1] = z + (dt/6) * (k1z + 2k2z + 2k3z + k4z)
    end

    # Numerical derivatives via central differences (interior points)
    n_interior = n_steps - 2
    dxdt = Vector{Float64}(undef, n_interior)
    dydt = Vector{Float64}(undef, n_interior)
    dzdt = Vector{Float64}(undef, n_interior)
    for i in 1:n_interior
        dxdt[i] = (xs[i+2] - xs[i]) / (2 * dt)
        dydt[i] = (ys[i+2] - ys[i]) / (2 * dt)
        dzdt[i] = (zs[i+2] - zs[i]) / (2 * dt)
    end

    # Subsample to keep evaluation fast
    indices = 1:subsample:n_interior
    n_points = length(indices)

    X = Matrix{Float32}(undef, 3, n_points)
    for (j, i) in enumerate(indices)
        X[1, j] = Float32(xs[i+1])  # +1 to align with central diff
        X[2, j] = Float32(ys[i+1])
        X[3, j] = Float32(zs[i+1])
    end

    # Normalize derivatives by their standard deviation so that
    # MSE of predicting zero ≈ 1.0, making fitness thresholds
    # interpretable across different trajectory lengths and parameters.
    dx_sub = dxdt[indices]
    dy_sub = dydt[indices]
    dz_sub = dzdt[indices]
    dx_std = max(_std(dx_sub), 1e-8)
    dy_std = max(_std(dy_sub), 1e-8)
    dz_std = max(_std(dz_sub), 1e-8)

    return (X,
            Float32.(dx_sub ./ dx_std),
            Float32.(dy_sub ./ dy_std),
            Float32.(dz_sub ./ dz_std))
end

@testset "Lorenz attractor recovery (TreeGenome)" begin
    X, dxdt, dydt, dzdt = _generate_lorenz_data()
    n_points = size(X, 2)
    println("  Lorenz data: $(n_points) training points, 3 features (x, y, z)")
    flush(stdout)

    operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])

    algorithm = GeneticProgramming(
        pop_size=200,
        generations=500,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    # --- dx/dt = σ(y - x) = 10y - 10x ---
    # Linear combination — the easiest of the three.
    @testset "dx/dt = σ(y - x)" begin
        evaluator = TreeFitnessEvaluator(X, dxdt, operators)

        results = map(1:5) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            (fitness=result.best_fitness, wall_time=result.wall_time)
        end

        fitnesses = [r.fitness for r in results]
        n_success = count(f -> f < 0.1, fitnesses)
        best = minimum(fitnesses)
        println("  dx/dt: $n_success/5 converged (< 0.1), best=$(round(best, digits=4))")
        flush(stdout)
        @test n_success >= 3
    end

    # --- dy/dt = x(ρ - z) - y = 28x - xz - y ---
    # Requires the cross-term xz. Hardest of the three.
    @testset "dy/dt = x(ρ - z) - y" begin
        evaluator = TreeFitnessEvaluator(X, dydt, operators)

        results = map(1:5) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            (fitness=result.best_fitness, wall_time=result.wall_time)
        end

        fitnesses = [r.fitness for r in results]
        n_success = count(f -> f < 0.1, fitnesses)
        best = minimum(fitnesses)
        println("  dy/dt: $n_success/5 converged (< 0.1), best=$(round(best, digits=4))")
        flush(stdout)
        # Cross-term xz is the hardest — require ≥2/5 (better than chance)
        @test n_success >= 2
    end

    # --- dz/dt = xy - βz ---
    # Requires the cross-term xy.
    @testset "dz/dt = xy - βz" begin
        evaluator = TreeFitnessEvaluator(X, dzdt, operators)

        results = map(1:5) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            (fitness=result.best_fitness, wall_time=result.wall_time)
        end

        fitnesses = [r.fitness for r in results]
        n_success = count(f -> f < 0.1, fitnesses)
        best = minimum(fitnesses)
        println("  dz/dt: $n_success/5 converged (< 0.1), best=$(round(best, digits=4))")
        flush(stdout)
        # Cross-term xy — require ≥2/5
        @test n_success >= 2
    end
end
