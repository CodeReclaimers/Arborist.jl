#!/usr/bin/env julia
# examples/lorenz_recovery.jl — Recover the Lorenz system from trajectory data.
#
# Demonstrates symbolic regression on a chaotic dynamical system using
# Arborist.jl's TreeGenome (backed by DynamicExpressions.jl).
#
# The Lorenz system:
#   dx/dt = σ(y - x)         = 10y - 10x
#   dy/dt = x(ρ - z) - y     = 28x - xz - y
#   dz/dt = xy - βz          = xy - 2.667z
#
# Given only trajectory data (x, y, z) and numerical derivatives, GP
# attempts to recover these expressions from scratch.
#
# Usage:
#   julia --project examples/lorenz_recovery.jl
#   julia --project examples/lorenz_recovery.jl --seed=42 --generations=300

using Arborist
using DynamicExpressions

# =============================================================================
# Lorenz data generation (Euler integration for simplicity)
# =============================================================================

function generate_lorenz_data(;
    sigma=10.0, rho=28.0, beta=8/3,
    dt=0.001, n_steps=5000, subsample=50,
    x0=1.0, y0=1.0, z0=1.0)

    xs = Vector{Float64}(undef, n_steps)
    ys = Vector{Float64}(undef, n_steps)
    zs = Vector{Float64}(undef, n_steps)
    xs[1], ys[1], zs[1] = x0, y0, z0

    for i in 1:(n_steps - 1)
        x, y, z = xs[i], ys[i], zs[i]
        xs[i+1] = x + dt * sigma * (y - x)
        ys[i+1] = y + dt * (x * (rho - z) - y)
        zs[i+1] = z + dt * (x * y - beta * z)
    end

    # Central-difference numerical derivatives
    n_interior = n_steps - 2
    dxdt = [(xs[i+2] - xs[i]) / (2dt) for i in 1:n_interior]
    dydt = [(ys[i+2] - ys[i]) / (2dt) for i in 1:n_interior]
    dzdt = [(zs[i+2] - zs[i]) / (2dt) for i in 1:n_interior]

    # Subsample
    idx = 1:subsample:n_interior
    n = length(idx)

    X = Matrix{Float32}(undef, 3, n)
    for (j, i) in enumerate(idx)
        X[1, j] = Float32(xs[i+1])
        X[2, j] = Float32(ys[i+1])
        X[3, j] = Float32(zs[i+1])
    end

    return (X       = X,
            dxdt    = Float32.(dxdt[idx]),
            dydt    = Float32.(dydt[idx]),
            dzdt    = Float32.(dzdt[idx]),
            n_points = n)
end

# =============================================================================
# Fit statistics
# =============================================================================

function fit_stats(predictions::Vector{Float32}, targets::Vector{Float32})
    n = length(targets)
    residuals = predictions .- targets
    mse = sum(residuals .^ 2) / n
    rmse = sqrt(mse)
    ss_res = sum(residuals .^ 2)
    mean_y = sum(targets) / n
    ss_tot = sum((targets .- mean_y) .^ 2)
    r_squared = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    max_err = maximum(abs.(residuals))
    return (; mse, rmse, r_squared, max_err)
end

# =============================================================================
# Parse CLI arguments
# =============================================================================

global seed = 42
global generations = 500
global pop_size = 200

for arg in ARGS
    if startswith(arg, "--seed=")
        global seed = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--generations=")
        global generations = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--pop_size=")
        global pop_size = parse(Int, split(arg, "=")[2])
    end
end

# =============================================================================
# Main
# =============================================================================

println("=" ^ 70)
println("Lorenz Attractor Recovery via Symbolic Regression")
println("=" ^ 70)
println()
println("True system:")
println("  dx/dt = 10(y - x)")
println("  dy/dt = x(28 - z) - y    = 28x - xz - y")
println("  dz/dt = xy - (8/3)z      = xy - 2.667z")
println()

data = generate_lorenz_data()
println("Training data: $(data.n_points) points from Euler integration (dt=0.001, 5000 steps)")
println("Features: x (x1), y (x2), z (x3)")
println()

operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])

algorithm = GeneticProgramming(
    pop_size=pop_size,
    generations=generations,
    mutation_rate=0.4,
    crossover_rate=0.2,
    elitism=2,
    bloat_penalty=0.005
)

# Component labels and true expressions for comparison
components = [
    ("dx/dt", "10*x2 - 10*x1",    data.dxdt),
    ("dy/dt", "28*x1 - x1*x3 - x2", data.dydt),
    ("dz/dt", "x1*x2 - 2.667*x3",   data.dzdt),
]

println("Algorithm: pop_size=$pop_size, generations=$generations, seed=$seed")
println("-" ^ 70)
println()

for (name, true_expr, targets) in components
    println("--- $name ---")
    println("  True:  $true_expr")
    flush(stdout)

    evaluator = TreeFitnessEvaluator(data.X, targets, operators)
    problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)

    t0 = time()
    result = solve(problem, algorithm; verbose=false)
    elapsed = time() - t0

    # Get the evolved expression as a string
    evolved_expr = serialize(result.best_genome)
    if length(evolved_expr) > 200
        evolved_expr = evolved_expr[1:197] * "..."
    end

    # Compute predictions for fit statistics
    preds = try
        Float32.(result.best_genome.tree(data.X, operators))
    catch
        fill(Float32(NaN), length(targets))
    end
    stats = fit_stats(preds, targets)

    n_nodes = complexity(result.best_genome)
    println("  Found: $evolved_expr")
    println("  Nodes:     $(Int(n_nodes))")
    println("  MSE:       $(round(stats.mse, digits=6))")
    println("  RMSE:      $(round(stats.rmse, digits=6))")
    println("  R-squared: $(round(stats.r_squared, digits=6))")
    println("  Max error: $(round(stats.max_err, digits=4))")
    println("  Wall time: $(round(elapsed, digits=1))s")
    println()
    flush(stdout)
end

println("=" ^ 70)
println("Note: x1=x, x2=y, x3=z in DynamicExpressions notation.")
println("A perfect recovery has R-squared = 1.0 and MSE = 0.0.")
