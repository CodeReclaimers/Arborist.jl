# nsga2_regression.jl — Multi-objective symbolic regression with NSGA-II
#
# Evolves expressions for y = x^2 + x with two objectives:
#   1. Prediction accuracy (MSE)
#   2. Expression complexity (node count)
#
# The Pareto front shows the tradeoff: simple expressions (e.g., "x1")
# have high MSE but low complexity, while complex expressions can achieve
# near-zero MSE. NSGA-II finds the full tradeoff surface automatically,
# unlike bloat_penalty which collapses it to a single compromise.
#
# Usage:
#   julia --project=. examples/nsga2_regression.jl

using Arborist
using DynamicExpressions: OperatorEnum
using Printf

# --- Data: y = x^2 + x over [-2, 2] ---
xs = Float32.(range(-2, 2, length=30))
X = reshape(xs, 1, :)
y = xs .^ 2 .+ xs

# --- Operators ---
operators = OperatorEnum(;
    binary_operators=[+, -, *, /],
    unary_operators=[sin, cos, abs]
)

# --- Multi-objective evaluator: MSE + complexity ---
inner_evaluator = TreeFitnessEvaluator(X, y, operators)
evaluator = ParsimonyEvaluator(inner_evaluator)

# --- Problem and algorithm ---
problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
algorithm = NSGAII(
    pop_size=200,
    generations=100,
    mutation_rate=0.3,
    crossover_rate=0.3,
    parallel=false
)

println("Running NSGA-II: pop=$(algorithm.pop_size), gen=$(algorithm.generations)")
println("Objectives: $(objective_names(evaluator))")
println()

result = solve(problem, algorithm; verbose=true)

# --- Print Pareto front (deduplicated) ---
println()
println("=" ^ 70)

# Deduplicate by (MSE rounded to 6 digits, complexity).
seen = Set{Tuple{Float64, Float64}}()
unique_indices = Int[]
for (i, fit) in enumerate(result.pareto_fitnesses)
    key = (round(fit[1], digits=6), fit[2])
    if key ∉ seen
        push!(seen, key)
        push!(unique_indices, i)
    end
end

println("Pareto front: $(length(result.pareto_front)) solutions, $(length(unique_indices)) unique")
println("=" ^ 70)
println()
println(lpad("MSE", 12), "  ", lpad("Nodes", 6), "  ", "Expression")
println("-" ^ 70)
for i in unique_indices
    genome = result.pareto_front[i]
    fit = result.pareto_fitnesses[i]
    mse_str = fit[1] < 0.001 ? @sprintf("%.2e", fit[1]) : @sprintf("%.6f", fit[1])
    expr = serialize(genome)
    if length(expr) > 40
        expr = expr[1:37] * "..."
    end
    println(lpad(mse_str, 12), "  ", lpad(Int(fit[2]), 6), "  ", expr)
end

println()
println("Wall time: $(round(result.wall_time, digits=2))s")
println("Final hypervolume: $(round(result.hypervolume_history[end], digits=4))")
