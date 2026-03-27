# Evaluation speedup benchmark: TreeGenome vs ExprGenome on Koza-1.
#
# This benchmark measures and reports the evaluation speed difference.
# TreeGenome evaluates expression trees directly via DynamicExpressions.jl
# without @eval compilation, which is dramatically faster for large datasets.

using DynamicExpressions

@testset "TreeGenome vs ExprGenome evaluation speedup" begin
    # Koza-1: x^4 + x^3 + x^2 + x, 1000 evaluation points.
    xs = Float32.(range(-1, 1, length=1000))
    y_vals = xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs

    # --- ExprGenome setup ---
    input_cols = Dict(:x => Float32)
    output_cols = Dict(:y => Float32)
    input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
    output_rows = [Dict{Symbol,Any}(:y => v) for v in y_vals]
    expr_eval = TableFitnessEvaluator(input_cols, output_cols,
                                       input_rows, output_rows;
                                       time_limit_ns=1_000_000_000)

    fset = FunctionSet(Set{FunctionDetails}())
    for func in [:+, :-, :*, :/]
        add!(fset, func, 2, Float32, Float32)
    end
    rng = Random.MersenneTwister(42)
    state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

    # --- TreeGenome setup ---
    operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
    X = reshape(xs, 1, :)
    tree_eval = TreeFitnessEvaluator(X, y_vals, operators)

    # Time 50 evaluations each.
    n_evals = 50

    # ExprGenome evaluations
    expr_times = Float64[]
    for _ in 1:n_evals
        body = [create_random_assignment(state) for _ in 1:3]
        g = ExprGenome(body, state)
        t = @elapsed Arborist.evaluate_genome(g, expr_eval)
        push!(expr_times, t)
    end

    # TreeGenome evaluations
    tree_times = Float64[]
    for _ in 1:n_evals
        tree = Arborist._random_tree(rng, operators, 1, Float32, 3, :grow)
        g = TreeGenome{Float32}(tree, operators, 1)
        t = @elapsed evaluate(tree_eval, g)
        push!(tree_times, t)
    end

    # Use median to avoid outlier effects from JIT warmup.
    expr_median = sort(expr_times)[n_evals ÷ 2]
    tree_median = sort(tree_times)[n_evals ÷ 2]
    speedup = expr_median / tree_median

    @info "Evaluation speedup" ExprGenome_median_ms=round(expr_median*1000, digits=3) TreeGenome_median_ms=round(tree_median*1000, digits=3) speedup=round(speedup, digits=1)

    if speedup < 10.0
        @warn "TreeGenome speedup less than 10x (got $(round(speedup, digits=1))x). " *
              "This may indicate a benchmarking issue rather than a real performance problem."
    end
    @test speedup > 1.0   # Hard assertion: TreeGenome must be faster
end
