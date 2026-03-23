# Boolean even parity (4-bit) benchmark — uses TreeGenome (DynamicExpressions.jl).
#
# Migrated from 2-bit ExprGenome in Phase 4. 4-bit parity was previously
# infeasible with ExprGenome due to @eval compilation cost (~11 minutes/seed).
# TreeGenome's vectorized evaluation makes 4-bit parity tractable.
#
# Even parity: return 1.0 iff an even number of inputs are 1.0.
# We encode Bool as Float32 (0.0 = false, 1.0 = true) since TreeGenome
# operates on numeric types. Fitness = MSE over all 16 input combinations.

using DynamicExpressions
const _ParityDynExt = Base.get_extension(Arborist, :DynExprExt)
const _ParityTreeGenome = _ParityDynExt.TreeGenome
const _ParityTreeEval = _ParityDynExt.TreeFitnessEvaluator

# Boolean operators that work on Float32 (0.0/1.0 encoded)
_f32_and(a::Float32, b::Float32) = Float32((a > 0.5f0) & (b > 0.5f0))
_f32_or(a::Float32, b::Float32) = Float32((a > 0.5f0) | (b > 0.5f0))
_f32_nand(a::Float32, b::Float32) = Float32(!((a > 0.5f0) & (b > 0.5f0)))
_f32_nor(a::Float32, b::Float32) = Float32(!((a > 0.5f0) | (b > 0.5f0)))
_f32_xor(a::Float32, b::Float32) = Float32(xor(a > 0.5f0, b > 0.5f0))
_f32_not(a::Float32) = Float32(!(a > 0.5f0))

@testset "Boolean even parity (4-bit, TreeGenome)" begin
    n_bits = 4
    operators = OperatorEnum(;
        binary_operators=[_f32_and, _f32_or, _f32_nand, _f32_nor, _f32_xor],
        unary_operators=[_f32_not]
    )

    # Generate all 2^4 = 16 input combinations as Float32 columns.
    n_cases = 2^n_bits
    X = zeros(Float32, n_bits, n_cases)
    y = zeros(Float32, n_cases)
    for bits in 0:(n_cases - 1)
        n_true = 0
        for i in 1:n_bits
            val = (bits >> (i - 1)) & 1
            X[i, bits + 1] = Float32(val)
            n_true += val
        end
        y[bits + 1] = Float32(n_true % 2 == 0)
    end

    evaluator = _ParityTreeEval(X, y, operators)

    algorithm = GeneticProgramming(
        pop_size=200,
        generations=500,
        mutation_rate=0.4,
        crossover_rate=0.3,
        elitism=2,
        tournament_size=5
    )

    # 4-bit parity is a hard benchmark. We require at least 1/5 seeds to reach
    # 90% accuracy (fitness <= 0.1). Higher convergence rates require larger
    # populations or more generations.
    successes = map(1:5) do seed
        problem = GPProblem(evaluator, _ParityTreeGenome{Float32}; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        println("    Parity seed=$seed: fitness=$(round(result.best_fitness, digits=4))")
        flush(stdout)
        result.best_fitness <= 0.1
    end

    n_success = count(successes)
    println("  Boolean parity 4-bit (TreeGenome): $n_success/5 seeds converged (fitness <= 0.1)")
    flush(stdout)
    @test n_success >= 1
end
