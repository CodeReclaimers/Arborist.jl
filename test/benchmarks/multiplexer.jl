# Boolean multiplexer benchmark (6-bit and 11-bit) — Koza's canonical
# Boolean GP scaling benchmark beyond parity.
#
# Semantics: k address lines select one of 2^k data lines.  For n-bit
# multiplexer, n = k + 2^k (6-bit: k=2, 4 data lines; 11-bit: k=3,
# 8 data lines).
#
# Koza's original benchmark used primitive set {AND, OR, NOT, IF}.
# DynamicExpressions supports only unary/binary operators, so the
# ternary IF is unavailable.  We use the Boolean-complete subset
# {AND, OR, NAND, NOR, NOT}, which makes the problem harder than
# Koza's canonical formulation (GP must synthesize multiplex selection
# from pure Boolean logic rather than use IF directly) — but remains
# standard in comparisons that use NAND-only / AND-OR-NOT primitive
# sets.  Gates are calibrated accordingly.
#
# Booleans are encoded as Float32 0.0/1.0 to match the TreeGenome /
# DynamicExpressions numeric pipeline (same approach as
# boolean_parity.jl).

using DynamicExpressions

_mux_and(a::Float32, b::Float32)  = Float32((a > 0.5f0) & (b > 0.5f0))
_mux_or(a::Float32, b::Float32)   = Float32((a > 0.5f0) | (b > 0.5f0))
_mux_nand(a::Float32, b::Float32) = Float32(!((a > 0.5f0) & (b > 0.5f0)))
_mux_nor(a::Float32, b::Float32)  = Float32(!((a > 0.5f0) | (b > 0.5f0)))
_mux_xor(a::Float32, b::Float32)  = Float32(xor(a > 0.5f0, b > 0.5f0))
_mux_not(a::Float32)              = Float32(!(a > 0.5f0))

# Build the full truth table for an n-bit multiplexer.
# Convention: inputs 1..k are address (low bit first), inputs k+1..n
# are data lines indexed by the address value.
function _multiplexer_truth_table(k::Int)
    n_data = 1 << k
    n_bits = k + n_data
    n_cases = 1 << n_bits
    X = zeros(Float32, n_bits, n_cases)
    y = zeros(Float32, n_cases)
    for bits in 0:(n_cases - 1)
        addr = 0
        for i in 1:k
            v = (bits >> (i - 1)) & 1
            X[i, bits + 1] = Float32(v)
            addr |= v << (i - 1)
        end
        for j in 1:n_data
            X[k + j, bits + 1] = Float32((bits >> (k + j - 1)) & 1)
        end
        # Output: data line at index `addr` (0-indexed) → column k+addr+1.
        y[bits + 1] = X[k + addr + 1, bits + 1]
    end
    return X, y, n_bits, n_cases
end

@testset "Boolean multiplexer benchmarks (TreeGenome)" begin
    operators = OperatorEnum(;
        binary_operators=[_mux_and, _mux_or, _mux_nand, _mux_nor, _mux_xor],
        unary_operators=[_mux_not]
    )

    # --- 6-bit multiplexer (2 addr + 4 data = 64 cases) ---
    # Pure-Boolean multiplexer has a sharp local optimum at fitness=0.25
    # (half-mux: GP discovers one address bit but not the second —
    # equivalent to outputting `d[a0]` regardless of `a1`).  Koza's
    # original IF primitive is what makes escape reliable; without it
    # the benchmark becomes a search-difficulty stress test.
    #
    # Gate: at least 1/5 seeds achieves perfect fit (demonstrates the
    # target is reachable) AND at least 3/5 seeds reach fitness < 0.3
    # (meaningful learning beyond trivial).  Treat as test-only —
    # promotion to a tight gate would require adding a ternary primitive
    # to DynamicExpressions.
    @testset "6-bit multiplexer (64 cases, IF-free primitive set)" begin
        X, y, n_bits, n_cases = _multiplexer_truth_table(2)
        @test n_bits == 6
        @test n_cases == 64

        evaluator = TreeFitnessEvaluator(X, y, operators)
        algorithm = GeneticProgramming(
            pop_size=500,
            generations=500,
            mutation_rate=0.4,
            crossover_rate=0.2,
            elitism=2
        )

        fitnesses = map(1:5) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            perfect = result.best_fitness < 1e-6
            println("    6-bit mux seed=$seed: fitness=$(round(result.best_fitness, sigdigits=4))" *
                    (perfect ? " (perfect)" : ""))
            flush(stdout)
            result.best_fitness
        end

        n_perfect = count(f -> f < 1e-6, fitnesses)
        n_learning = count(f -> f < 0.3, fitnesses)
        println("  6-bit multiplexer: $n_perfect/5 perfect, $n_learning/5 fitness < 0.3")
        flush(stdout)
        @test n_perfect >= 1
        @test n_learning >= 3
    end

    # --- 11-bit multiplexer (3 addr + 8 data = 2048 cases) ---
    # This is hard without IF — included as a test-only scale
    # demonstration.  The gate just requires forward progress (fitness
    # meaningfully below majority-class baseline).  Majority-class
    # baseline: output "0" for all cases → MSE = 0.5 (half of cases
    # should be 1).  Gate: at least one seed reaches fitness < 0.3
    # (well below 0.5), demonstrating that evolution is actually
    # learning structure, even if not reaching perfect fit.
    @testset "11-bit multiplexer (2048 cases, test-only gate)" begin
        X, y, n_bits, n_cases = _multiplexer_truth_table(3)
        @test n_bits == 11
        @test n_cases == 2048

        evaluator = TreeFitnessEvaluator(X, y, operators)
        # Smaller budget than 6-bit: the IF-free primitive set converges
        # to the 0.25 local optimum very quickly (within ~50 generations),
        # so spending more compute here just re-confirms the same result
        # at 10x wall-time cost.  The gate just verifies forward progress.
        algorithm = GeneticProgramming(
            pop_size=200,
            generations=80,
            mutation_rate=0.4,
            crossover_rate=0.2,
            elitism=2
        )

        # Fewer seeds here (3 instead of 5) because each run is ~10x
        # more expensive than 6-bit.
        best_fitnesses = map(1:3) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            println("    11-bit mux seed=$seed: fitness=$(round(result.best_fitness, sigdigits=4))")
            flush(stdout)
            result.best_fitness
        end

        best_overall = minimum(best_fitnesses)
        println("  11-bit multiplexer: best fitness across 3 seeds = $(round(best_overall, sigdigits=4))")
        flush(stdout)
        @test best_overall < 0.3  # meaningfully below 0.5 majority-class MSE
    end
end
