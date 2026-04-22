using Test
using Arborist
using Random

@testset "Lexicase selection" begin
    @testset "needs_cases trait" begin
        @test needs_cases(TournamentSelection(3)) == false
        @test needs_cases(LexicaseSelection()) == true
        @test needs_cases(EpsilonLexicaseSelection()) == true
        @test needs_cases(EpsilonLexicaseSelection(epsilon=0.5)) == true
    end

    @testset "pure lexicase picks the dominator" begin
        # 4 individuals, 4 cases. Individual 1 has the uniquely best loss on
        # every case, so lexicase must select it regardless of case order.
        case_fitnesses = [
            [0.0, 0.0, 0.0, 0.0],   # dominator
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
        sel_fit = [0.0, 1.0, 2.0, 3.0]
        rng = MersenneTwister(1)
        hits = 0
        for _ in 1:500
            idx = Arborist.select_parent(LexicaseSelection(), sel_fit, case_fitnesses, rng)
            hits += (idx == 1)
        end
        @test hits == 500
    end

    @testset "pure lexicase is a specialist preserver" begin
        # 3 individuals, 3 cases. No dominator — each specialist is best on
        # exactly one case. Lexicase should pick each with positive probability.
        case_fitnesses = [
            [0.0, 1.0, 1.0],  # best on case 1
            [1.0, 0.0, 1.0],  # best on case 2
            [1.0, 1.0, 0.0],  # best on case 3
        ]
        sel_fit = [0.67, 0.67, 0.67]
        rng = MersenneTwister(42)
        counts = zeros(Int, 3)
        for _ in 1:3000
            idx = Arborist.select_parent(LexicaseSelection(), sel_fit, case_fitnesses, rng)
            counts[idx] += 1
        end
        # Equal cases → each should appear roughly 1/3 of the time. Expect
        # all three > 500 (with 3000 trials, 3-sigma on 1000-count Bernoulli
        # ≈ 66, so 500 is well outside noise).
        @test all(counts .> 500)
    end

    @testset "epsilon-lexicase respects threshold" begin
        # 4 individuals: 2 within epsilon of best on every case, 2 far outside.
        case_fitnesses = [
            [0.0, 0.0, 0.0],     # best
            [0.05, 0.05, 0.05],  # within eps=0.1
            [0.5, 0.5, 0.5],     # outside eps=0.1
            [1.0, 1.0, 1.0],     # outside eps=0.1
        ]
        sel_fit = [0.0, 0.05, 0.5, 1.0]
        rng = MersenneTwister(7)
        counts = zeros(Int, 4)
        for _ in 1:2000
            idx = Arborist.select_parent(EpsilonLexicaseSelection(epsilon=0.1),
                                         sel_fit, case_fitnesses, rng)
            counts[idx] += 1
        end
        # Only the first two should be selectable; the others must be zero.
        @test counts[3] == 0
        @test counts[4] == 0
        @test counts[1] + counts[2] == 2000
    end

    @testset "auto-epsilon (MAD) activates when epsilon=0" begin
        # Two specialists with equal deviations; MAD-based threshold should
        # include both even though their aggregates differ.
        case_fitnesses = [
            [0.0, 1.0],   # specialist on case 1
            [1.0, 0.0],   # specialist on case 2
        ]
        sel_fit = [0.5, 0.5]
        rng = MersenneTwister(11)
        counts = [0, 0]
        for _ in 1:2000
            idx = Arborist.select_parent(EpsilonLexicaseSelection(),  # epsilon=0 → auto
                                         sel_fit, case_fitnesses, rng)
            counts[idx] += 1
        end
        # Both should be selectable.
        @test counts[1] > 500
        @test counts[2] > 500
    end

    @testset "lexicase throws on nothing case_fitnesses" begin
        @test_throws ArgumentError Arborist.select_parent(
            LexicaseSelection(), [0.0, 1.0], nothing, MersenneTwister(1))
        @test_throws ArgumentError Arborist.select_parent(
            EpsilonLexicaseSelection(), [0.0, 1.0], nothing, MersenneTwister(1))
    end

    @testset "tournament select_parent ignores case_fitnesses" begin
        sel_fit = [3.0, 1.0, 2.0, 4.0]
        rng = MersenneTwister(99)
        # Tournament samples with replacement, so even size=4 is not
        # guaranteed to cover the whole population. We assert only that
        # index 2 (the best) wins most of the time, and that passing
        # either `nothing` or a matrix for case_fitnesses gives identical
        # behavior (tournament must ignore the second argument).
        counts = zeros(Int, 4)
        for _ in 1:2000
            idx = Arborist.select_parent(TournamentSelection(4), sel_fit, nothing, rng)
            counts[idx] += 1
        end
        @test argmax(counts) == 2  # best fitness → most frequently selected
        @test counts[2] > counts[1]
        @test counts[2] > counts[3]
        @test counts[2] > counts[4]

        # Passing case_fitnesses must not change tournament behavior.
        cf = [[0.0, 1.0] for _ in 1:4]
        rng2 = MersenneTwister(99)
        counts2 = zeros(Int, 4)
        for _ in 1:2000
            idx = Arborist.select_parent(TournamentSelection(4), sel_fit, cf, rng2)
            counts2[idx] += 1
        end
        @test argmax(counts2) == 2
    end

    @testset "lexicase GP end-to-end (TreeGenome)" begin
        using DynamicExpressions
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        X = reshape(collect(Float32, -1.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        alg = GeneticProgramming(pop_size=30, generations=10,
                                 parallel=false,
                                 selection=LexicaseSelection())
        result = solve(problem, alg)
        @test isfinite(result.best_fitness)
        @test result.generations_run == 10
    end

    @testset "epsilon-lexicase GP end-to-end (TreeGenome)" begin
        using DynamicExpressions
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        X = reshape(collect(Float32, -1.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        alg = GeneticProgramming(pop_size=30, generations=10,
                                 parallel=false,
                                 selection=EpsilonLexicaseSelection())
        result = solve(problem, alg)
        @test isfinite(result.best_fitness)
        @test result.generations_run == 10
    end
end
