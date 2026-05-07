using Test
using Arborist
using Random
using DynamicExpressions

@testset "MAP-Elites" begin
    @testset "config validation" begin
        # mismatched feature_bounds vs n_bins
        @test_throws ArgumentError MAPElites(
            feature_fn = g -> (0.0, 0.0),
            feature_bounds = [(0.0, 1.0)],  # 1-dim
            n_bins = [5, 5],                # 2-dim
            mutation_ops = AbstractMutationOperator[SubtreeMutation()],
        )
        # n_bins must be positive
        @test_throws ArgumentError MAPElites(
            feature_fn = g -> (0.0,),
            feature_bounds = [(0.0, 1.0)],
            n_bins = [0],
            mutation_ops = AbstractMutationOperator[SubtreeMutation()],
        )
        # crossover_rate out of range
        @test_throws ArgumentError MAPElites(
            feature_fn = g -> (0.0,),
            feature_bounds = [(0.0, 1.0)],
            n_bins = [3],
            mutation_ops = AbstractMutationOperator[SubtreeMutation()],
            crossover_rate = 1.5,
        )
    end

    @testset "_cell_index discretization" begin
        bounds = [(0.0, 10.0), (-1.0, 1.0)]
        n_bins = [5, 4]
        # Low corner -> (1, 1)
        @test Arborist._cell_index((0.0, -1.0), bounds, n_bins) == (1, 1)
        # High corner clamps to last bin.
        @test Arborist._cell_index((10.0, 1.0), bounds, n_bins) == (5, 4)
        # Below-range clamps to 1.
        @test Arborist._cell_index((-5.0, -5.0), bounds, n_bins) == (1, 1)
        # Above-range clamps to last.
        @test Arborist._cell_index((20.0, 5.0), bounds, n_bins) == (5, 4)
        # Interior: 5.0 is center of [0, 10] with 5 bins -> bin 3.
        @test Arborist._cell_index((5.0, 0.0), bounds, n_bins)[1] == 3
    end

    @testset "_cell_index rejects feature dimension mismatch" begin
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        n_bins = [2, 2]
        @test_throws ArgumentError Arborist._cell_index((0.5,), bounds, n_bins)
        @test_throws ArgumentError Arborist._cell_index((0.5, 0.5, 0.5), bounds, n_bins)
    end

    @testset "coverage and qd_score" begin
        archive = MAPElitesArchive{Int}([3, 3])
        @test coverage(archive) == 0.0
        archive.grid[(1, 1)] = Pair(0, 0.5)
        archive.grid[(2, 2)] = Pair(0, 0.3)
        @test coverage(archive) ≈ 2.0 / 9.0
        # qd_score with worst=1.0: sum((1.0 - 0.5) + (1.0 - 0.3)) = 1.2
        @test qd_score(archive; worst_fitness=1.0) ≈ 1.2
    end

    @testset "MAP-Elites end-to-end (TreeGenome SR feature grid)" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        X = reshape(collect(Float32, -1.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]  # y = x^2

        # Feature grid: (tree size, tree depth). MAP-Elites should fill cells
        # across a range of sizes/depths rather than converging to a single
        # structure. Bin small, count actual unique cells.
        fp_fn = g -> (Float64(count_nodes(g.tree)), Float64(count_depth(g.tree)))

        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)

        alg = MAPElites(
            feature_fn = fp_fn,
            feature_bounds = [(1.0, 15.0), (0.0, 6.0)],
            n_bins = [5, 4],
            mutation_ops = AbstractMutationOperator[SubtreeMutation(), PointMutation()],
            crossover_ops = AbstractCrossoverOperator[SubtreeCrossover()],
            generations = 20,
            batch_size = 25,
            n_init = 40,
            crossover_rate = 0.4,
            parallel = false,
        )

        result = solve(problem, alg)
        @test result isa MAPElitesResult{TreeGenome{Float32}}
        @test length(result.coverage_history) >= 1
        @test length(result.qd_score_history) >= 1
        # Archive should hold several distinct structure cells.
        @test length(result.archive) >= 3
        # best_fitness should be finite and non-negative (MSE).
        @test isfinite(result.best_fitness)
        @test result.best_fitness >= 0.0
        # Coverage monotonically non-decreasing over time.
        @test result.coverage_history[end] >= result.coverage_history[1]
    end

    @testset "MAP-Elites reproducible with fixed seed" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        X = reshape(collect(Float32, -1.0:0.2:1.0), 1, :)
        y = X[1, :] .* X[1, :]

        function _run()
            evaluator = TreeFitnessEvaluator(X, y, ops)
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=7)
            alg = MAPElites(
                feature_fn = g -> (Float64(count_nodes(g.tree)),),
                feature_bounds = [(1.0, 15.0)],
                n_bins = [5],
                mutation_ops = AbstractMutationOperator[SubtreeMutation()],
                generations = 8, batch_size = 15, n_init = 20,
                parallel = false,
            )
            return solve(problem, alg)
        end

        r1 = _run()
        r2 = _run()
        @test length(r1.archive) == length(r2.archive)
        @test r1.best_fitness == r2.best_fitness
    end
end
