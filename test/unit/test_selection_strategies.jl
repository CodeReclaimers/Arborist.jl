using Test
using Arborist
using Random

@testset "Selection strategies (roulette / rank / truncation)" begin
    # Toy 10-individual population: fitnesses 1.0, 2.0, ..., 10.0
    # (best = 1.0 at index 1, worst = 10.0 at index 10).
    fits = collect(1.0:10.0)

    @testset "needs_cases trait" begin
        @test !needs_cases(FitnessProportionateSelection())
        @test !needs_cases(RankSelection())
        @test !needs_cases(TruncationSelection())
    end

    @testset "FitnessProportionateSelection: best > worst" begin
        rng = MersenneTwister(0)
        s = FitnessProportionateSelection()
        counts = zeros(Int, 10)
        for _ in 1:20_000
            counts[select_parent(s, fits, nothing, rng)] += 1
        end
        # With fitnesses 1..10 and `eps=1e-12`, the best (f=1=f_min) has
        # weight ≈ 1e12 and overwhelms the rest. That's textbook behavior
        # for proportionate selection on minimization — correct but
        # aggressive. Don't over-assert distribution shape; just confirm
        # best is dominant and worst is rare.
        @test counts[1] > counts[10]
        @test counts[1] > 15_000   # best dominates the population
    end

    @testset "FitnessProportionateSelection: softer distribution on shifted fits" begin
        # Fitnesses in [10, 20]: f_min = 10, so weights are 1/(eps + 0..9).
        # Same ratio as fits 1..10 — confirms shift-invariance of the scheme.
        rng = MersenneTwister(10)
        s = FitnessProportionateSelection()
        shifted = collect(10.0:19.0)
        counts = zeros(Int, 10)
        for _ in 1:20_000
            counts[select_parent(s, shifted, nothing, rng)] += 1
        end
        @test counts[1] > counts[10]
    end

    @testset "FitnessProportionateSelection: all-identical uniform" begin
        rng = MersenneTwister(1)
        s = FitnessProportionateSelection()
        flat = fill(5.0, 10)
        counts = zeros(Int, 10)
        for _ in 1:10_000
            counts[select_parent(s, flat, nothing, rng)] += 1
        end
        # Uniform: each bucket ≈ 1000; allow wide tolerance.
        @test all(c -> 500 < c < 1500, counts)
    end

    @testset "FitnessProportionateSelection: all-Inf uniform fallback" begin
        rng = MersenneTwister(2)
        s = FitnessProportionateSelection()
        all_inf = fill(Inf, 10)
        counts = zeros(Int, 10)
        for _ in 1:5000
            counts[select_parent(s, all_inf, nothing, rng)] += 1
        end
        @test all(c -> 300 < c < 700, counts)
    end

    @testset "FitnessProportionateSelection: Inf individuals skipped" begin
        rng = MersenneTwister(3)
        s = FitnessProportionateSelection()
        mixed = [1.0, 2.0, Inf, 3.0, Inf]
        counts = zeros(Int, 5)
        for _ in 1:5000
            counts[select_parent(s, mixed, nothing, rng)] += 1
        end
        @test counts[3] == 0
        @test counts[5] == 0
        @test counts[1] + counts[2] + counts[4] == 5000
    end

    @testset "RankSelection: pressure=2 gives best >> worst" begin
        rng = MersenneTwister(4)
        s = RankSelection(; selection_pressure=2.0)
        counts = zeros(Int, 10)
        for _ in 1:20_000
            counts[select_parent(s, fits, nothing, rng)] += 1
        end
        # At sp=2, weight for worst (rank N) is 0 — never selected.
        @test counts[10] == 0
        # Best (rank 1, weight sp=2) should dominate; roughly 2/n = 20% share.
        @test 3000 < counts[1] < 5000
    end

    @testset "RankSelection: pressure=1 is uniform" begin
        rng = MersenneTwister(5)
        s = RankSelection(; selection_pressure=1.0)
        counts = zeros(Int, 10)
        for _ in 1:10_000
            counts[select_parent(s, fits, nothing, rng)] += 1
        end
        @test all(c -> 700 < c < 1300, counts)
    end

    @testset "RankSelection: validation" begin
        @test_throws ArgumentError RankSelection(; selection_pressure=0.9)
        @test_throws ArgumentError RankSelection(; selection_pressure=2.5)
    end

    @testset "TruncationSelection: only top-k selected" begin
        rng = MersenneTwister(6)
        s = TruncationSelection(; ratio=0.3)
        counts = zeros(Int, 10)
        for _ in 1:5000
            counts[select_parent(s, fits, nothing, rng)] += 1
        end
        # ceil(0.3 * 10) = 3, so indices 1, 2, 3 only.
        @test counts[1] + counts[2] + counts[3] == 5000
        @test all(c -> c == 0, counts[4:10])
    end

    @testset "TruncationSelection: ratio=1 is uniform" begin
        rng = MersenneTwister(7)
        s = TruncationSelection(; ratio=1.0)
        counts = zeros(Int, 10)
        for _ in 1:10_000
            counts[select_parent(s, fits, nothing, rng)] += 1
        end
        @test all(c -> 700 < c < 1300, counts)
    end

    @testset "TruncationSelection: validation" begin
        @test_throws ArgumentError TruncationSelection(; ratio=0.0)
        @test_throws ArgumentError TruncationSelection(; ratio=1.1)
        @test_throws ArgumentError TruncationSelection(; ratio=-0.5)
    end

    @testset "End-to-end smoke: TreeGenome solves with each strategy" begin
        # Run a tiny solve with each new strategy; just verify no errors
        # and the result has expected structure.
        using DynamicExpressions
        nguyen1 = Arborist.Benchmarks.nguyen(1)
        ops_spec = Arborist.Benchmarks.canonical_sr_operators(Float32)
        ops = OperatorEnum(binary_operators=ops_spec.binary,
                           unary_operators=ops_spec.unary)
        ev = TreeFitnessEvaluator(nguyen1.X, nguyen1.y, ops)
        problem = GPProblem(ev, TreeGenome{Float32}; seed=42)

        for sel in [FitnessProportionateSelection(),
                    RankSelection(; selection_pressure=1.7),
                    TruncationSelection(; ratio=0.4)]
            alg = GeneticProgramming(;
                pop_size=30, generations=5,
                mutation_rate=0.3, crossover_rate=0.3,
                elitism=2, parallel=false,
                selection=sel,
            )
            result = solve(problem, alg; verbose=false)
            @test length(result.population) == 30
            @test isfinite(result.best_fitness)
        end
    end
end
