using Test
using Arborist
using Random
using DynamicExpressions

@testset "HallOfFame archive" begin
    @testset "construction and basic ops" begin
        hof = HallOfFame{TreeGenome{Float32}}(5)
        @test length(hof) == 0
        @test hof.capacity == 5
        @test isempty(hof)
        @test_throws ArgumentError HallOfFame{TreeGenome{Float32}}(0)
        @test_throws ArgumentError HallOfFame{TreeGenome{Float32}}(-1)
    end

    @testset "push preserves sorted order and capacity" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g = (v) -> TreeGenome{Float32}(Node{Float32}(; val=Float32(v)), ops, 1)
        hof = HallOfFame{TreeGenome{Float32}}(3)

        # Insert in mixed order.
        push!(hof, g(1.0), 5.0)
        push!(hof, g(2.0), 1.0)
        push!(hof, g(3.0), 3.0)
        @test length(hof) == 3
        @test hof.fitnesses == [1.0, 3.0, 5.0]  # sorted ascending

        # Insert a value that's better than the worst (5.0).
        push!(hof, g(4.0), 2.0)
        @test length(hof) == 3
        @test hof.fitnesses == [1.0, 2.0, 3.0]

        # Insert a value worse than current worst — rejected.
        push!(hof, g(5.0), 100.0)
        @test length(hof) == 3
        @test hof.fitnesses == [1.0, 2.0, 3.0]
    end

    @testset "push rejects non-finite fitness" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g = TreeGenome{Float32}(Node{Float32}(; val=0.0f0), ops, 1)
        hof = HallOfFame{TreeGenome{Float32}}(5)
        push!(hof, g, Inf)
        push!(hof, g, NaN)
        push!(hof, g, -Inf)
        @test length(hof) == 0
    end

    @testset "push deduplicates on fitness tolerance" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g = (v) -> TreeGenome{Float32}(Node{Float32}(; val=Float32(v)), ops, 1)
        hof = HallOfFame{TreeGenome{Float32}}(5)
        push!(hof, g(1.0), 1.0)
        # Second push with same fitness (within 1e-12) — rejected.
        push!(hof, g(2.0), 1.0)
        push!(hof, g(3.0), 1.0 + 1e-13)  # also within tol
        @test length(hof) == 1
        # Slightly different fitness — accepted.
        push!(hof, g(4.0), 1.1)
        @test length(hof) == 2
    end

    @testset "iteration and getindex" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g = (v) -> TreeGenome{Float32}(Node{Float32}(; val=Float32(v)), ops, 1)
        hof = HallOfFame{TreeGenome{Float32}}(10)
        for (gv, f) in [(1.0, 3.0), (2.0, 1.0), (3.0, 2.0)]
            push!(hof, g(gv), f)
        end
        collected = [item for item in hof]
        @test length(collected) == 3
        @test hof[1].tree.val ≈ Float32(2.0)  # fitness 1.0 (best)
    end

    @testset "end-to-end: solve populates hall_of_fame" begin
        nguyen1 = Arborist.Benchmarks.nguyen(1)
        ops_spec = Arborist.Benchmarks.canonical_sr_operators(Float32)
        ops = OperatorEnum(binary_operators=ops_spec.binary,
                           unary_operators=ops_spec.unary)
        ev = TreeFitnessEvaluator(nguyen1.X, nguyen1.y, ops)
        problem = GPProblem(ev, TreeGenome{Float32}; seed=42)

        # Default (no HoF).
        result_nil = solve(problem, GeneticProgramming(;
            pop_size=20, generations=3, parallel=false))
        @test result_nil.hall_of_fame === nothing

        # With HoF.
        result = solve(problem, GeneticProgramming(;
            pop_size=30, generations=8, parallel=false);
            hall_of_fame_size=15)
        hof = result.hall_of_fame
        @test hof !== nothing
        @test hof isa HallOfFame
        @test 1 <= length(hof) <= 15
        # Sorted ascending, all finite.
        @test all(isfinite, hof.fitnesses)
        @test issorted(hof.fitnesses)
        # Best in hof should equal result.best_fitness (or be strictly better,
        # since the HoF tracks all-time not just final).
        @test hof.fitnesses[1] <= result.best_fitness + 1e-9
    end

    @testset "solve seeds hall_of_fame from initial population" begin
        nguyen1 = Arborist.Benchmarks.nguyen(1)
        ops_spec = Arborist.Benchmarks.canonical_sr_operators(Float32)
        ops = OperatorEnum(binary_operators=ops_spec.binary,
                           unary_operators=ops_spec.unary)
        ev = TreeFitnessEvaluator(nguyen1.X, nguyen1.y, ops)
        problem = GPProblem(ev, TreeGenome{Float32}; seed=7)

        result = solve(problem, GeneticProgramming(;
            pop_size=12, generations=0, parallel=false);
            hall_of_fame_size=5)

        @test result.hall_of_fame !== nothing
        @test 1 <= length(result.hall_of_fame) <= 5
        @test result.hall_of_fame.fitnesses[1] ≈ result.best_fitness
    end

    @testset "checkpoint resume restores hall_of_fame archive" begin
        nguyen1 = Arborist.Benchmarks.nguyen(1)
        ops_spec = Arborist.Benchmarks.canonical_sr_operators(Float32)
        ops = OperatorEnum(binary_operators=ops_spec.binary,
                           unary_operators=ops_spec.unary)
        ev = TreeFitnessEvaluator(nguyen1.X, nguyen1.y, ops)
        alg = GeneticProgramming(; pop_size=16, generations=3, parallel=false)
        ckpt_path = tempname() * ".ckpt"

        problem1 = GPProblem(ev, TreeGenome{Float32}; seed=11)
        solve(problem1, alg;
              checkpoint_every=1,
              checkpoint_path=ckpt_path,
              hall_of_fame_size=6)

        ckpt = load_checkpoint(ckpt_path)
        @test ckpt.hall_of_fame isa HallOfFame
        @test 1 <= length(ckpt.hall_of_fame) <= 6

        problem2 = GPProblem(ev, TreeGenome{Float32}; seed=11)
        resumed = solve(problem2, alg;
                        resume_from=ckpt_path,
                        hall_of_fame_size=6)

        @test resumed.hall_of_fame !== nothing
        @test 1 <= length(resumed.hall_of_fame) <= 6
        @test resumed.hall_of_fame.fitnesses == ckpt.hall_of_fame.fitnesses

        rm(ckpt_path; force=true)
    end

    @testset "end-to-end: solve with ExprGenome populates hall_of_fame" begin
        fs = default_function_set()
        inputs  = Dict(:x => Float64)
        outputs = Dict(:out => Float64)
        ev = TableFitnessEvaluator(
            Dict{Symbol, DataType}(:x => Float64),
            Dict{Symbol, DataType}(:out => Float64),
            [Dict{Symbol, Any}(:x => 1.0), Dict{Symbol, Any}(:x => 2.0)],
            [Dict{Symbol, Any}(:out => 1.0), Dict{Symbol, Any}(:out => 2.0)])
        problem = GPProblem(ev, ExprGenome; function_set=fs, num_temps=2, seed=0)
        result = solve(problem, GeneticProgramming(;
            pop_size=20, generations=5, parallel=false);
            hall_of_fame_size=10)
        @test result.hall_of_fame !== nothing
        @test length(result.hall_of_fame) <= 10
    end

    @testset "hall_of_fame_size=0 returns nothing (default behavior)" begin
        nguyen1 = Arborist.Benchmarks.nguyen(1)
        ops_spec = Arborist.Benchmarks.canonical_sr_operators(Float32)
        ops = OperatorEnum(binary_operators=ops_spec.binary,
                           unary_operators=ops_spec.unary)
        ev = TreeFitnessEvaluator(nguyen1.X, nguyen1.y, ops)
        problem = GPProblem(ev, TreeGenome{Float32}; seed=1)
        result = solve(problem, GeneticProgramming(;
            pop_size=10, generations=2, parallel=false);
            hall_of_fame_size=0)
        @test result.hall_of_fame === nothing
    end

    @testset "validation" begin
        nguyen1 = Arborist.Benchmarks.nguyen(1)
        ops_spec = Arborist.Benchmarks.canonical_sr_operators(Float32)
        ops = OperatorEnum(binary_operators=ops_spec.binary,
                           unary_operators=ops_spec.unary)
        ev = TreeFitnessEvaluator(nguyen1.X, nguyen1.y, ops)
        problem = GPProblem(ev, TreeGenome{Float32}; seed=1)
        alg = GeneticProgramming(; pop_size=10, generations=1, parallel=false)
        @test_throws ArgumentError solve(problem, alg; hall_of_fame_size=-1)
    end
end
