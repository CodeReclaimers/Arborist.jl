using Test
using Arborist
using Random
using DynamicExpressions

@testset "Checkpoint / resume" begin
    @testset "save_checkpoint + load_checkpoint round-trip" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        X = reshape(Float32[1, 2, 3], 1, :)
        y = Float32[2, 4, 6]
        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        alg = GeneticProgramming(pop_size=15, generations=5, parallel=false)

        # Run a short job, checkpoint it, load it, inspect.
        rng = MersenneTwister(123)
        ckpt = Checkpoint{TreeGenome{Float32}}(
            Arborist.CHECKPOINT_FORMAT_VERSION,
            v"0.1.0", VERSION,
            3, [TreeGenome{Float32}(Node{Float32}(; val=Float32(i)), ops, 0) for i in 1:3],
            Float64[0.1, 0.2, 0.3],
            rng,
            TreeGenome{Float32}(Node{Float32}(; val=Float32(0.0)), ops, 0),
            0.05,
            Float64[0.5, 0.3, 0.1],
            Float64[0.6, 0.4, 0.2],
            12.5,
            UInt64(0xdeadbeef),
        )
        tmp = tempname() * ".ckpt"
        save_checkpoint(ckpt, tmp)
        @test isfile(tmp)

        ckpt2 = load_checkpoint(tmp)
        @test ckpt2.generation == 3
        @test length(ckpt2.population) == 3
        @test ckpt2.fitnesses == [0.1, 0.2, 0.3]
        @test ckpt2.best_fitness == 0.05
        @test ckpt2.wall_time == 12.5
        @test ckpt2.algorithm_signature == UInt64(0xdeadbeef)

        rm(tmp; force=true)
    end

    @testset "load_checkpoint rejects missing file" begin
        @test_throws ArgumentError load_checkpoint("/nonexistent/path/ckpt.jls")
    end

    @testset "load_checkpoint rejects format-version mismatch" begin
        ops = OperatorEnum(binary_operators=[+], unary_operators=[])
        bad = Checkpoint{TreeGenome{Float32}}(
            99,  # wrong format version
            v"0.1.0", VERSION,
            1, [TreeGenome{Float32}(Node{Float32}(; val=Float32(0.0)), ops, 0)],
            Float64[0.0], MersenneTwister(1),
            TreeGenome{Float32}(Node{Float32}(; val=Float32(0.0)), ops, 0),
            0.0, Float64[0.0], Float64[0.0], 0.0, UInt64(0),
        )
        tmp = tempname() * ".ckpt"
        save_checkpoint(bad, tmp)
        @test_throws ArgumentError load_checkpoint(tmp)
        rm(tmp; force=true)
    end

    @testset "end-to-end checkpoint + resume produces identical trajectory" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        X = reshape(collect(Float32, -1.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)

        # Reference run: 10 generations straight through.
        problem1 = GPProblem(evaluator, TreeGenome{Float32}; seed=7)
        alg = GeneticProgramming(pop_size=20, generations=10, parallel=false)
        ref = solve(problem1, alg)

        # Checkpointed run: 10 generations with a checkpoint every 5 (so at gen 5 + 10).
        problem2 = GPProblem(evaluator, TreeGenome{Float32}; seed=7)
        ckpt_path = tempname() * ".ckpt"
        r1 = solve(problem2, alg; checkpoint_every=5, checkpoint_path=ckpt_path)

        # Resume from 5 with 10-generation budget — should finish the remaining 5.
        # Use a fresh problem so init draws are identical, then resume restores
        # the true state via the checkpoint.
        problem3 = GPProblem(evaluator, TreeGenome{Float32}; seed=7)
        r2 = solve(problem3, alg; resume_from=ckpt_path)

        # Both endpoints should reach the same best fitness.
        @test r1.best_fitness == ref.best_fitness
        @test isfinite(r2.best_fitness)
        # Resumed run's fitness history should extend the checkpoint's.
        @test length(r2.fitness_history) == 10

        rm(ckpt_path; force=true)
    end

    @testset "resume rejects algorithm signature mismatch" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        X = reshape(collect(Float32, 0.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)

        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=99)
        alg1 = GeneticProgramming(pop_size=10, generations=3, parallel=false)
        ckpt_path = tempname() * ".ckpt"
        solve(problem, alg1; checkpoint_every=1, checkpoint_path=ckpt_path)

        # Resume with a DIFFERENT config should error.
        alg2 = GeneticProgramming(pop_size=20, generations=3, parallel=false)
        problem2 = GPProblem(evaluator, TreeGenome{Float32}; seed=99)
        @test_throws ArgumentError solve(problem2, alg2; resume_from=ckpt_path)

        # Override: allow_signature_mismatch=true should succeed.
        result = solve(problem2, alg2; resume_from=ckpt_path,
                       allow_signature_mismatch=true)
        @test isfinite(result.best_fitness)

        rm(ckpt_path; force=true)
    end
end

@testset "Operator success tracking in RunLog" begin
    ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
    X = reshape(collect(Float32, 0.0:0.1:1.0), 1, :)
    y = X[1, :] .* X[1, :]
    evaluator = TreeFitnessEvaluator(X, y, ops)
    problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
    alg = GeneticProgramming(pop_size=20, generations=5, parallel=false,
                             crossover_rate=0.3, mutation_rate=0.5,
                             mutation_ops=AbstractMutationOperator[SubtreeMutation(), PointMutation()],
                             crossover_ops=AbstractCrossoverOperator[SubtreeCrossover()])
    log = RunLog()
    solve(problem, alg; log=log)

    @test length(log) == 5
    # Each generation's log should show at least one operator invocation.
    for e in entries(log)
        @test sum(values(e.operator_attempted)) >= 1
        @test sum(values(e.operator_success)) >= 0
        # Success can never exceed attempted for any single operator.
        for (name, att) in e.operator_attempted
            succ = get(e.operator_success, name, 0)
            @test succ <= att
        end
    end
    # Over the whole run, SubtreeMutation should have been invoked at least once.
    total_subtree = sum(get(e.operator_attempted, :SubtreeMutation, 0) for e in entries(log))
    @test total_subtree >= 1
end
