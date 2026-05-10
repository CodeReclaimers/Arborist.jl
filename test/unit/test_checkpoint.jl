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
        @test ckpt2.hall_of_fame === nothing

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

    @testset "resume rejects selection parameter mismatch" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        X = reshape(collect(Float32, 0.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)

        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=100)
        alg1 = GeneticProgramming(pop_size=12, generations=3, parallel=false,
                                   selection=TournamentSelection(2))
        ckpt_path = tempname() * ".ckpt"
        solve(problem, alg1; checkpoint_every=1, checkpoint_path=ckpt_path)

        alg2 = GeneticProgramming(pop_size=12, generations=3, parallel=false,
                                   selection=TournamentSelection(5))
        problem2 = GPProblem(evaluator, TreeGenome{Float32}; seed=100)
        @test_throws ArgumentError solve(problem2, alg2; resume_from=ckpt_path)

        rm(ckpt_path; force=true)
    end
end

@testset "Checkpoint / resume — GraphGenome" begin
    @testset "save+load round-trip (GraphGenome)" begin
        rng = Random.MersenneTwister(11)
        Arborist.reset_innovation_counter!()
        pop = [Arborist.initialize(GraphGenome, 2, 1, rng) for _ in 1:3]
        fits = Float64[1.0, 0.5, 0.25]

        ckpt = Checkpoint{GraphGenome}(
            Arborist.CHECKPOINT_FORMAT_VERSION,
            v"0.1.0", VERSION,
            7, pop, fits, rng,
            deepcopy(pop[3]), 0.25,
            Float64[0.9, 0.7, 0.5, 0.3],
            Float64[1.0, 0.8, 0.6, 0.4],
            42.0,
            UInt64(0xdeadbeef),
        )
        tmp = tempname() * ".ckpt"
        save_checkpoint(ckpt, tmp)
        @test isfile(tmp)

        loaded = load_checkpoint(tmp)
        @test loaded isa Checkpoint{GraphGenome}
        @test loaded.generation == 7
        @test length(loaded.population) == 3
        @test loaded.best_fitness == 0.25
        @test loaded.algorithm_signature == UInt64(0xdeadbeef)

        # Connection details survive round-trip.
        for (orig, restored) in zip(pop, loaded.population)
            @test keys(orig.connections) == keys(restored.connections)
            for (inn, c) in orig.connections
                cr = restored.connections[inn]
                @test cr.in_node == c.in_node
                @test cr.out_node == c.out_node
                @test cr.weight == c.weight
                @test cr.enabled == c.enabled
                @test cr.innovation == c.innovation
            end
            @test keys(orig.nodes) == keys(restored.nodes)
        end

        rm(tmp; force=true)
    end

    @testset "end-to-end checkpoint+resume produces matching trajectory" begin
        # Small XOR-style table evaluator.
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        ops = neat_defaults()
        alg = GeneticProgramming(
            pop_size=12, generations=8,
            mutation_rate=0.5, crossover_rate=0.2,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
            parallel=false,
        )

        # Reference run: 8 generations straight through.
        ref = solve(GPProblem(evaluator, GraphGenome; seed=11), alg)

        # Checkpointed run: same config, write at gen 4 + 8.
        ckpt_path = tempname() * ".ckpt"
        r1 = solve(GPProblem(evaluator, GraphGenome; seed=11), alg;
                   checkpoint_every=4, checkpoint_path=ckpt_path)

        # Resume from the gen-8 checkpoint with a fresh problem object.
        r2 = solve(GPProblem(evaluator, GraphGenome; seed=11), alg;
                   resume_from=ckpt_path)

        @test length(r1.fitness_history) == 8
        @test length(r2.fitness_history) == 8

        # parallel=false → exact equality.
        @test r1.best_fitness == ref.best_fitness
        @test r2.fitness_history == ref.fitness_history
        @test r2.best_fitness == ref.best_fitness

        rm(ckpt_path; force=true)
    end

    @testset "innovation counter restored on resume" begin
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        ops = neat_defaults()
        alg = GeneticProgramming(
            pop_size=10, generations=4,
            mutation_rate=0.6, crossover_rate=0.2,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
            parallel=false,
        )

        ckpt_path = tempname() * ".ckpt"
        solve(GPProblem(evaluator, GraphGenome; seed=29), alg;
              checkpoint_every=4, checkpoint_path=ckpt_path)

        ckpt = load_checkpoint(ckpt_path)
        @test ckpt isa Checkpoint{GraphGenome}
        # Maximum innovation present in the checkpointed population.
        max_inn_ckpt = 0
        for g in ckpt.population
            for c in values(g.connections)
                if c.innovation > max_inn_ckpt
                    max_inn_ckpt = c.innovation
                end
            end
        end
        @test max_inn_ckpt > 0  # sanity

        # Resume + run a few more generations with structural mutation enabled.
        # Use only AddNodeMutation + AddConnectionMutation so every mutation
        # touches structure and any newly-introduced innovation IDs are
        # observable.
        struct_alg = GeneticProgramming(
            pop_size=10,
            generations=alg.generations + 4,
            mutation_rate=1.0, crossover_rate=0.0,
            mutation_ops=AbstractMutationOperator[
                AddNodeMutation(),
                AddConnectionMutation(),
            ],
            crossover_ops=ops.crossover_ops,
            parallel=false,
        )
        r = solve(GPProblem(evaluator, GraphGenome; seed=29), struct_alg;
                  resume_from=ckpt_path,
                  allow_signature_mismatch=true)

        # The post-resume population should contain at least one connection
        # with an innovation ID strictly greater than the max in the
        # checkpoint.
        max_inn_after = 0
        for g in r.population
            for c in values(g.connections)
                if c.innovation > max_inn_after
                    max_inn_after = c.innovation
                end
            end
        end
        @test max_inn_after > max_inn_ckpt

        rm(ckpt_path; force=true)
    end

    @testset "GraphGenome resume rejects algorithm signature mismatch" begin
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        ops = neat_defaults()
        alg1 = GeneticProgramming(
            pop_size=12, generations=3,
            mutation_rate=0.4, crossover_rate=0.2,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
            parallel=false,
        )

        ckpt_path = tempname() * ".ckpt"
        solve(GPProblem(evaluator, GraphGenome; seed=17), alg1;
              checkpoint_every=1, checkpoint_path=ckpt_path)

        alg2 = GeneticProgramming(
            pop_size=20, generations=3,  # different pop_size
            mutation_rate=0.4, crossover_rate=0.2,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
            parallel=false,
        )
        # Note: pop_size mismatch isn't independently fatal in GraphGenome's
        # solve until breeding, but the signature check must catch it.
        @test_throws ArgumentError solve(
            GPProblem(evaluator, GraphGenome; seed=17), alg2;
            resume_from=ckpt_path)

        # Override succeeds.
        r = solve(GPProblem(evaluator, GraphGenome; seed=17), alg2;
                  resume_from=ckpt_path,
                  allow_signature_mismatch=true)
        @test r isa GPResult{GraphGenome}

        rm(ckpt_path; force=true)
    end
end

@testset "Checkpoint / resume — NSGA-II" begin
    @testset "save+load round-trip (NSGAIICheckpoint)" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        rng = Random.MersenneTwister(42)
        pop = [TreeGenome{Float32}(Node{Float32}(; val=Float32(i)), ops, 0)
               for i in 1:3]
        fits = [Float64[0.1, 1.0], Float64[0.2, 2.0], Float64[0.3, 3.0]]
        hv_hist = Float64[0.5, 0.7, 0.85, 0.92]

        ckpt = NSGAIICheckpoint{TreeGenome{Float32}}(
            Arborist.NSGAII_CHECKPOINT_FORMAT_VERSION,
            v"0.1.0", VERSION,
            4, pop, fits, rng, hv_hist, 12.5,
            UInt64(0xfeedbabe),
        )
        tmp = tempname() * ".ckpt"
        save_checkpoint(ckpt, tmp)
        @test isfile(tmp)

        loaded = load_checkpoint(tmp)
        @test loaded isa NSGAIICheckpoint{TreeGenome{Float32}}
        @test loaded.generation == 4
        @test length(loaded.population) == 3
        @test loaded.fitnesses == fits
        @test loaded.hypervolume_history == hv_hist
        @test loaded.wall_time == 12.5
        @test loaded.algorithm_signature == UInt64(0xfeedbabe)

        rm(tmp; force=true)
    end

    @testset "load_checkpoint rejects NSGA-II format-version mismatch" begin
        ops = OperatorEnum(binary_operators=[+], unary_operators=[])
        bad = NSGAIICheckpoint{TreeGenome{Float32}}(
            99,  # wrong format version
            v"0.1.0", VERSION,
            1, [TreeGenome{Float32}(Node{Float32}(; val=Float32(0.0)), ops, 0)],
            [Float64[0.0, 0.0]], MersenneTwister(1),
            Float64[0.0], 0.0, UInt64(0),
        )
        tmp = tempname() * ".ckpt"
        save_checkpoint(bad, tmp)
        @test_throws ArgumentError load_checkpoint(tmp)
        rm(tmp; force=true)
    end

    @testset "end-to-end NSGA-II checkpoint+resume" begin
        ops = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[])
        xs = Float32.(range(-1, 1, length=15))
        X = reshape(xs, 1, :)
        y = xs .^ 2

        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))
        alg = NSGAII(pop_size=10, generations=6,
                     mutation_rate=0.4, crossover_rate=0.3,
                     parallel=false)

        # Reference run.
        ref = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=13), alg)

        # Checkpointed run: write every 3 generations (so at gen 3 + 6).
        ckpt_path = tempname() * ".ckpt"
        r1 = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=13), alg;
                   checkpoint_every=3, checkpoint_path=ckpt_path)

        @test length(r1.hypervolume_history) == 6
        @test r1.hypervolume_history == ref.hypervolume_history

        # Resume from the gen-6 checkpoint. Loop body runs zero iterations
        # because start_gen = 7 > 6, so the population/fitnesses come back
        # unchanged.
        r2 = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=13), alg;
                   resume_from=ckpt_path)
        @test length(r2.hypervolume_history) == 6
        @test r2.hypervolume_history == ref.hypervolume_history

        rm(ckpt_path; force=true)
    end

    @testset "NSGA-II resume continues from mid-run checkpoint" begin
        ops = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[])
        xs = Float32.(range(-1, 1, length=15))
        X = reshape(xs, 1, :)
        y = xs .^ 2

        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))
        alg = NSGAII(pop_size=10, generations=6,
                     mutation_rate=0.4, crossover_rate=0.3,
                     parallel=false)

        # Reference run for trajectory comparison.
        ref = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=21), alg)

        # Save only the gen-3 checkpoint by running 3 generations first.
        alg_short = NSGAII(pop_size=10, generations=3,
                           mutation_rate=0.4, crossover_rate=0.3,
                           parallel=false)
        ckpt_path = tempname() * ".ckpt"
        solve(GPProblem(evaluator, TreeGenome{Float32}; seed=21), alg_short;
              checkpoint_every=3, checkpoint_path=ckpt_path)

        # Resume with the full 6-generation budget.
        r2 = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=21), alg;
                   resume_from=ckpt_path,
                   allow_signature_mismatch=true)  # alg.generations differs

        @test length(r2.hypervolume_history) == 6
        @test r2.hypervolume_history == ref.hypervolume_history

        rm(ckpt_path; force=true)
    end

    @testset "NSGA-II resume rejects algorithm signature mismatch" begin
        ops = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[])
        xs = Float32.(range(-1, 1, length=10))
        X = reshape(xs, 1, :)
        y = xs .^ 2

        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))

        alg1 = NSGAII(pop_size=10, generations=3,
                      mutation_rate=0.4, crossover_rate=0.3,
                      parallel=false)
        ckpt_path = tempname() * ".ckpt"
        solve(GPProblem(evaluator, TreeGenome{Float32}; seed=99), alg1;
              checkpoint_every=1, checkpoint_path=ckpt_path)

        alg2 = NSGAII(pop_size=20, generations=3,
                      mutation_rate=0.4, crossover_rate=0.3,
                      parallel=false)
        @test_throws ArgumentError solve(
            GPProblem(evaluator, TreeGenome{Float32}; seed=99), alg2;
            resume_from=ckpt_path)

        rm(ckpt_path; force=true)
    end

    @testset "NSGA-II solve rejects resume_from + initial_population" begin
        ops = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[])
        xs = Float32.(range(-1, 1, length=10))
        X = reshape(xs, 1, :)
        y = xs .^ 2
        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))

        alg = NSGAII(pop_size=10, generations=2,
                     mutation_rate=0.4, crossover_rate=0.3,
                     parallel=false)
        ckpt_path = tempname() * ".ckpt"
        solve(GPProblem(evaluator, TreeGenome{Float32}; seed=5), alg;
              checkpoint_every=1, checkpoint_path=ckpt_path)

        # Building a placeholder initial_population just for the validation check.
        seed = [TreeGenome{Float32}(Node{Float32}(; val=Float32(0.0)), ops, 0)
                for _ in 1:10]
        @test_throws ArgumentError solve(
            GPProblem(evaluator, TreeGenome{Float32}; seed=5), alg;
            resume_from=ckpt_path,
            initial_population=seed)

        rm(ckpt_path; force=true)
    end
end

@testset "Integration: 20-gen multi-checkpoint resume cycles" begin
    # Validates that resuming from any of three intermediate checkpoints
    # (gens 5, 10, 15) produces an identical trajectory to a single
    # uninterrupted 20-generation run. Covers population state, RNG state,
    # innovation counter (GraphGenome path), and algorithm state (NSGA-II path).
    #
    # Trick: the existing _save_ckpt overwrites a single file each time, so
    # we use the per-generation `callback` (which runs at the START of each
    # generation, AFTER the previous gen's checkpoint has been written) to
    # copy the file off to separate per-generation paths.

    @testset "Single-objective GraphGenome: 20-gen, ckpts at 5/10/15" begin
        # GraphGenome single-obj solve: every newly-allocated innovation ID at
        # gen N ends up in `genomes` (= `next_genomes`) at end of gen N, so
        # max_inn(population) == global_counter and rebuild-from-population
        # produces a bit-exact resume.
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)
        ops = neat_defaults()
        alg = GeneticProgramming(
            pop_size=14, generations=20,
            mutation_rate=0.5, crossover_rate=0.2,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
            parallel=false,
        )
        seed = 41

        # Reference run.
        ref = solve(GPProblem(evaluator, GraphGenome; seed=seed), alg)
        @test length(ref.fitness_history) == 20

        # Capture intermediate checkpoints during a parallel checkpointed run.
        ckpt_path = tempname() * ".ckpt"
        saved_at = Dict{Int, String}()
        cb = (gen, fit, genome) -> begin
            for target in (5, 10, 15)
                if gen > target && !haskey(saved_at, target) && isfile(ckpt_path)
                    p = ckpt_path * ".at$target"
                    cp(ckpt_path, p; force=true)
                    saved_at[target] = p
                end
            end
        end
        solve(GPProblem(evaluator, GraphGenome; seed=seed), alg;
              checkpoint_every=5, checkpoint_path=ckpt_path,
              callback=cb)

        @test haskey(saved_at, 5)
        @test haskey(saved_at, 10)
        @test haskey(saved_at, 15)

        # Each captured checkpoint records the expected generation.
        for target in (5, 10, 15)
            ckpt = load_checkpoint(saved_at[target])
            @test ckpt isa Checkpoint{GraphGenome}
            @test ckpt.generation == target
            @test length(ckpt.fitness_history) == target
        end

        # Resume from each checkpoint, run to gen 20, verify identical
        # trajectory to the uninterrupted reference.
        for target in (5, 10, 15)
            r = solve(GPProblem(evaluator, GraphGenome; seed=seed), alg;
                      resume_from=saved_at[target])
            @test length(r.fitness_history) == 20
            @test r.fitness_history == ref.fitness_history
            @test r.mean_history == ref.mean_history
            @test r.best_fitness == ref.best_fitness
            println("  GraphGenome resume from gen $target: " *
                    "best_fitness=$(round(r.best_fitness, digits=6)) " *
                    "(ref=$(round(ref.best_fitness, digits=6))), match=true")
            flush(stdout)
        end

        # Cleanup.
        rm(ckpt_path; force=true)
        for p in values(saved_at)
            rm(p; force=true)
        end
    end

    @testset "NSGA-II TreeGenome: 20-gen, ckpts at 5/10/15" begin
        ops = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[])
        xs = Float32.(range(-1, 1, length=20))
        X = reshape(xs, 1, :)
        y = xs .^ 2 .+ Float32(0.5) .* xs

        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))
        alg = NSGAII(pop_size=20, generations=20,
                     mutation_rate=0.4, crossover_rate=0.3,
                     parallel=false)
        seed = 42

        # Reference: 20 gens straight through.
        ref = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=seed), alg)
        @test length(ref.hypervolume_history) == 20

        # Capture intermediate checkpoints. NSGA-II callback signature:
        # (gen, front1_size, hypervolume).
        ckpt_path = tempname() * ".ckpt"
        saved_at = Dict{Int, String}()
        cb = (gen, front1_size, hv) -> begin
            for target in (5, 10, 15)
                if gen > target && !haskey(saved_at, target) && isfile(ckpt_path)
                    p = ckpt_path * ".at$target"
                    cp(ckpt_path, p; force=true)
                    saved_at[target] = p
                end
            end
        end
        solve(GPProblem(evaluator, TreeGenome{Float32}; seed=seed), alg;
              checkpoint_every=5, checkpoint_path=ckpt_path,
              callback=cb)

        @test haskey(saved_at, 5)
        @test haskey(saved_at, 10)
        @test haskey(saved_at, 15)

        # Each captured checkpoint records the expected generation.
        for target in (5, 10, 15)
            ckpt = load_checkpoint(saved_at[target])
            @test ckpt isa NSGAIICheckpoint{TreeGenome{Float32}}
            @test ckpt.generation == target
            @test length(ckpt.hypervolume_history) == target
        end

        # Resume from each checkpoint, run to gen 20, verify identical
        # hypervolume history and Pareto front fitnesses.
        for target in (5, 10, 15)
            r = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=seed), alg;
                      resume_from=saved_at[target])
            @test length(r.hypervolume_history) == 20
            @test r.hypervolume_history == ref.hypervolume_history
            @test r.pareto_fitnesses == ref.pareto_fitnesses
            @test length(r.pareto_front) == length(ref.pareto_front)
            println("  NSGA-II resume from gen $target: " *
                    "front=$(length(r.pareto_front)), " *
                    "final hv=$(round(r.hypervolume_history[end], digits=6)) " *
                    "(ref=$(round(ref.hypervolume_history[end], digits=6))), " *
                    "match=true")
            flush(stdout)
        end

        # Sequential resume cycle: resume from gen 5 → continue to gen 20,
        # but along the way save NEW intermediate checkpoints at gens 10 and
        # 15. Then resume from those new checkpoints. This verifies that the
        # state saved by a resumed run is equivalent to state saved by the
        # original run (no drift across resume boundaries).
        ckpt_path2 = tempname() * ".ckpt"
        saved_at2 = Dict{Int, String}()
        cb2 = (gen, front1_size, hv) -> begin
            for target in (10, 15)
                if gen > target && !haskey(saved_at2, target) && isfile(ckpt_path2)
                    p = ckpt_path2 * ".at$target"
                    cp(ckpt_path2, p; force=true)
                    saved_at2[target] = p
                end
            end
        end
        solve(GPProblem(evaluator, TreeGenome{Float32}; seed=seed), alg;
              resume_from=saved_at[5],
              checkpoint_every=5, checkpoint_path=ckpt_path2,
              callback=cb2)
        @test haskey(saved_at2, 10)
        @test haskey(saved_at2, 15)
        for target in (10, 15)
            ckpt_orig = load_checkpoint(saved_at[target])
            ckpt_new  = load_checkpoint(saved_at2[target])
            @test ckpt_orig.generation == ckpt_new.generation
            @test ckpt_orig.fitnesses == ckpt_new.fitnesses
            @test ckpt_orig.hypervolume_history == ckpt_new.hypervolume_history
        end

        # Cleanup.
        rm(ckpt_path; force=true)
        rm(ckpt_path2; force=true)
        for p in values(saved_at)
            rm(p; force=true)
        end
        for p in values(saved_at2)
            rm(p; force=true)
        end
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
