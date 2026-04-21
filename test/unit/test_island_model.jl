@testset "IslandModel" begin
    @testset "IslandModel construction" begin
        im = IslandModel()
        @test im isa AbstractEvolutionaryAlgorithm
        @test im.n_islands == 4
        @test im.migration_interval == 10
        @test im.migration_size == 2
        @test im.island_algorithm isa GeneticProgramming

        im2 = IslandModel(n_islands=8, migration_interval=5, migration_size=3)
        @test im2.n_islands == 8
        @test im2.migration_interval == 5
        @test im2.migration_size == 3
    end

    @testset "IslandModel solve runs without error" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = IslandModel(
            n_islands=3,
            island_algorithm=GeneticProgramming(
                pop_size=20,
                generations=20,
                mutation_rate=0.4,
                crossover_rate=0.2
            ),
            migration_interval=5,
            migration_size=2
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test result.best_fitness >= 0.0
        @test result.best_fitness < Inf
        @test length(result.fitness_history) == 20
        @test length(result.mean_history) == 20
        # Population should be all islands merged.
        @test length(result.population) == 3 * 20
        @test result.wall_time > 0.0
    end

    @testset "IslandModel migration transfers individuals" begin
        # Run with very frequent migration to exercise the migration code.
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        add!(fset, :-, 2, Float32, Float32)

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=99)
        algorithm = IslandModel(
            n_islands=2,
            island_algorithm=GeneticProgramming(
                pop_size=10,
                generations=10,
                mutation_rate=0.3,
                crossover_rate=0.2
            ),
            migration_interval=2,  # migrate every 2 generations
            migration_size=1
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test length(result.population) == 20  # 2 islands × 10
    end

    @testset "_naturalize_migrant rebinds ExprGenome state" begin
        rng1 = Random.MersenneTwister(1)
        rng2 = Random.MersenneTwister(2)
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        add!(fset, :>, 2, Float32, Bool)
        inputs = Dict(:x => Float32)
        outputs = Dict(:y => Float32)
        state1 = GenState(rng1, fset, inputs, outputs, 2)
        state2 = GenState(rng2, fset, inputs, outputs, 2)

        body = [:(y = x + Float32(1.0))]
        genome = ExprGenome(body, state1)
        @test genome.state === state1
        @test genome.state.rng === rng1

        naturalized = Arborist._naturalize_migrant(genome, state2)
        @test naturalized.state === state2
        @test naturalized.state.rng === rng2
        @test naturalized.body == body
        println("  _naturalize_migrant: ExprGenome state rebound to destination")
        flush(stdout)
    end

    @testset "IslandModel solve with GraphGenome (sequential)" begin
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)
        problem = GPProblem(evaluator, GraphGenome; seed=42)

        ops = neat_defaults()
        algorithm = IslandModel(
            n_islands=3,
            island_algorithm=GeneticProgramming(
                pop_size=30, generations=20,
                mutation_rate=0.5, crossover_rate=0.3,
                mutation_ops=ops.mutation_ops,
                crossover_ops=ops.crossover_ops,
                speciation=ThresholdSpeciation(threshold=3.0),
            ),
            migration_interval=5,
            migration_size=2,
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{GraphGenome}
        @test result.best_fitness >= 0.0
        @test isfinite(result.best_fitness)
        @test length(result.fitness_history) == 20
        println("  IslandModel GraphGenome: best_fitness=$(round(result.best_fitness, digits=4)), " *
                "nodes=$(length(result.best_genome.nodes))")
        flush(stdout)
    end

    @testset "IslandModel rejects incompatible ops for GraphGenome" begin
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)
        problem = GPProblem(evaluator, GraphGenome; seed=1)

        # Default GP ops are ExprGenome-flavored — must be caught at solve entry.
        bad = IslandModel(n_islands=2,
                          island_algorithm=GeneticProgramming(pop_size=10, generations=2))
        @test_throws ArgumentError solve(problem, bad; verbose=false)
    end
end
