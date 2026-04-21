using Distributed

@testset "Distributed Island Model" begin
    @testset "distributed=false backward compatibility" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*]
            add!(fset, func, 2, Float32, Float32)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = IslandModel(
            n_islands=2,
            island_algorithm=GeneticProgramming(
                pop_size=10, generations=10,
                mutation_rate=0.3, crossover_rate=0.2,
                parallel=false
            ),
            distributed=false
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test result.best_fitness < Inf
        @test length(result.fitness_history) == 10
        @test length(result.population) == 2 * 10
    end

    @testset "insufficient workers without auto_addprocs errors" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        input_rows = [Dict{Symbol,Any}(:x => 1.0f0)]
        output_rows = [Dict{Symbol,Any}(:y => 1.0f0)]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        problem = GPProblem(fe, ExprGenome; seed=42)
        algorithm = IslandModel(n_islands=4, distributed=true)
        @test_throws ErrorException solve(problem, algorithm)
    end

    @testset "Sync distributed solve (2 islands)" begin
        added = addprocs(2; exeflags="--project=$(Base.active_project())")
        @everywhere using Arborist

        try
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
            for op in [:>, :<, :(==), :>=, :<=]
                add!(fset, op, 2, Float32, Bool)
            end

            problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
            algorithm = IslandModel(
                n_islands=2,
                island_algorithm=GeneticProgramming(
                    pop_size=20, generations=15,
                    mutation_rate=0.4, crossover_rate=0.2,
                    parallel=false
                ),
                migration_interval=5, migration_size=2,
                distributed=true, async=false
            )

            result = solve(problem, algorithm; verbose=false)
            @test result isa GPResult{ExprGenome}
            @test result.best_fitness >= 0.0
            @test result.best_fitness < Inf
            @test length(result.fitness_history) == 15
            @test length(result.population) == 2 * 20
        finally
            rmprocs(added)
        end
    end

    @testset "Async distributed solve (2 islands)" begin
        added = addprocs(2; exeflags="--project=$(Base.active_project())")
        @everywhere using Arborist

        try
            input_cols = Dict(:x => Float32)
            output_cols = Dict(:y => Float32)
            xs = Float32[-1.0, 0.0, 1.0]
            input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
            output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
            fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                        time_limit_ns=1_000_000_000)

            fset = FunctionSet(Set{FunctionDetails}())
            for func in [:+, :-, :*]
                add!(fset, func, 2, Float32, Float32)
            end

            problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
            algorithm = IslandModel(
                n_islands=2,
                island_algorithm=GeneticProgramming(
                    pop_size=20, generations=15,
                    mutation_rate=0.4, crossover_rate=0.2,
                    parallel=false
                ),
                migration_interval=5, migration_size=2,
                distributed=true, async=true
            )

            result = solve(problem, algorithm; verbose=false)

            # Validity checks (async is non-deterministic)
            @test result isa GPResult{ExprGenome}
            @test result.best_fitness >= 0.0
            @test result.best_fitness < Inf
            @test result.generations_run == 15
            @test length(result.population) == 2 * 20
            @test !isempty(result.fitness_history)
            @test result.wall_time > 0.0
        finally
            rmprocs(added)
        end
    end

    @testset "Sync distributed GraphGenome (2 islands, disjoint innovations)" begin
        added = addprocs(2; exeflags="--project=$(Base.active_project())")
        @everywhere using Arborist

        try
            input_data = Float64[0 0 1 1; 0 1 0 1]
            output_data = Float64[0 1 1 0]
            evaluator = GraphEvaluator(input_data, output_data)
            problem = GPProblem(evaluator, GraphGenome; seed=42)

            ops = neat_defaults()
            algorithm = IslandModel(
                n_islands=2,
                island_algorithm=GeneticProgramming(
                    pop_size=20, generations=10,
                    mutation_rate=0.5, crossover_rate=0.2,
                    parallel=false,
                    mutation_ops=ops.mutation_ops,
                    crossover_ops=ops.crossover_ops,
                ),
                migration_interval=5, migration_size=2,
                distributed=true, async=false,
            )

            result = solve(problem, algorithm; verbose=false)
            @test result isa GPResult{GraphGenome}
            @test result.best_fitness >= 0.0
            @test result.best_fitness < Inf
            @test length(result.fitness_history) == 10
            @test length(result.population) == 2 * 20

            # Disjoint-range verification: every genome in the final population
            # should have all innovations within exactly one worker's range
            # ([0, STRIDE) or [STRIDE, 2*STRIDE)). If worker 1 assigned innovation
            # 3 and worker 2 also assigned innovation 3, migrants would carry
            # innovation 3 back to worker 1 where it would match an unrelated
            # connection — that's the bug this fix prevents. Post-migration,
            # genomes can carry innovations from either range, but never values
            # outside both ranges.
            stride = Arborist.INNOVATION_STRIDE
            for g in result.population
                for inn in keys(g.connections)
                    in_range_1 = 0 < inn <= stride
                    in_range_2 = stride < inn <= 2 * stride
                    @test in_range_1 || in_range_2
                end
            end
        finally
            rmprocs(added)
        end
    end
end
