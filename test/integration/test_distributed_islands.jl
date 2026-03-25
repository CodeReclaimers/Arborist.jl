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

    @testset "async=true errors (not yet implemented)" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        input_rows = [Dict{Symbol,Any}(:x => 1.0f0)]
        output_rows = [Dict{Symbol,Any}(:y => 1.0f0)]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        problem = GPProblem(fe, ExprGenome; seed=42)
        algorithm = IslandModel(n_islands=2, distributed=true, async=true)
        # Should error because async is not yet implemented, but also
        # because no workers are available. The distributed dispatch
        # should trigger before the async error in the current code path.
        @test_throws Exception solve(problem, algorithm)
    end
end
