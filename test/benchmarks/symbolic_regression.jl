@testset "Symbolic regression x^2 benchmark" begin
    # Target: y = x^2 for Float32 inputs.
    input_cols = Dict(:x => Float32)
    output_cols = Dict(:y => Float32)
    xs = Float32[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
    output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]

    # Use a generous time limit (1 second) to avoid non-determinism from GC/JIT
    # timing variations. Loop safety is handled by add_loop_checks instead.
    fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                time_limit_ns=1_000_000_000)

    # Function set matching the existing test suite.
    fset = FunctionSet(Set{FunctionDetails}())
    for func in [:+, :-, :*, :/]
        add!(fset, func, 2, Float32, Float32)
    end
    for func in [:cos, :sin, :tanh, :sign]
        add!(fset, func, 1, Float32, Float32)
    end
    for op in [:>, :<, :(==), :!=, :>=, :<=]
        add!(fset, op, 2, Float32, Bool)
    end

    problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=4, seed=456)
    algorithm = GeneticProgramming(
        pop_size=100,
        generations=200,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    result = solve(problem, algorithm; verbose=false)

    println("  x^2 best fitness after $(result.generations_run) generations: $(result.best_fitness)")
    println("  x^2 converged: $(result.converged)")

    # The plan specifies fitness < 1.0 within 200 generations with pop_size=100.
    @test result.best_fitness < 1.0
    @test result isa GPResult{ExprGenome}
    @test length(result.fitness_history) == 200
    @test length(result.mean_history) == 200
    @test result.wall_time > 0.0
end
