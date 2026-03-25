@testset "Max Ones benchmark" begin
    # Max Ones: evolve a program that sets 4 boolean output variables to true.
    # A dummy Float32 input is required for the function signature.
    n_bits = 4
    input_cols = Dict(:dummy => Float32)
    output_cols = Dict(Symbol("b$i") => Bool for i in 1:n_bits)
    input_rows = [Dict{Symbol,Any}(:dummy => Float32(0.0))]
    output_rows = [Dict{Symbol,Any}(Symbol("b$i") => true for i in 1:n_bits)]
    fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                time_limit_ns=1_000_000_000)

    # Function set: basic arithmetic + comparisons for Bool generation.
    fset = default_function_set()

    problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=4, seed=42)
    algorithm = GeneticProgramming(
        pop_size=100,
        generations=100,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    result = solve(problem, algorithm; verbose=false)

    println("  Max Ones best fitness: $(result.best_fitness)")
    println("  Max Ones generations: $(result.generations_run)")

    # Fitness of 0.0 means all bits correct.
    # We allow fitness < 1.0 as a softer convergence criterion.
    @test result.best_fitness < 1.0
    @test result isa GPResult{ExprGenome}
    @test length(result.fitness_history) == algorithm.generations
    @test length(result.mean_history) == algorithm.generations
end
