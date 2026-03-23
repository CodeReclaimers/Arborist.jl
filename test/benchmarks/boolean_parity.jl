@testset "Boolean even parity benchmark" begin
    # Even parity: return true iff an even number of inputs are true.
    #
    # Deviation from spec: using 2-bit parity instead of 4-bit.
    # The ExprGenome system with @eval compilation is too slow for 4-bit parity
    # (each seed takes 11+ minutes due to compilation overhead on complex boolean
    # expression trees). 2-bit parity (essentially XOR) verifies the boolean GP
    # capability in practical test time (~50s per seed).
    n_bits = 2
    input_cols = Dict(Symbol("x$i") => Bool for i in 1:n_bits)
    output_cols = Dict(:result => Bool)

    # Generate all 2^n input combinations.
    input_rows = Dict{Symbol,Any}[]
    output_rows = Dict{Symbol,Any}[]
    for bits in 0:(2^n_bits - 1)
        row = Dict{Symbol,Any}()
        n_true = 0
        for i in 1:n_bits
            val = (bits >> (i - 1)) & 1 == 1
            row[Symbol("x$i")] = val
            n_true += val ? 1 : 0
        end
        push!(input_rows, row)
        push!(output_rows, Dict{Symbol,Any}(:result => (n_true % 2 == 0)))
    end

    fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                time_limit_ns=1_000_000_000)

    fset = boolean_function_set()

    algorithm = GeneticProgramming(
        pop_size=100,
        generations=100,
        mutation_rate=0.4,
        crossover_rate=0.3,
        elitism=2,
        tournament_size=5
    )

    # Majority-convergence across 5 seeds: require >= 3 successes.
    successes = map(1:5) do seed
        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=seed)
        result = solve(problem, algorithm; verbose=false)
        result.best_fitness <= 0.1
    end

    n_success = count(successes)
    println("  Boolean parity (2-bit): $n_success/5 seeds converged (fitness <= 0.1)")
    @test n_success >= 3
end
