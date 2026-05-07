using Test
using Arborist
using Random

@testset "Bloat penalty" begin
    @testset "bloat_penalty=0.0 preserves existing behavior" begin
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

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        alg = GeneticProgramming(pop_size=20, generations=5, bloat_penalty=0.0)

        result = solve(problem, alg; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test result.best_fitness >= 0.0
    end

    @testset "bloat_penalty > 0 increases fitness relative to raw" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)

        # With bloat penalty, fitness should be higher (worse) than raw evaluation.
        alg_bp = GeneticProgramming(pop_size=20, generations=5, bloat_penalty=0.01)
        result_bp = solve(problem, alg_bp; verbose=false)

        # The best fitness should include the penalty component.
        # Verify it's non-negative and the penalty is being applied by checking
        # that evaluate_genome gives a lower value than the stored fitness.
        g = result_bp.best_genome
        raw_fitness = Arborist.evaluate_genome(g, problem.evaluator)
        if isfinite(raw_fitness)
            expected = raw_fitness + 0.01 * complexity(g)
            @test result_bp.best_fitness ≈ expected atol=1e-10
        end
    end

    @testset "_evaluate_with_penalty helper" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        fe = TableFitnessEvaluator(input_cols, output_cols,
            [Dict{Symbol,Any}(:x => 1.0f0)],
            [Dict{Symbol,Any}(:y => 1.0f0)];
            time_limit_ns=1_000_000_000)

        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)
        g = ExprGenome([:(y = x)], s)

        raw = Arborist.evaluate_genome(g, fe)
        penalized = Arborist._evaluate_with_penalty(g, fe, 0.1)

        @test raw == 0.0  # y = x is perfect for y = x
        @test penalized ≈ 0.0 + 0.1 * complexity(g)
    end

    @testset "GraphGenome solve applies bloat_penalty" begin
        input_data = Float64[0 1 0 1; 0 0 1 1]
        output_data = reshape(Float64[0, 1, 1, 0], 1, :)
        evaluator = GraphEvaluator(input_data, output_data)
        problem = GPProblem(evaluator, GraphGenome; seed=7)
        ops = neat_defaults()
        alg = GeneticProgramming(;
            pop_size=6,
            generations=0,
            bloat_penalty=0.25,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
            parallel=false)

        result = solve(problem, alg)
        raw = evaluate_genome(result.best_genome, evaluator)

        @test isfinite(raw)
        @test result.best_fitness ≈ raw + 0.25 * complexity(result.best_genome)
        @test result.best_genome.fitness ≈ result.best_fitness
    end
end
