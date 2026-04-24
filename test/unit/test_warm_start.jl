using Test
using Arborist
using Random
using DynamicExpressions

@testset "Warm-start (initial_population)" begin
    @testset "TreeGenome: carry-over fitness floor" begin
        # Build a linear dataset y = 2x + 1. Evolve once to convergence,
        # harvest the population, and warm-start a second run. The second
        # run's best fitness must be <= the seed's best because the first
        # generation's evaluation already sees the converged genomes.
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        xs = collect(Float32, -1.0f0:0.1f0:1.0f0)
        X  = reshape(xs, 1, :)
        y  = Float32.(2.0f0 .* xs .+ 1.0f0)
        eval_ = TreeFitnessEvaluator(X, y, ops)

        problem = GPProblem(eval_, TreeGenome{Float32}; seed=7)
        alg1 = GeneticProgramming(;
            pop_size=30, generations=30,
            mutation_rate=0.3, crossover_rate=0.3,
            elitism=2, parallel=false,
        )
        r1 = solve(problem, alg1)
        seed_pop = r1.population
        @test length(seed_pop) == 30

        problem2 = GPProblem(eval_, TreeGenome{Float32}; seed=99)
        alg2 = GeneticProgramming(;
            pop_size=30, generations=1,
            mutation_rate=0.0, crossover_rate=0.0,
            elitism=30,
            parallel=false,
        )
        r2 = solve(problem2, alg2; initial_population=seed_pop)
        @test r2.best_fitness <= r1.best_fitness + 1e-6
    end

    @testset "TreeGenome: validation errors" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        xs = collect(Float32, 0.0f0:0.1f0:1.0f0)
        X  = reshape(xs, 1, :)
        y  = xs
        eval_ = TreeFitnessEvaluator(X, y, ops)

        problem = GPProblem(eval_, TreeGenome{Float32}; seed=1)
        alg = GeneticProgramming(;
            pop_size=10, generations=1, parallel=false,
        )

        # Build one valid genome to use in a shortened list.
        tmp = solve(problem, GeneticProgramming(; pop_size=5, generations=1,
                                                 mutation_rate=0.0, crossover_rate=0.0,
                                                 elitism=5, parallel=false))
        g0 = tmp.population[1]

        # Length mismatch.
        short_pop = [deepcopy(g0) for _ in 1:5]
        @test_throws ArgumentError solve(problem, alg; initial_population=short_pop)

        long_pop = [deepcopy(g0) for _ in 1:20]
        @test_throws ArgumentError solve(problem, alg; initial_population=long_pop)

        # Mutual exclusion with resume_from.
        ckpt_path = joinpath(mktempdir(), "ckpt.jls")
        # Create a real checkpoint so load_checkpoint doesn't fail before the
        # mutual-exclusion check.
        right_pop = [deepcopy(g0) for _ in 1:10]
        @test_throws ArgumentError solve(problem, alg;
            initial_population=right_pop, resume_from=ckpt_path)
    end

    @testset "ExprGenome: generic solve accepts initial_population" begin
        # Exercise the generic solve path (ExprGenome). We don't care about
        # fitness shape, only that the path validates and runs.
        fs = default_function_set()
        inputs  = Dict(:x => Float64)
        outputs = Dict(:out => Float64)
        state = GenState(Random.MersenneTwister(0), fs, inputs, outputs, 2)

        seed_genomes = [ExprGenome([create_random_assignment(state) for _ in 1:3], state)
                        for _ in 1:8]

        ev = TableFitnessEvaluator(
            Dict{Symbol, DataType}(:x => Float64),
            Dict{Symbol, DataType}(:out => Float64),
            [Dict{Symbol, Any}(:x => 1.0), Dict{Symbol, Any}(:x => 2.0)],
            [Dict{Symbol, Any}(:out => 1.0), Dict{Symbol, Any}(:out => 2.0)])
        problem = GPProblem(ev, ExprGenome; function_set=fs, num_temps=2, seed=0)
        alg = GeneticProgramming(;
            pop_size=8, generations=1,
            mutation_rate=0.0, crossover_rate=0.0, elitism=8,
            parallel=false,
        )
        result = solve(problem, alg; initial_population=seed_genomes)
        @test length(result.population) == 8
    end

    @testset "NSGA-II: initial_population threaded" begin
        # Use TreeGenome + ParsimonyEvaluator (single-objective fitness lifted
        # to (fitness, complexity)). Warm-start must produce front-size > 0.
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        xs = collect(Float32, -1.0f0:0.2f0:1.0f0)
        X  = reshape(xs, 1, :)
        y  = Float32.(xs .+ 0.5f0)
        base_eval = TreeFitnessEvaluator(X, y, ops)
        mo_eval = ParsimonyEvaluator(base_eval)

        problem = GPProblem(mo_eval, TreeGenome{Float32}; seed=7)
        # Build a seed population via a single-objective solve.
        seed_prob = GPProblem(base_eval, TreeGenome{Float32}; seed=7)
        seed_alg = GeneticProgramming(; pop_size=20, generations=3,
                                        mutation_rate=0.3, crossover_rate=0.3,
                                        elitism=2, parallel=false)
        seed_result = solve(seed_prob, seed_alg)
        seed_pop = seed_result.population

        nsga = NSGAII(; pop_size=20, generations=1,
                        mutation_rate=0.0, crossover_rate=0.0, parallel=false)
        result = solve(problem, nsga; initial_population=seed_pop)
        @test !isempty(result.pareto_front)
        @test length(result.population) == 20
    end
end
