@testset "AntGenome" begin
    @testset "initialization" begin
        rng = Random.MersenneTwister(42)
        primitives = [:gp_ant_move, :gp_ant_left, :gp_ant_right]
        conditions = [:gp_ant_food_ahead]
        g = initialize(AntGenome, primitives, conditions, 3, rng)
        @test g isa AntGenome
        @test g.program isa Expr
        @test complexity(g) > 0.0
    end

    @testset "mutation" begin
        rng = Random.MersenneTwister(42)
        primitives = [:gp_ant_move, :gp_ant_left, :gp_ant_right]
        conditions = [:gp_ant_food_ahead]
        g = initialize(AntGenome, primitives, conditions, 3, rng)
        for _ in 1:20
            g2 = mutate(g, rng)
            @test g2 isa AntGenome
        end
    end

    @testset "crossover" begin
        rng = Random.MersenneTwister(42)
        primitives = [:gp_ant_move, :gp_ant_left, :gp_ant_right]
        conditions = [:gp_ant_food_ahead]
        g1 = initialize(AntGenome, primitives, conditions, 3, rng)
        g2 = initialize(AntGenome, primitives, conditions, 3, rng)
        (c1, c2) = crossover(g1, g2, rng)
        @test c1 isa AntGenome
        @test c2 isa AntGenome
    end

    @testset "serialize/deserialize" begin
        rng = Random.MersenneTwister(42)
        primitives = [:gp_ant_move, :gp_ant_left, :gp_ant_right]
        conditions = [:gp_ant_food_ahead]
        g = initialize(AntGenome, primitives, conditions, 3, rng)
        s = serialize(g)
        @test s isa String
        @test length(s) > 0
    end

    @testset "evaluation" begin
        food = [(1,2), (1,3), (1,4)]
        evaluator = AntEvaluator(food, 20)
        rng = Random.MersenneTwister(42)

        primitives = [:gp_ant_move, :gp_ant_left, :gp_ant_right]
        conditions = [:gp_ant_food_ahead]
        g = initialize(AntGenome, primitives, conditions, 3, rng)

        fitness = evaluate_genome(g, evaluator)
        @test fitness isa Float64
        @test fitness >= 0.0
        @test fitness <= length(food)
    end

    @testset "solve (quick)" begin
        food = [(1,2), (1,3), (1,4)]
        evaluator = AntEvaluator(food, 50)

        problem = GPProblem(evaluator, AntGenome; seed=42)
        algorithm = GeneticProgramming(
            pop_size=10, generations=3,
            mutation_rate=0.4, crossover_rate=0.2,
            parallel=false
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{AntGenome}
        @test result.best_fitness >= 0.0
        @test length(result.fitness_history) == 3
    end
end
