@testset "GraphGenome" begin
    @testset "initialization" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g = initialize(GraphGenome, 2, 1, rng)
        @test g isa GraphGenome
        @test g.n_inputs == 2
        @test g.n_outputs == 1
        # Should have input nodes + bias + output
        @test length(g.nodes) == 4  # 2 inputs + 1 bias + 1 output
        # Fully connected: 3 inputs (2+bias) * 1 output = 3 connections
        @test length(g.connections) == 3
    end

    @testset "mutation" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g = initialize(GraphGenome, 2, 1, rng)
        for _ in 1:20
            g2 = mutate(g, rng)
            @test g2 isa GraphGenome
        end
    end

    @testset "crossover" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g1 = initialize(GraphGenome, 2, 1, rng)
        g2 = initialize(GraphGenome, 2, 1, rng)
        for _ in 1:10
            (c1, c2) = crossover(g1, g2, rng)
            @test c1 isa GraphGenome
            @test c2 isa GraphGenome
        end
    end

    @testset "distance" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g1 = initialize(GraphGenome, 2, 1, rng)
        g2 = initialize(GraphGenome, 2, 1, rng)
        d = distance(g1, g2)
        @test d isa Float64
        @test d >= 0.0
    end

    @testset "complexity and serialize" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g = initialize(GraphGenome, 2, 1, rng)
        @test complexity(g) isa Float64
        @test complexity(g) >= 0.0
        s = serialize(g)
        @test s isa String
        @test length(s) > 0
    end

    @testset "evaluation" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g = initialize(GraphGenome, 2, 1, rng)

        # XOR data
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        fitness = evaluate_genome(g, evaluator)
        @test fitness isa Float64
        @test fitness >= 0.0
    end

    @testset "solve (quick)" begin
        reset_innovation_counter!()
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        problem = GPProblem(evaluator, GraphGenome; seed=42)
        algorithm = GeneticProgramming(
            pop_size=30, generations=10,
            mutation_rate=0.5, crossover_rate=0.2,
            speciation=ThresholdSpeciation(threshold=3.0)
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{GraphGenome}
        @test result.best_fitness >= 0.0
        @test length(result.fitness_history) == 10
    end

    @testset "topological sort detects cycles" begin
        # Create a genome with a cycle
        nodes = Dict(
            1 => NodeGene(1, :input, :identity),
            2 => NodeGene(2, :hidden, :sigmoid),
            3 => NodeGene(3, :output, :sigmoid),
        )
        # Cycle: 2 -> 3 -> 2
        conns = Dict(
            1 => ConnectionGene(1, 2, 1.0, true, 1),
            2 => ConnectionGene(2, 3, 1.0, true, 2),
            3 => ConnectionGene(3, 2, 1.0, true, 3),  # cycle!
        )
        g = GraphGenome(nodes, conns, 1, 1, Inf)

        input_data = Float64[0.0 1.0; 0.0 0.0]  # 2 inputs x 2 samples
        output_data = reshape(Float64[0.0, 1.0], 1, :)  # 1 output x 2 samples
        evaluator = GraphEvaluator(input_data, output_data)

        # Should return Inf due to cycle
        @test evaluate_genome(g, evaluator) == Inf
    end
end
