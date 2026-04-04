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

    @testset "mutation determinism (sorted Dict iteration)" begin
        # Verify that GraphGenome mutations produce identical results
        # when run with the same seed, regardless of Dict iteration order.
        # This tests the fix for non-deterministic Dict iteration in
        # _mutate_weights!, _mutate_weight_replace!, _mutate_add_connection!,
        # _mutate_add_node!, _mutate_toggle_connection!, and _neat_crossover.
        for trial in 1:5
            reset_innovation_counter!()
            rng1 = Random.MersenneTwister(trial)
            g1 = initialize(GraphGenome, 3, 2, rng1)
            # Run 50 mutations to exercise all mutation types
            for _ in 1:50
                g1 = mutate(g1, rng1)
            end
            weights1 = sort([c.weight for c in values(g1.connections)])

            reset_innovation_counter!()
            rng2 = Random.MersenneTwister(trial)
            g2 = initialize(GraphGenome, 3, 2, rng2)
            for _ in 1:50
                g2 = mutate(g2, rng2)
            end
            weights2 = sort([c.weight for c in values(g2.connections)])

            @test length(weights1) == length(weights2)
            @test weights1 ≈ weights2
            println("  GraphGenome determinism trial $trial: $(length(weights1)) connections, weights match")
            flush(stdout)
        end
    end

    @testset "crossover produces distinct children" begin
        # With swapped parent order for second child, the two children
        # should differ when parents have disjoint innovations.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(42)
        g1 = initialize(GraphGenome, 2, 1, rng)
        g2 = initialize(GraphGenome, 2, 1, rng)
        # Mutate to create structural differences (disjoint innovations)
        for _ in 1:15
            g1 = mutate(g1, rng)
        end
        for _ in 1:15
            g2 = mutate(g2, rng)
        end
        g1.fitness = 0.3; g2.fitness = 0.9
        c1, c2 = crossover(g1, g2, rng)
        # Children should have different connection sets because
        # c1 gets fitter parent's disjoint/excess, c2 gets other parent's
        inns1 = Set(keys(c1.connections))
        inns2 = Set(keys(c2.connections))
        @test inns1 != inns2
        println("  Crossover distinct children: $(length(inns1)) vs $(length(inns2)) innovations")
        flush(stdout)
    end

    @testset "crossover determinism (sorted Set iteration)" begin
        for trial in 1:5
            reset_innovation_counter!()
            rng1 = Random.MersenneTwister(100 + trial)
            g1 = initialize(GraphGenome, 2, 1, rng1)
            g2 = initialize(GraphGenome, 2, 1, rng1)
            # Mutate to create structural differences
            for _ in 1:10
                g1 = mutate(g1, rng1)
                g2 = mutate(g2, rng1)
            end
            g1.fitness = 0.5; g2.fitness = 1.0
            c1a, c2a = crossover(g1, g2, rng1)

            reset_innovation_counter!()
            rng2 = Random.MersenneTwister(100 + trial)
            g3 = initialize(GraphGenome, 2, 1, rng2)
            g4 = initialize(GraphGenome, 2, 1, rng2)
            for _ in 1:10
                g3 = mutate(g3, rng2)
                g4 = mutate(g4, rng2)
            end
            g3.fitness = 0.5; g4.fitness = 1.0
            c1b, c2b = crossover(g3, g4, rng2)

            w1a = sort([c.weight for c in values(c1a.connections)])
            w1b = sort([c.weight for c in values(c1b.connections)])
            @test w1a ≈ w1b
            println("  GraphGenome crossover determinism trial $trial: match")
            flush(stdout)
        end
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
