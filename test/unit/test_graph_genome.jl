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
        mut = NEATDefaultMutation()
        for _ in 1:20
            g2 = mutate(mut, g, rng)
            @test g2 isa GraphGenome
        end
    end

    @testset "crossover" begin
        rng = Random.MersenneTwister(42)
        reset_innovation_counter!()
        g1 = initialize(GraphGenome, 2, 1, rng)
        g2 = initialize(GraphGenome, 2, 1, rng)
        xo = NEATCrossover()
        for _ in 1:10
            (c1, c2) = crossover(xo, g1, g2, rng)
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

    @testset "NEAT distance separates disjoint and excess" begin
        # Genome A: innovations 1,2,3,5 (max=5)
        # Genome B: innovations 1,2,4,6,7 (max=7)
        # Matching: {1,2}
        # Disjoint (within range of both): 3 (in A, <=7), 4 (in B, <=5), 5 (in A, <=7) = 3
        # Excess (beyond other's max): 6 (in B, >5), 7 (in B, >5) = 2
        nodes_a = Dict(1 => NodeGene(1, :input, :identity),
                       2 => NodeGene(2, :output, :sigmoid))
        conns_a = Dict(
            1 => ConnectionGene(1, 2, 0.5, true, 1),
            2 => ConnectionGene(1, 2, 0.5, true, 2),
            3 => ConnectionGene(1, 2, 0.5, true, 3),
            5 => ConnectionGene(1, 2, 0.5, true, 5),
        )
        ga = GraphGenome(nodes_a, conns_a, 1, 1, Inf)

        nodes_b = Dict(1 => NodeGene(1, :input, :identity),
                       2 => NodeGene(2, :output, :sigmoid))
        conns_b = Dict(
            1 => ConnectionGene(1, 2, 0.5, true, 1),
            2 => ConnectionGene(1, 2, 0.5, true, 2),
            4 => ConnectionGene(1, 2, 0.5, true, 4),
            6 => ConnectionGene(1, 2, 0.5, true, 6),
            7 => ConnectionGene(1, 2, 0.5, true, 7),
        )
        gb = GraphGenome(nodes_b, conns_b, 1, 1, Inf)

        # With c1=1, c2=1, c3=0: distance = E/N + D/N = 2/5 + 3/5 = 1.0
        # (N = max(4,5) = 5, below 20 threshold so N=1.0 is NOT used)
        # Wait: N < 20 → N = 1.0. So distance = 2*1.0 + 3*1.0 = 5.0
        d = Arborist._neat_distance(ga, gb; c1=1.0, c2=1.0, c3=0.0)
        # E=2 excess, D=3 disjoint, N=1.0 (both genomes < 20 connections)
        @test d ≈ 2.0 * 1.0 + 3.0 * 1.0  # c1*E/N + c2*D/N = 2+3 = 5.0
        println("  NEAT distance disjoint/excess: E=2, D=3, d=$d")

        # With different c1/c2, excess and disjoint are weighted differently
        d2 = Arborist._neat_distance(ga, gb; c1=2.0, c2=0.5, c3=0.0)
        @test d2 ≈ 2.0 * 2.0 + 0.5 * 3.0  # 4.0 + 1.5 = 5.5
        println("  NEAT distance with c1=2, c2=0.5: d=$d2")
        flush(stdout)
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

    @testset "deserialize round-trip (structural + functional)" begin
        reset_innovation_counter!()
        rng = Random.MersenneTwister(17)
        g = initialize(GraphGenome, 2, 1, rng)
        # Grow the topology a bit so it's non-trivial.
        for _ in 1:5
            Arborist._mutate_weights!(g, rng)
        end
        Arborist._mutate_add_node!(g, rng)
        Arborist._mutate_add_connection!(g, rng)

        s = serialize(g)
        g2 = deserialize(GraphGenome, s, g.n_inputs, g.n_outputs)
        @test g2 !== nothing

        # Structural equality.
        @test keys(g.nodes) == keys(g2.nodes)
        for (id, n) in g.nodes
            n2 = g2.nodes[id]
            @test n2.type == n.type
            @test n2.activation == n.activation
        end
        @test keys(g.connections) == keys(g2.connections)
        for (inn, c) in g.connections
            c2 = g2.connections[inn]
            @test c2.in_node == c.in_node
            @test c2.out_node == c.out_node
            @test c2.weight == c.weight
            @test c2.enabled == c.enabled
            @test c2.innovation == c.innovation
        end

        # Functional equivalence on a small input set.
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)
        f1 = evaluate_genome(g, evaluator)
        f2 = evaluate_genome(g2, evaluator)
        @test f1 == f2
    end

    @testset "deserialize with reassign_innovations" begin
        reset_innovation_counter!()
        rng = Random.MersenneTwister(33)
        g = initialize(GraphGenome, 2, 1, rng)
        s = serialize(g)
        # Bump the global counter so fresh IDs would differ.
        reset_innovation_counter!()
        Arborist.init_innovation_range!(1000)

        g2 = deserialize(GraphGenome, s, g.n_inputs, g.n_outputs;
                         reassign_innovations=true)
        @test g2 !== nothing
        # Node IDs unchanged.
        @test keys(g.nodes) == keys(g2.nodes)
        # Connection count preserved but innovation keys are fresh.
        @test length(g2.connections) == length(g.connections)
        for c in values(g2.connections)
            @test c.innovation > 1000
        end
    end

    @testset "deserialize tolerates whitespace + preamble" begin
        reset_innovation_counter!()
        rng = Random.MersenneTwister(51)
        g = initialize(GraphGenome, 2, 1, rng)
        s = serialize(g)
        messy = "Here's the mutated genome:\n\n```\n" * s * "\n```\n"
        g2 = deserialize(GraphGenome, messy, g.n_inputs, g.n_outputs)
        @test g2 !== nothing
        @test keys(g.connections) == keys(g2.connections)
    end

    @testset "deserialize rejects malformed input" begin
        # Missing node reference.
        bad = """
        N 1 input identity
        N 3 output sigmoid
        C 1->2 w=0.5 en=true i=1
        """
        @test deserialize(GraphGenome, bad, 1, 1) === nothing

        # Wrong input count.
        ok = """
        N 1 input identity
        N 2 output sigmoid
        C 1->2 w=0.5 en=true i=1
        """
        @test deserialize(GraphGenome, ok, 2, 1) === nothing  # only 1 input decoded

        # Duplicate innovation ID.
        dup = """
        N 1 input identity
        N 2 output sigmoid
        C 1->2 w=0.5 en=true i=1
        C 1->2 w=0.6 en=true i=1
        """
        @test deserialize(GraphGenome, dup, 1, 1) === nothing

        # Garbled weight.
        garbled = """
        N 1 input identity
        N 2 output sigmoid
        C 1->2 w=NaNsense en=true i=1
        """
        @test deserialize(GraphGenome, garbled, 1, 1) === nothing
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
        ops = neat_defaults()
        algorithm = GeneticProgramming(
            pop_size=30, generations=10,
            mutation_rate=0.5, crossover_rate=0.2,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
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
        mut = NEATDefaultMutation()
        for trial in 1:5
            reset_innovation_counter!()
            rng1 = Random.MersenneTwister(trial)
            g1 = initialize(GraphGenome, 3, 2, rng1)
            # Run 50 mutations to exercise all mutation types
            for _ in 1:50
                g1 = mutate(mut, g1, rng1)
            end
            weights1 = sort([c.weight for c in values(g1.connections)])

            reset_innovation_counter!()
            rng2 = Random.MersenneTwister(trial)
            g2 = initialize(GraphGenome, 3, 2, rng2)
            for _ in 1:50
                g2 = mutate(mut, g2, rng2)
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
        mut = NEATDefaultMutation()
        xo = NEATCrossover()
        # Mutate to create structural differences (disjoint innovations)
        for _ in 1:15
            g1 = mutate(mut, g1, rng)
        end
        for _ in 1:15
            g2 = mutate(mut, g2, rng)
        end
        g1.fitness = 0.3; g2.fitness = 0.9
        c1, c2 = crossover(xo, g1, g2, rng)
        # Children should have different connection sets because
        # c1 gets fitter parent's disjoint/excess, c2 gets other parent's
        inns1 = Set(keys(c1.connections))
        inns2 = Set(keys(c2.connections))
        @test inns1 != inns2
        println("  Crossover distinct children: $(length(inns1)) vs $(length(inns2)) innovations")
        flush(stdout)
    end

    @testset "crossover determinism (sorted Set iteration)" begin
        mut = NEATDefaultMutation()
        xo = NEATCrossover()
        for trial in 1:5
            reset_innovation_counter!()
            rng1 = Random.MersenneTwister(100 + trial)
            g1 = initialize(GraphGenome, 2, 1, rng1)
            g2 = initialize(GraphGenome, 2, 1, rng1)
            # Mutate to create structural differences
            for _ in 1:10
                g1 = mutate(mut, g1, rng1)
                g2 = mutate(mut, g2, rng1)
            end
            g1.fitness = 0.5; g2.fitness = 1.0
            c1a, c2a = crossover(xo, g1, g2, rng1)

            reset_innovation_counter!()
            rng2 = Random.MersenneTwister(100 + trial)
            g3 = initialize(GraphGenome, 2, 1, rng2)
            g4 = initialize(GraphGenome, 2, 1, rng2)
            for _ in 1:10
                g3 = mutate(mut, g3, rng2)
                g4 = mutate(mut, g4, rng2)
            end
            g3.fitness = 0.5; g4.fitness = 1.0
            c1b, c2b = crossover(xo, g3, g4, rng2)

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

    @testset "operator dispatch — individual NEAT operators" begin
        # Each NEAT mutation operator runs standalone and produces a valid GraphGenome.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(7)
        g = initialize(GraphGenome, 3, 2, rng)
        # Grow a few connections so add_node has enabled edges to split.
        for _ in 1:5
            g = mutate(AddConnectionMutation(), g, rng)
        end

        wp = mutate(WeightPerturbMutation(), g, rng)
        @test wp isa GraphGenome
        @test length(wp.connections) == length(g.connections)

        wr = mutate(WeightReplaceMutation(), g, rng)
        @test wr isa GraphGenome

        ac = mutate(AddConnectionMutation(), g, rng)
        @test ac isa GraphGenome
        @test length(ac.connections) >= length(g.connections)

        an = mutate(AddNodeMutation(), g, rng)
        @test an isa GraphGenome
        # add-node inserts one hidden node + two connections when an enabled
        # edge exists; with no enabled edge it's a no-op.
        @test length(an.nodes) >= length(g.nodes)

        tc = mutate(ToggleConnectionMutation(), g, rng)
        @test tc isa GraphGenome
        @test length(tc.connections) == length(g.connections)
    end

    @testset "NEATDefaultMutation reproduces legacy branching" begin
        # The composite operator with default rates (0.80/0.10/0.05/0.03/0.02)
        # should trace the same RNG-consumption path that the old bare
        # `mutate(g, rng)` did — if both were run with the same seed we'd get
        # the same result. Here we verify that NEATDefaultMutation is
        # deterministic across two independent runs with the same seed.
        mut = NEATDefaultMutation()
        reset_innovation_counter!()
        rng1 = Random.MersenneTwister(2026)
        g1 = initialize(GraphGenome, 3, 2, rng1)
        for _ in 1:40
            g1 = mutate(mut, g1, rng1)
        end

        reset_innovation_counter!()
        rng2 = Random.MersenneTwister(2026)
        g2 = initialize(GraphGenome, 3, 2, rng2)
        for _ in 1:40
            g2 = mutate(mut, g2, rng2)
        end

        w1 = sort([c.weight for c in values(g1.connections)])
        w2 = sort([c.weight for c in values(g2.connections)])
        @test length(w1) == length(w2)
        @test w1 ≈ w2
    end

    @testset "NEATDefaultMutation rate validation" begin
        @test_throws ArgumentError NEATDefaultMutation(
            weight_perturb_rate=0.5, weight_replace_rate=0.1,
            add_connection_rate=0.05, add_node_rate=0.03, toggle_rate=0.02)  # sums to 0.70
    end

    @testset "_validate_ops rejects ExprGenome ops for GraphGenome" begin
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        # Default GeneticProgramming uses SubtreeMutation/PointMutation/SubtreeCrossover
        # which dispatch on ExprGenome, not GraphGenome. We expect a clear error.
        bad = GeneticProgramming(pop_size=10, generations=2)
        problem = GPProblem(evaluator, GraphGenome; seed=1)

        err = try
            solve(problem, bad; verbose=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("mutation operator", msg) || occursin("crossover operator", msg)
        @test occursin("neat_defaults", msg)
    end

    @testset "neat_defaults returns operator-compatible vectors" begin
        ops = neat_defaults()
        @test ops.mutation_ops isa Vector{AbstractMutationOperator}
        @test ops.crossover_ops isa Vector{AbstractCrossoverOperator}
        @test !isempty(ops.mutation_ops)
        @test !isempty(ops.crossover_ops)
        # Smoke test: validation passes with neat_defaults.
        @test Arborist._validate_ops(ops.mutation_ops, ops.crossover_ops, GraphGenome) === nothing
    end

    @testset "GraphEvaluator recurrent mode accepts cycles" begin
        # Manually-built genome with a cycle (hidden node self-loops through output):
        # inputs: 1 (x1), 2 (bias), output: 3; hidden node 4 with self-cycle 4->4
        nodes = Dict(
            1 => NodeGene(1, :input, :identity),
            2 => NodeGene(2, :bias,  :identity),
            3 => NodeGene(3, :output, :sigmoid),
            4 => NodeGene(4, :hidden, :tanh),
        )
        conns = Dict(
            1 => ConnectionGene(1, 4, 0.5, true, 1),
            2 => ConnectionGene(2, 4, 0.1, true, 2),
            3 => ConnectionGene(4, 4, 0.3, true, 3),   # self-cycle (recurrent)
            4 => ConnectionGene(4, 3, 0.7, true, 4),
        )
        g = GraphGenome(nodes, conns, 1, 1, Inf)

        input_data  = Float64[0.0 0.5 1.0]  # 1 input, 3 timesteps
        output_data = Float64[0.2 0.4 0.6]  # target values

        # Feedforward evaluator: cycle → Inf
        ff_eval = GraphEvaluator(input_data, output_data)
        @test evaluate_genome(g, ff_eval) == Inf

        # Recurrent evaluator: finite MSE
        rec_eval = GraphEvaluator(input_data, output_data;
                                  allow_recurrent=true, relaxation_passes=2)
        fitness = evaluate_genome(g, rec_eval)
        @test isfinite(fitness)
        @test fitness >= 0.0
        println("  recurrent XOR-like toy: MSE=$(round(fitness, digits=6))")
        flush(stdout)
    end

    @testset "GraphEvaluator recurrent mode preserves feedforward XOR" begin
        # Feedforward default must still behave identically on XOR data.
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        ev_default = GraphEvaluator(input_data, output_data)
        ev_explicit = GraphEvaluator(input_data, output_data;
                                      allow_recurrent=false, relaxation_passes=1)
        reset_innovation_counter!()
        rng = Random.MersenneTwister(99)
        g = initialize(GraphGenome, 2, 1, rng)
        f_default  = evaluate_genome(g, ev_default)
        f_explicit = evaluate_genome(g, ev_explicit)
        @test f_default ≈ f_explicit
        @test isfinite(f_default)
    end

    @testset "GraphEvaluator relaxation_passes validation" begin
        input_data  = Float64[0.0 1.0]
        output_data = Float64[0.0 1.0]
        @test_throws ArgumentError GraphEvaluator(input_data, output_data;
                                                    allow_recurrent=true,
                                                    relaxation_passes=0)
    end

    @testset "GraphEvaluator data-shape validation" begin
        # Extra target columns (review reproduction case): 2 input samples, 3 target samples.
        @test_throws ArgumentError GraphEvaluator(
            reshape([1.0, 2.0], 1, 2),
            reshape([1.0, 2.0, 999.0], 1, 3))

        # Missing target columns: 3 input samples, 2 target samples.
        @test_throws ArgumentError GraphEvaluator(
            reshape([1.0, 2.0, 3.0], 1, 3),
            reshape([1.0, 2.0], 1, 2))

        # Zero samples in input_data.
        @test_throws ArgumentError GraphEvaluator(
            Matrix{Float64}(undef, 1, 0),
            Matrix{Float64}(undef, 1, 0))

        # Zero input rows.
        @test_throws ArgumentError GraphEvaluator(
            Matrix{Float64}(undef, 0, 2),
            reshape([1.0, 2.0], 1, 2))

        # Zero output rows.
        @test_throws ArgumentError GraphEvaluator(
            reshape([1.0, 2.0], 1, 2),
            Matrix{Float64}(undef, 0, 2))

        # Sanity: matched-shape construction still succeeds.
        ev = GraphEvaluator(reshape([1.0, 2.0], 1, 2),
                            reshape([1.0, 2.0], 1, 2))
        @test ev isa GraphEvaluator
    end

    @testset "ACTIVATION_FNS CPPN additions" begin
        fns = Arborist.ACTIVATION_FNS
        # All eight documented activations must resolve and be callable.
        for key in (:sigmoid, :tanh, :relu, :identity, :gauss, :sin, :abs, :step)
            @test haskey(fns, key)
            @test fns[key] isa Function
        end

        # Evaluate each CPPN-flavored activation at a grid of inputs and check
        # the numeric value agrees with the defining formula to 1e-10.
        xs = (-2.0, -0.5, 0.0, 0.5, 2.0)
        for x in xs
            @test fns[:gauss](x) ≈ exp(-x * x)   atol = 1e-10
            @test fns[:sin](x)   ≈ sin(x)        atol = 1e-10
            @test fns[:abs](x)   ≈ abs(x)        atol = 1e-10
            @test fns[:step](x)  ≈ (x > 0.0 ? 1.0 : 0.0)
        end

        # The step function is discontinuous at 0; confirm the convention
        # (strictly greater than 0 produces 1, so step(0) == 0).
        @test fns[:step](0.0) == 0.0
        @test fns[:step](-1.0e-12) == 0.0
        @test fns[:step](1.0e-12) == 1.0

        # End-to-end: a NodeGene carrying one of the new activations should
        # evaluate correctly via the standard GraphEvaluator path. Use a
        # gauss-activated output node with zero-weight edges so the pre-
        # activation net is 0 and the output is gauss(0) = 1.0.
        nodes = Dict(
            1 => NodeGene(1, :input,  :identity),
            2 => NodeGene(2, :bias,   :identity),
            3 => NodeGene(3, :output, :gauss),
        )
        conns = Dict(
            1 => ConnectionGene(1, 3, 0.0, true, 1),
            2 => ConnectionGene(2, 3, 0.0, true, 2),
        )
        g = GraphGenome(nodes, conns, 1, 1, Inf)
        ev = GraphEvaluator(reshape([0.0], 1, 1), reshape([1.0], 1, 1))
        # Evaluator returns MSE; gauss(0) = 1, target = 1, MSE = 0.
        @test evaluate_genome(g, ev) ≈ 0.0  atol = 1e-10
    end
end
