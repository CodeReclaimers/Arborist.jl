@testset "Speciation" begin
    @testset "NoSpeciation" begin
        ns = NoSpeciation()
        @test ns isa AbstractSpeciation
    end

    @testset "ThresholdSpeciation construction" begin
        ts = ThresholdSpeciation()
        @test ts isa AbstractSpeciation
        @test ts.threshold == 10.0
        @test ts.min_species_size == 2
        @test ts.stagnation_limit == 15

        ts2 = ThresholdSpeciation(threshold=5.0, min_species_size=3, stagnation_limit=10)
        @test ts2.threshold == 5.0
        @test ts2.min_species_size == 3
        @test ts2.stagnation_limit == 10
    end

    @testset "ThresholdSpeciation species assignment" begin
        # Create genomes with known structure for distance computation.
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        add!(fset, :-, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        # Create several genomes.
        genomes = [ExprGenome([create_random_assignment(s) for _ in 1:3], s) for _ in 1:20]
        fitnesses = [Float64(i) for i in 1:20]  # simple fitness ordering

        spec = ThresholdSpeciation(threshold=5.0, min_species_size=1, stagnation_limit=100)
        species_state = Arborist._init_species_state(spec)

        shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)

        # Species should have been created.
        @test length(species_state) > 0
        # Shared fitnesses should differ from raw when species have multiple members.
        @test length(shared) == length(fitnesses)
        # All shared fitnesses should be non-negative.
        @test all(f -> f >= 0.0, shared)
    end

    @testset "NoSpeciation returns raw fitnesses" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        genomes = [ExprGenome([create_random_assignment(s)], s) for _ in 1:5]
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = Arborist._apply_speciation!(genomes, fitnesses, NoSpeciation(), nothing, rng)
        @test result === fitnesses
    end

    @testset "Fitness sharing penalizes large species (minimization)" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        # Create identical genomes so they all land in the same species.
        body = [:(y = x + Float32(1.0))]
        genomes = [ExprGenome(deepcopy(body), s) for _ in 1:10]
        fitnesses = fill(2.0, 10)

        spec = ThresholdSpeciation(threshold=100.0)  # large threshold → all in one species
        species_state = Arborist._init_species_state(spec)

        shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)

        # All in one species of size 10. For minimization (lower=better),
        # shared = raw * species_size = 2.0 * 10 = 20.0.
        # This makes large-species members appear WORSE, protecting
        # structural innovations in small species.
        @test all(f -> f ≈ 20.0, shared)
    end

    @testset "Stagnation tracking" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        body = [:(y = x + Float32(1.0))]
        genomes = [ExprGenome(deepcopy(body), s) for _ in 1:5]
        fitnesses = fill(2.0, 5)

        spec = ThresholdSpeciation(threshold=100.0, stagnation_limit=3)
        species_state = Arborist._init_species_state(spec)

        # Run speciation multiple times with no improvement.
        for _ in 1:5
            Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)
        end

        # Species should have accumulated stagnation.
        @test species_state[1].stagnation >= 4
    end

    @testset "ThresholdSpeciation integrates with solve" begin
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
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = GeneticProgramming(
            pop_size=30,
            generations=10,
            mutation_rate=0.4,
            crossover_rate=0.2,
            speciation=ThresholdSpeciation(threshold=5.0)
        )

        # Should not error.
        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test length(result.fitness_history) == 10
        @test result.best_fitness >= 0.0
    end
end
