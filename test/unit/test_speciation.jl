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
        @test ts.sharing_formula == :log2

        ts2 = ThresholdSpeciation(threshold=5.0, min_species_size=3,
                                   stagnation_limit=10, sharing_formula=:sqrt)
        @test ts2.threshold == 5.0
        @test ts2.min_species_size == 3
        @test ts2.stagnation_limit == 10
        @test ts2.sharing_formula == :sqrt
    end

    @testset "BehavioralSpeciation construction" begin
        fp_fn = g -> "fingerprint"
        dist_fn = (a, b) -> a == b ? 0.0 : 1.0

        bs = BehavioralSpeciation(fingerprint_fn=fp_fn, distance_fn=dist_fn)
        @test bs isa AbstractSpeciation
        @test bs.threshold == 0.15
        @test bs.min_species_size == 2
        @test bs.stagnation_limit == 15
        @test bs.sharing_formula == :sqrt

        bs2 = BehavioralSpeciation(fingerprint_fn=fp_fn, distance_fn=dist_fn,
                                    threshold=0.3, sharing_formula=:log2)
        @test bs2.threshold == 0.3
        @test bs2.sharing_formula == :log2
    end

    @testset "apply_sharing formulas" begin
        @test apply_sharing(1.0, 1, :linear) ≈ 1.0   # singletons unpenalized
        @test apply_sharing(1.0, 1, :sqrt) ≈ 1.0
        @test apply_sharing(1.0, 1, :log2) ≈ 1.0
        @test apply_sharing(1.0, 1, :none) ≈ 1.0

        @test apply_sharing(1.0, 4, :linear) ≈ 4.0
        @test apply_sharing(1.0, 4, :sqrt)   ≈ 2.0
        @test apply_sharing(1.0, 4, :log2)   ≈ 1.0 + log2(4)  # = 3.0
        @test apply_sharing(1.0, 4, :none)   ≈ 1.0

        @test apply_sharing(2.5, 8, :sqrt) ≈ 2.5 * sqrt(8)
        @test apply_sharing(2.5, 8, :log2) ≈ 2.5 * (1.0 + log2(8))  # 2.5 * 4.0 = 10.0

        @test_throws ArgumentError apply_sharing(1.0, 4, :unknown)
    end

    @testset "ThresholdSpeciation species assignment" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        add!(fset, :-, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        genomes = [ExprGenome([create_random_assignment(s) for _ in 1:3], s) for _ in 1:20]
        fitnesses = [Float64(i) for i in 1:20]

        spec = ThresholdSpeciation(threshold=5.0, min_species_size=1, stagnation_limit=100)
        species_state = Arborist._init_species_state(spec)

        shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)

        @test length(species_state) > 0
        @test length(shared) == length(fitnesses)
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

        # All in one species of size 10. Default :log2 sharing:
        # shared = raw * (1 + log2(species_size)) = 2.0 * (1 + log2(10)).
        expected = 2.0 * (1.0 + log2(10))
        @test all(f -> f ≈ expected, shared)
    end

    @testset "ThresholdSpeciation sharing_formula variants" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        body = [:(y = x + Float32(1.0))]
        genomes = [ExprGenome(deepcopy(body), s) for _ in 1:4]
        fitnesses = fill(3.0, 4)

        for (formula, expected_factor) in [
            (:linear, 4.0),
            (:sqrt, sqrt(4)),
            (:log2, 1.0 + log2(4)),
            (:none, 1.0),
        ]
            spec = ThresholdSpeciation(threshold=100.0, sharing_formula=formula)
            species_state = Arborist._init_species_state(spec)
            shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)
            @test all(f -> f ≈ 3.0 * expected_factor, shared)
        end
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

        for _ in 1:5
            Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)
        end

        @test species_state[1].stagnation >= 4
    end

    @testset "BehavioralSpeciation collapses identical behaviors" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        # Two different ASTs but same "behavior"
        body1 = [:(y = x + Float32(1.0))]
        body2 = [:(y = Float32(1.0) + x)]
        genomes = vcat(
            [ExprGenome(deepcopy(body1), s) for _ in 1:5],
            [ExprGenome(deepcopy(body2), s) for _ in 1:5]
        )
        fitnesses = collect(1.0:10.0)

        # Fingerprint is just "same" for all — same behavior
        fp_fn = g -> :same_behavior
        dist_fn = (a, b) -> a == b ? 0.0 : 1.0

        spec = BehavioralSpeciation(fingerprint_fn=fp_fn, distance_fn=dist_fn,
                                     threshold=0.5)
        species_state = Arborist._init_species_state(spec)
        shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)

        # All in one species since all fingerprints are identical
        @test length(species_state) == 1
        # With :sqrt sharing on species of 10: penalty = sqrt(10) ≈ 3.16
        for i in 1:10
            @test shared[i] ≈ fitnesses[i] * sqrt(10)
        end
    end

    @testset "BehavioralSpeciation separates different behaviors" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        body = [:(y = x + Float32(1.0))]
        genomes = [ExprGenome(deepcopy(body), s) for _ in 1:6]
        fitnesses = collect(1.0:6.0)

        # Fingerprint alternates: even/odd groups
        fp_fn = g -> hash(g.body) % 2  # won't reliably alternate but test the mechanism
        # Use a deterministic fingerprint instead
        counter = Ref(0)
        fp_fn_det = function(g)
            counter[] += 1
            return counter[] <= 3 ? :group_a : :group_b
        end
        dist_fn = (a, b) -> a == b ? 0.0 : 1.0

        spec = BehavioralSpeciation(fingerprint_fn=fp_fn_det, distance_fn=dist_fn,
                                     threshold=0.5)
        species_state = Arborist._init_species_state(spec)
        shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)

        # Should have 2 species: group_a (3 members) and group_b (3 members)
        @test length(species_state) == 2
    end

    @testset "BehavioralSpeciation with Hamming distance" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        body = [:(y = x + Float32(1.0))]
        genomes = [ExprGenome(deepcopy(body), s) for _ in 1:4]
        fitnesses = [1.0, 2.0, 3.0, 4.0]

        # Vector fingerprints with Hamming-like distance
        fps = [[1,2,3,4], [1,2,3,4], [5,6,7,8], [5,6,7,9]]
        idx = Ref(0)
        fp_fn = g -> (idx[] += 1; fps[idx[]])
        dist_fn = (a, b) -> sum(a[i] != b[i] for i in 1:length(a)) / length(a)

        spec = BehavioralSpeciation(fingerprint_fn=fp_fn, distance_fn=dist_fn,
                                     threshold=0.5)
        species_state = Arborist._init_species_state(spec)
        shared = Arborist._apply_speciation!(genomes, fitnesses, spec, species_state, rng)

        # Genomes 1,2 identical fingerprint → same species
        # Genome 3 different (distance=1.0 from genomes 1,2) → new species
        # Genome 4 close to genome 3 (distance=0.25) → same species as 3
        @test length(species_state) == 2
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

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test length(result.fitness_history) == 10
        @test result.best_fitness >= 0.0
    end
end
