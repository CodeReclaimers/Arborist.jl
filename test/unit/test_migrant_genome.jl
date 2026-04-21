@testset "MigrantGenome" begin
    @testset "ExprGenome round-trip" begin
        rng = Random.MersenneTwister(42)
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        add!(fset, :-, 2, Float32, Float32)
        inputs = Dict(:x => Float32)
        outputs = Dict(:y => Float32)
        state = GenState(rng, fset, inputs, outputs, 2)

        body = Expr[:(y = x + __temp_1)]
        genome = ExprGenome(body, state)

        migrant = to_migrant(genome, 0.5)
        @test migrant isa MigrantGenome
        @test migrant.fitness == 0.5
        @test migrant.genome_type == :ExprGenome
        @test migrant.data isa Vector{Expr}

        # Reconstruct on receiving side
        rng2 = Random.MersenneTwister(99)
        state2 = GenState(rng2, fset, inputs, outputs, 2)
        reconstructed = from_migrant(migrant, state2)
        @test reconstructed isa ExprGenome
        @test length(reconstructed.body) == 1
        @test reconstructed.state === state2  # uses local state, not source state
    end

    @testset "MigrantGenome data is deep-copied" begin
        rng = Random.MersenneTwister(42)
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        inputs = Dict(:x => Float32)
        outputs = Dict(:y => Float32)
        state = GenState(rng, fset, inputs, outputs, 2)

        body = Expr[:(y = x + __temp_1), :(y = y + x)]
        genome = ExprGenome(body, state)
        migrant = to_migrant(genome, 1.0)

        # Modifying original body should not affect migrant
        push!(genome.body, :(y = y))
        @test length(migrant.data) == 2
        @test length(genome.body) == 3
    end

    @testset "AntGenome round-trip" begin
        program = Expr(:call, :gp_ant_move, true)
        genome = AntGenome(program, [:gp_ant_move, :gp_ant_left, :gp_ant_right],
                           [:gp_ant_food_ahead], 4)
        migrant = to_migrant(genome, 3.0)
        @test migrant.genome_type == :AntGenome
        @test migrant.fitness == 3.0

        reconstructed = from_migrant(migrant, genome.primitives, genome.conditions, genome.max_depth)
        @test reconstructed isa AntGenome
        @test reconstructed.primitives == genome.primitives
        @test reconstructed.max_depth == genome.max_depth
    end

    @testset "GraphGenome round-trip" begin
        reset_innovation_counter!()
        rng = Random.MersenneTwister(42)
        g = initialize(GraphGenome, 3, 2, rng)
        # Evolve some structure so the round-trip exercises more than the
        # minimal fully-connected init.
        mut = NEATDefaultMutation()
        for _ in 1:20
            g = mutate(mut, g, rng)
        end
        g.fitness = 0.42

        migrant = to_migrant(g, g.fitness)
        @test migrant isa MigrantGenome
        @test migrant.fitness == 0.42
        @test migrant.genome_type === :GraphGenome

        reconstructed = from_migrant(migrant, GraphGenome)
        @test reconstructed isa GraphGenome
        @test reconstructed.n_inputs == g.n_inputs
        @test reconstructed.n_outputs == g.n_outputs
        @test reconstructed.fitness == g.fitness
        @test length(reconstructed.nodes) == length(g.nodes)
        @test length(reconstructed.connections) == length(g.connections)
        # Connection innovations should round-trip exactly.
        @test Set(keys(reconstructed.connections)) == Set(keys(g.connections))

        # Deep-copy check: mutating the reconstruction should not affect the original.
        first_inn = first(keys(reconstructed.connections))
        reconstructed.connections[first_inn].weight = 12345.0
        @test g.connections[first_inn].weight != 12345.0
    end
end
