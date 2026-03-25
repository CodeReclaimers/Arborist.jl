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
end
