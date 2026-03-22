@testset "Operators" begin

    function make_test_genomes()
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end
        s = GenState(fset, Dict(:x => Float32), Dict(:y => Float32), 4)
        g1 = ExprGenome([create_random_assignment(s) for _ in 1:3], s)
        g2 = ExprGenome([create_random_assignment(s) for _ in 1:3], s)
        return (g1, g2, s)
    end

    @testset "SubtreeMutation" begin
        Random.seed!(42)
        (g1, _, s) = make_test_genomes()
        rng = Random.MersenneTwister(123)
        op = SubtreeMutation()
        for _ in 1:20
            g_mutated = mutate(op, g1, rng)
            @test g_mutated isa ExprGenome
            @test length(g_mutated.body) == length(g1.body)
            @test g_mutated.state === g1.state
        end
    end

    @testset "PointMutation" begin
        Random.seed!(42)
        (g1, _, s) = make_test_genomes()
        rng = Random.MersenneTwister(456)
        op = PointMutation()
        for _ in 1:20
            g_mutated = mutate(op, g1, rng)
            @test g_mutated isa ExprGenome
            @test g_mutated.state === g1.state
        end
    end

    @testset "SubtreeCrossover" begin
        Random.seed!(42)
        (g1, g2, s) = make_test_genomes()
        rng = Random.MersenneTwister(789)
        op = SubtreeCrossover()
        for _ in 1:20
            (c1, c2) = crossover(op, g1, g2, rng)
            @test c1 isa ExprGenome
            @test c2 isa ExprGenome
            @test !isempty(c1.body)
            @test !isempty(c2.body)
        end
    end

    @testset "TournamentSelection" begin
        sel = TournamentSelection(5)
        @test sel.tournament_size == 5
        @test sel isa AbstractSelectionStrategy
    end

    @testset "Mutation does not modify original genome" begin
        Random.seed!(42)
        (g1, _, s) = make_test_genomes()
        original_body_str = string(g1.body)
        rng = Random.MersenneTwister(111)
        mutate(SubtreeMutation(), g1, rng)
        @test string(g1.body) == original_body_str
        mutate(PointMutation(), g1, rng)
        @test string(g1.body) == original_body_str
    end

end
