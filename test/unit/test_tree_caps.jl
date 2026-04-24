using DynamicExpressions: Node, OperatorEnum

@testset "tree_depth + operator caps" begin
    # ------------------------------------------------------------------
    # tree_depth: per-genome implementations
    # ------------------------------------------------------------------
    @testset "tree_depth(::TreeGenome)" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin, cos])
        leaf  = TreeGenome(Node(Float64; feature=1), ops, 1)
        @test tree_depth(leaf) == 1
        binary = TreeGenome(Node(1, Node(Float64; feature=1),
                                     Node(Float64; feature=1)), ops, 1)
        @test tree_depth(binary) == 2
        # sin( +(x, x) )
        deep = TreeGenome(Node(1, Node(1, Node(Float64; feature=1),
                                           Node(Float64; feature=1))), ops, 1)
        @test tree_depth(deep) == 3
    end

    @testset "tree_depth(::ExprGenome) via _expr_depth" begin
        @test Arborist._expr_depth(:x) == 1
        @test Arborist._expr_depth(42) == 1
        @test Arborist._expr_depth(:(a = 1)) == 1
        @test Arborist._expr_depth(:(a = x + 1)) == 2
        @test Arborist._expr_depth(:(a = sin(x + 1))) == 3
        @test Arborist._expr_depth(:(a = sin(cos(tan(x))))) == 4
    end

    # ------------------------------------------------------------------
    # Operator caps: opting in via kwargs does not break zero-arg callers
    # ------------------------------------------------------------------
    @testset "operator kwargs-only constructors preserve no-arg form" begin
        # These are the no-arg calls existing code relies on; they must still work.
        @test SubtreeMutation() isa SubtreeMutation
        @test PointMutation() isa PointMutation
        @test HoistMutation() isa HoistMutation
        @test ExpansionMutation() isa ExpansionMutation
        @test SubtreeCrossover() isa SubtreeCrossover
        # And the fields default to nothing — meaning no cap.
        for op in (SubtreeMutation(), PointMutation(), HoistMutation(),
                   ExpansionMutation(), SubtreeCrossover())
            @test op.max_depth === nothing
            @test op.max_size === nothing
        end
    end

    # ------------------------------------------------------------------
    # ExprGenome cap enforcement
    # ------------------------------------------------------------------
    function _expr_genome_fixture()
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*]
            add!(fset, func, 2, Float32, Float32)
        end
        s = GenState(fset, Dict(:x => Float32), Dict(:y => Float32), 4)
        return (s, fset)
    end

    @testset "SubtreeMutation max_size caps ExprGenome output" begin
        (s, _) = _expr_genome_fixture()
        rng = Random.MersenneTwister(123)
        parent = ExprGenome([create_random_assignment(s) for _ in 1:3], s)
        op = SubtreeMutation(max_size=5)
        # A size cap below the parent's own size forces every mutation whose
        # result exceeds 5 nodes to revert to the parent. Parent size may
        # already be > 5 — in that case the cap is trivially violated and the
        # mutation always reverts, which is also a valid cap-enforcement case.
        for _ in 1:50
            g2 = mutate(op, parent, rng)
            @test complexity(g2) <= 5 || g2 === parent
        end
    end

    @testset "SubtreeCrossover max_depth caps ExprGenome output" begin
        (s, _) = _expr_genome_fixture()
        rng = Random.MersenneTwister(321)
        g1 = ExprGenome([create_random_assignment(s) for _ in 1:3], s)
        g2 = ExprGenome([create_random_assignment(s) for _ in 1:3], s)
        op = SubtreeCrossover(max_depth=2)
        for _ in 1:50
            (c1, c2) = crossover(op, g1, g2, rng)
            @test tree_depth(c1) <= 2 || c1 === g1
            @test tree_depth(c2) <= 2 || c2 === g2
        end
    end

    @testset "behavior-preserving: no caps == old behavior" begin
        # Running the same mutation twice, once with SubtreeMutation() and
        # once with SubtreeMutation(max_depth=nothing, max_size=nothing),
        # must produce bit-identical genomes. ExprGenome mutation goes
        # through codegen helpers that read `Random.default_rng()` as well
        # as the passed RNG, so we reseed the global RNG before each call.
        (s, _) = _expr_genome_fixture()
        Random.seed!(12345)
        parent = ExprGenome([create_random_assignment(s) for _ in 1:3], s)

        Random.seed!(99)
        rng_a = Random.MersenneTwister(999)
        g_a = mutate(SubtreeMutation(), parent, rng_a)

        Random.seed!(99)
        rng_b = Random.MersenneTwister(999)
        g_b = mutate(SubtreeMutation(max_depth=nothing, max_size=nothing),
                     parent, rng_b)

        @test complexity(g_a) == complexity(g_b)
        @test tree_depth(g_a) == tree_depth(g_b)
        @test repr(g_a.body) == repr(g_b.body)
    end

    # ------------------------------------------------------------------
    # TreeGenome cap enforcement via the shared operator
    # ------------------------------------------------------------------
    @testset "SubtreeMutation caps TreeGenome output" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin, cos])
        # Build a deep TreeGenome so mutations have room to go either way.
        deep = Node(1, Node(1, Node(Float64; feature=1),
                               Node(Float64; feature=1)),
                       Node(Float64; feature=1))
        parent = TreeGenome(deep, ops, 1)
        rng = Random.MersenneTwister(222)
        op = SubtreeMutation(max_size=2)
        for _ in 1:50
            child = mutate(op, parent, rng)
            # Cap respected OR reverted to the parent.
            @test complexity(child) <= 2 || child === parent
        end
    end
end
