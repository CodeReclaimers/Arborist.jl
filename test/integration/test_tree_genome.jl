# TreeGenome integration tests — requires DynamicExpressions.jl.
using DynamicExpressions

@testset "TreeGenome integration" begin

    @testset "TreeGenome construction" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin, cos])
        x1 = Node{Float32}(; feature=1)
        c = Node{Float32}(; val=2.0f0)
        tree = Node{Float32}(; op=1, l=x1, r=c)
        g = TreeGenome{Float32}(tree, ops, 1)
        @test g isa AbstractGenome
        @test complexity(g) == 3.0
    end

    @testset "TreeFitnessEvaluator" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin])
        X = Float32[1.0 2.0 3.0; 0.0 0.0 0.0]
        y = Float32[2.0, 3.0, 4.0]  # y = x1 + 1
        evaluator = TreeFitnessEvaluator(X, y, ops)

        # Perfect tree: x1 + 1.0
        tree = Node{Float32}(; op=1, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=1.0f0))
        g = TreeGenome{Float32}(tree, ops, 2)
        @test evaluate(evaluator, g) ≈ 0.0

        # Constant tree: 0.0 — MSE = mean((0 - y)^2)
        g_const = TreeGenome{Float32}(Node{Float32}(; val=0.0f0), ops, 2)
        expected_mse = sum(y .^ 2) / length(y)
        @test evaluate(evaluator, g_const) ≈ expected_mse
    end

    @testset "TreeGenome mutate" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin])
        rng = Random.MersenneTwister(42)
        tree = Node{Float32}(; op=1, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=1.0f0))
        g = TreeGenome{Float32}(tree, ops, 2)

        for _ in 1:20
            g2 = mutate(g, rng)
            @test g2 isa TreeGenome{Float32}
            @test g2.operators === g.operators
        end
    end

    @testset "TreeGenome crossover" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin])
        rng = Random.MersenneTwister(42)
        t1 = Node{Float32}(; op=1, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=1.0f0))
        t2 = Node{Float32}(; op=2, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=2.0f0))
        g1 = TreeGenome{Float32}(t1, ops, 2)
        g2 = TreeGenome{Float32}(t2, ops, 2)

        for _ in 1:20
            (c1, c2) = crossover(g1, g2, rng)
            @test c1 isa TreeGenome{Float32}
            @test c2 isa TreeGenome{Float32}
        end
    end

    @testset "TreeGenome distance and serialize" begin
        ops = OperatorEnum(; binary_operators=[+, -], unary_operators=[sin])
        t1 = Node{Float32}(; feature=1)
        t2 = Node{Float32}(; op=1, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=1.0f0))
        g1 = TreeGenome{Float32}(t1, ops, 1)
        g2 = TreeGenome{Float32}(t2, ops, 1)

        d = distance(g1, g2)
        @test d isa Float64
        @test d >= 0.0
        @test distance(g1, g1) == 0.0

        s = serialize(g1)
        @test s isa String
    end

    @testset "TreeGenome solve (quick)" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
        xs = Float32.(range(-1, 1, length=20))
        X = reshape(xs, 1, :)
        y = xs .^ 2

        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        algorithm = GeneticProgramming(
            pop_size=50, generations=20,
            mutation_rate=0.4, crossover_rate=0.2
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{TreeGenome{Float32}}
        @test result.best_fitness >= 0.0
        @test result.best_fitness < Inf
        @test length(result.fitness_history) == 20
        @test result.wall_time > 0.0
    end

    @testset "TreeGenome serialize/deserialize round-trip" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin, cos])

        # `deserialize` accepts both the infix form emitted by `serialize` /
        # DynamicExpressions' `string_tree` (e.g. "x1 + 1.0") and the
        # prefix s-expression form used by older callers (e.g. "+(x1, 1.0)").
        # `Meta.parse` normalizes both to the same `Expr(:call, ...)` shape,
        # so the parser handles them with one walker. The prefix tests below
        # remain as the regression guard for the old form.

        # Prefix format: binary
        g1 = deserialize(TreeGenome{Float32}, "+(x1, 1.0)", ops, 2)
        @test g1 !== nothing
        @test g1 isa TreeGenome{Float32}
        println("  TreeGenome prefix parse: '+(x1, 1.0)' -> '$(serialize(g1))'")

        # Prefix format: unary
        g2 = deserialize(TreeGenome{Float32}, "sin(x1)", ops, 2)
        @test g2 !== nothing
        @test serialize(g2) == "sin(x1)"
        println("  TreeGenome unary round-trip: '$(serialize(g2))'")

        # Nested prefix: sin(*(x1, x2))
        g3 = deserialize(TreeGenome{Float32}, "sin(*(x1, x2))", ops, 2)
        @test g3 !== nothing
        println("  TreeGenome nested prefix: '$(serialize(g3))'")

        # Feature variable only
        g4 = deserialize(TreeGenome{Float32}, "x1", ops, 2)
        @test g4 !== nothing
        @test serialize(g4) == "x1"

        # Numeric constant
        g5 = deserialize(TreeGenome{Float32}, "3.14", ops, 2)
        @test g5 !== nothing

        # Invalid input returns nothing.
        @test deserialize(TreeGenome{Float32}, "not a tree", ops, 2) === nothing
        @test deserialize(TreeGenome{Float32}, "", ops, 2) === nothing

        # ---- Semantic round-trip via string_tree (the previously broken path) ----
        # `serialize` emits DynamicExpressions' infix; `deserialize` must
        # reconstruct a tree whose predictions are bit-identical to the source.
        rt_ops = OperatorEnum(; binary_operators=[+, -, *, /],
                                unary_operators=[sin, cos, abs])
        rt_X = reshape(Float32.(range(-2, 2, length=25)), 1, :)

        # `isequal` rather than `==`: random trees can hit `1/x1` at x1==0
        # and produce `NaN`, which compares unequal to itself under `==`.
        # `isequal` treats matching `NaN`s as equal, which is what we want
        # for a bit-level round-trip check.
        function round_trip_eq(g::TreeGenome{Float32})
            s = serialize(g)
            g2 = deserialize(TreeGenome{Float32}, s, g.operators, g.n_features)
            g2 === nothing && return false
            p1 = g.tree(rt_X, g.operators)
            p2 = g2.tree(rt_X, g2.operators)
            return isequal(p1, p2)
        end

        # Single terminal (feature and constant)
        @test round_trip_eq(TreeGenome{Float32}(Node{Float32}(; feature=1), rt_ops, 1))
        @test round_trip_eq(TreeGenome{Float32}(Node{Float32}(; val=2.5f0), rt_ops, 1))

        # The historical failure case: x1 + 1.0
        let
            t = Node{Float32}(; op=1, l=Node{Float32}(; feature=1),
                                 r=Node{Float32}(; val=1.0f0))
            @test round_trip_eq(TreeGenome{Float32}(t, rt_ops, 1))
        end

        # All four binary ops in isolation
        for bi in 1:4
            t = Node{Float32}(; op=UInt8(bi),
                                 l=Node{Float32}(; feature=1),
                                 r=Node{Float32}(; val=2.0f0))
            @test round_trip_eq(TreeGenome{Float32}(t, rt_ops, 1))
        end

        # All three unary ops in isolation
        for ui in 1:3
            t = Node{Float32}(; op=UInt8(ui), l=Node{Float32}(; feature=1))
            @test round_trip_eq(TreeGenome{Float32}(t, rt_ops, 1))
        end

        # Nested: sin((x1 + 1) * x1)
        let
            inner = Node{Float32}(; op=1, l=Node{Float32}(; feature=1),
                                     r=Node{Float32}(; val=1.0f0))
            prod  = Node{Float32}(; op=3, l=inner, r=Node{Float32}(; feature=1))
            top   = Node{Float32}(; op=1, l=prod)  # sin(...)
            @test round_trip_eq(TreeGenome{Float32}(top, rt_ops, 1))
        end

        # Random trees from `_random_tree`
        rt_rng = Random.MersenneTwister(42)
        for _ in 1:10
            tree = Arborist._random_tree(rt_rng, rt_ops, 1, Float32, 4, :grow)
            @test round_trip_eq(TreeGenome{Float32}(tree, rt_ops, 1))
        end
    end

    @testset "TreeGenome crossover preserves correct operators per parent" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin])
        rng = Random.MersenneTwister(42)
        t1 = Node{Float32}(; op=1, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=1.0f0))
        t2 = Node{Float32}(; op=2, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=2.0f0))
        g1 = TreeGenome{Float32}(t1, ops, 2)
        g2 = TreeGenome{Float32}(t2, ops, 2)

        for _ in 1:20
            (c1, c2) = crossover(g1, g2, rng)
            # Verify that offspring reference the correct parent's operators.
            @test c1.operators === g1.operators
            @test c2.operators === g2.operators
        end
        println("  TreeGenome crossover: operator references verified")
    end

    @testset "SubtreeMutation/PointMutation dispatch on TreeGenome" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sin])
        rng = Random.MersenneTwister(42)
        tree = Node{Float32}(; op=1, l=Node{Float32}(; feature=1), r=Node{Float32}(; val=1.0f0))
        g = TreeGenome{Float32}(tree, ops, 2)

        g2 = mutate(SubtreeMutation(), g, rng)
        @test g2 isa TreeGenome{Float32}
        g3 = mutate(PointMutation(), g, rng)
        @test g3 isa TreeGenome{Float32}
    end
end
