# TreeGenome integration tests — requires DynamicExpressions.jl for extension loading.
using DynamicExpressions
const DynExt = Base.get_extension(Arborist, :DynExprExt)
const TreeGenome = DynExt.TreeGenome
const TreeFitnessEvaluator = DynExt.TreeFitnessEvaluator

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
