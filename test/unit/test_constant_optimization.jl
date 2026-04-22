using Test
using Arborist
using Random
using DynamicExpressions

@testset "Constant optimization" begin
    @testset "config validation" begin
        @test_throws ArgumentError ConstantOptimization(frequency=0)
        @test_throws ArgumentError ConstantOptimization(top_k=0)
        @test_throws ArgumentError ConstantOptimization(max_iter=0)
        @test_throws ArgumentError ConstantOptimization(tol=0.0)
        @test_throws ArgumentError ConstantOptimization(fd_step=-1.0)

        c = ConstantOptimization()
        @test c.frequency == 25
        @test c.top_k == 5
        @test c.max_iter == 50
        @test c.tol ≈ 1e-8
        @test c.fd_step ≈ 1e-3
    end

    @testset "optimize_constants! recovers exact constants on a linear fit" begin
        # Target: y = 2.0*x + 1.0. Seed the tree with the right structure but
        # wrong constants (5.0 and -3.0). BFGS should recover (2.0, 1.0).
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        # Find operator indices dynamically.
        idx_plus  = findfirst(==(+), Arborist._get_binary_ops(ops))
        idx_times = findfirst(==(*), Arborist._get_binary_ops(ops))

        tree = Node{Float32}(;
            op=UInt8(idx_plus),
            l=Node{Float32}(;
                op=UInt8(idx_times),
                l=Node{Float32}(; val=Float32(5.0)),
                r=Node{Float32}(; feature=UInt16(1)),
            ),
            r=Node{Float32}(; val=Float32(-3.0)),
        )

        xs = collect(Float32, -1.0:0.1:1.0)
        X  = reshape(xs, 1, :)
        y  = Float32.(2.0 .* xs .+ 1.0)
        e = TreeFitnessEvaluator(X, y, ops)

        g = TreeGenome{Float32}(tree, ops, 1)
        pre = Arborist.evaluate(e, g)
        post = optimize_constants!(g, e; max_iter=100, tol=1e-12, fd_step=1e-3)
        @test post < pre
        @test post < 1e-4

        # Verify recovered constants are close to the true values.
        constants, _ = DynamicExpressions.get_scalar_constants(g.tree)
        @test length(constants) == 2
        # The first constant we placed was 5.0 (coeff of x), should become 2.0.
        # The second constant was -3.0 (intercept), should become 1.0. Order
        # in the returned vector follows depth-first traversal — the first
        # constant encountered is 5.0 (left subtree), then -3.0 (right).
        @test isapprox(constants[1], 2.0f0; atol=1e-2)
        @test isapprox(constants[2], 1.0f0; atol=1e-2)
    end

    @testset "optimize_constants! is no-op on a tree with no constants" begin
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        # Tree is just a feature node — no constants.
        tree = Node{Float32}(; feature=UInt16(1))
        X = reshape(Float32[1.0, 2.0, 3.0], 1, :)
        y = Float32[1.0, 2.0, 3.0]
        e = TreeFitnessEvaluator(X, y, ops)
        g = TreeGenome{Float32}(tree, ops, 1)
        pre = Arborist.evaluate(e, g)
        post = optimize_constants!(g, e)
        @test post == pre
        @test post ≈ 0.0  # x fits y perfectly
    end

    @testset "optimize_constants! rolls back on non-improvement" begin
        # Degenerate case: target constant is exactly where tree already sits.
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        tree = Node{Float32}(; val=Float32(1.0))  # constant output = 1.0
        X = reshape(Float32[0.0, 1.0, 2.0], 1, :)
        y = Float32[1.0, 1.0, 1.0]
        e = TreeFitnessEvaluator(X, y, ops)
        g = TreeGenome{Float32}(tree, ops, 1)
        pre = Arborist.evaluate(e, g)
        @test pre ≈ 0.0  # already exact
        post = optimize_constants!(g, e)
        # Either unchanged (gradient is zero, tol halts immediately) or
        # strictly non-greater.
        @test post <= pre + 1e-8
        # Constant should still be very close to 1.0 — rollback guarantees no drift.
        constants, _ = DynamicExpressions.get_scalar_constants(g.tree)
        @test isapprox(constants[1], 1.0f0; atol=1e-2)
    end

    @testset "GP with constant_optimization improves on control" begin
        # Small SR: y = 3x + 2 on 20 points. Run GP with and without const opt
        # using the same seed + identical config otherwise. Assert const-opt
        # run produces <= control run's best fitness.
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        xs = collect(Float32, -1.0:0.1:1.0)
        X = reshape(xs, 1, :)
        y = Float32.(3.0 .* xs .+ 2.0)

        function _run(const_opt)
            evaluator = TreeFitnessEvaluator(X, y, ops)
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
            alg = GeneticProgramming(
                pop_size=30, generations=20, parallel=false,
                mutation_rate=0.4, crossover_rate=0.4,
                constant_optimization=const_opt,
            )
            return solve(problem, alg).best_fitness
        end

        control = _run(nothing)
        with_opt = _run(ConstantOptimization(; frequency=5, top_k=3, max_iter=30))

        println("  SR y=3x+2: control=$(round(control, digits=6)), " *
                "with_opt=$(round(with_opt, digits=6))")
        flush(stdout)
        @test with_opt <= control
    end
end
