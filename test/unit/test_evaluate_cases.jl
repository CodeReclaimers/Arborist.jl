using Test
using Arborist
using Random

# Dummy evaluator without an evaluate_cases method — used to verify that
# lexicase-style callers get a clean MethodError.
struct _NoCasesEvaluator <: AbstractEvaluator end
Arborist.evaluate(::_NoCasesEvaluator, f::Function) = 0.0
Arborist.input_signature(::_NoCasesEvaluator) = Dict{Symbol, DataType}()
Arborist.output_signature(::_NoCasesEvaluator) = Dict{Symbol, DataType}()

@testset "evaluate_cases" begin
    @testset "TableFitnessEvaluator function form" begin
        # Target: y = 2x; dataset covers x ∈ {1, 2, 3}.
        input_cols = Dict(:x => Int)
        output_cols = Dict(:y => Int)
        input_rows = [Dict{Symbol,Any}(:x => i) for i in 1:3]
        output_rows = [Dict{Symbol,Any}(:y => 2*i) for i in 1:3]
        # 1s per-call cap; plenty of headroom for JIT on the first invocation.
        e = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                  time_limit_ns=1_000_000_000)

        # Warm up each test function so JIT compilation doesn't blow the per-call limit.
        let f = x -> 2*x; f(1); end
        let f = x -> 2*x + 1; f(1); end

        # Perfect function: all zero.
        cases_perfect = evaluate_cases(e, x -> 2*x)
        @test cases_perfect == [0.0, 0.0, 0.0]

        # Offset-by-1 function: squared error is 1 for every row.
        cases_off = evaluate_cases(e, x -> 2*x + 1)
        @test cases_off == [1.0, 1.0, 1.0]

        # mean(evaluate_cases) matches evaluate within tolerance.
        @test isapprox(sum(cases_off)/length(cases_off), Arborist.evaluate(e, x -> 2*x + 1); atol=1e-12)

        # Function that throws -> Inf per case.
        cases_throw = evaluate_cases(e, x -> error("boom"))
        @test all(isinf, cases_throw)
    end

    @testset "TreeFitnessEvaluator" begin
        using DynamicExpressions
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        # Dataset: y = x^2 on 5 points.
        X = reshape(Float32[1, 2, 3, 4, 5], 1, :)
        y = X[1, :] .* X[1, :]
        e = TreeFitnessEvaluator(X, y, ops)

        # Genome representing x*x (perfect fit).
        tree = Node{Float32}(; op=UInt8(3),  # '*' is 3rd binary
                               l=Node{Float32}(; feature=UInt16(1)),
                               r=Node{Float32}(; feature=UInt16(1)))
        g = TreeGenome{Float32}(tree, ops, 1)
        cases = evaluate_cases(g, e)
        @test length(cases) == 5
        @test all(isapprox.(cases, 0.0; atol=1e-5))

        # mean(evaluate_cases) ≈ evaluate.
        @test isapprox(sum(cases)/length(cases), Arborist.evaluate(e, g); atol=1e-6)

        # Off-by-1 genome: y = x (first-power). Per-sample se = (x - x^2)^2.
        tree2 = Node{Float32}(; feature=UInt16(1))
        g2 = TreeGenome{Float32}(tree2, ops, 1)
        cases2 = evaluate_cases(g2, e)
        for i in 1:5
            expected = (X[1, i] - X[1, i]^2)^2
            @test isapprox(cases2[i], Float64(expected); atol=1e-4)
        end
    end

    @testset "GraphEvaluator feedforward" begin
        # Tiny network: 1 input, 1 output. Force all existing connections
        # to weight 1 (identity pass-through) and the output node to :identity.
        rng = MersenneTwister(42)
        g = Arborist.initialize(GraphGenome, 1, 1, rng)
        for conn in values(g.connections)
            conn.weight = 1.0
            conn.enabled = true
        end
        for (id, node) in g.nodes
            if node.type == :output
                g.nodes[id] = Arborist.NodeGene(node.id, node.type, :identity)
            end
        end

        # 3-sample dataset; mean(cases) must match evaluate_genome exactly.
        input_data = reshape([0.0, 0.5, 1.0], 1, :)
        output_data = reshape([0.0, 0.5, 1.0], 1, :)
        e = GraphEvaluator(input_data, output_data)
        cases = evaluate_cases(g, e)
        @test length(cases) == 3
        mse_eval = Arborist.evaluate_genome(g, e)
        @test isapprox(sum(cases)/length(cases), mse_eval; atol=1e-10)
    end

    @testset "GraphEvaluator recurrent raises" begin
        rng = MersenneTwister(42)
        g = Arborist.initialize(GraphGenome, 1, 1, rng)
        input_data = reshape([0.0, 0.5], 1, :)
        output_data = reshape([0.0, 0.5], 1, :)
        e = GraphEvaluator(input_data, output_data; allow_recurrent=true)
        @test_throws ArgumentError evaluate_cases(g, e)
    end

    @testset "MethodError for unsupported evaluator" begin
        e = _NoCasesEvaluator()
        @test_throws MethodError evaluate_cases(e, identity)
    end
end
