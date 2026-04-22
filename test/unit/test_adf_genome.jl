using Test
using Arborist
using Random
using DynamicExpressions

@testset "ADFGenome" begin
    @testset "initialization" begin
        rng = MersenneTwister(42)
        base_ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        g = initialize_adf(Float32, base_ops, 2, 2; arity=2, max_depth=3, rng=rng)
        @test g isa ADFGenome{Float32}
        @test g.n_features == 2
        @test g.n_adfs == 2
        @test g.arity == 2
        @test length(g.adfs) == 2
        # Augmented operator enum should have base + n_adfs binary slots.
        @test length(Arborist._get_binary_ops(g.operators)) == 5  # 3 base + 2 ADFs
    end

    @testset "non-binary arity rejected" begin
        rng = MersenneTwister(1)
        base_ops = OperatorEnum(binary_operators=[+], unary_operators=[])
        @test_throws ArgumentError initialize_adf(Float32, base_ops, 2, 1;
                                                   arity=3, rng=rng)
    end

    @testset "expand_adfs round-trip on hand-built genome" begin
        # Hand-build: ADF0(a, b) := a + b. Main := ADF0(x1, x2).
        # Expanded main should be: x1 + x2.
        base_ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        n_features = 2
        n_adfs = 1
        aug_ops = augmented_operators(base_ops, n_adfs)
        # ARG0 = feature 3, ARG1 = feature 4.
        adf_body = Node{Float32}(; op=UInt8(1),  # '+' is op 1 in base_ops
                                   l=Node{Float32}(; feature=UInt16(3)),
                                   r=Node{Float32}(; feature=UInt16(4)))
        # ADF call uses op slot n_base_binary+1 = 4 in the augmented enum.
        main = Node{Float32}(; op=UInt8(4),
                              l=Node{Float32}(; feature=UInt16(1)),
                              r=Node{Float32}(; feature=UInt16(2)))
        g = ADFGenome{Float32}(main, [adf_body], 2, aug_ops, n_features, n_adfs)

        expanded = expand_adfs(g)
        # Verify expanded shape: it should be op=+ with two feature children.
        @test expanded.degree == 2
        @test expanded.op == UInt8(1)  # '+'
        @test expanded.l.feature == UInt16(1)
        @test expanded.r.feature == UInt16(2)
    end

    @testset "evaluate_adf computes correct MSE" begin
        # Same hand-built genome: y_pred = ADF0(x1, x2) = x1 + x2.
        # Target: y = x1 + x2 → MSE should be 0.
        base_ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        n_features = 2
        aug_ops = augmented_operators(base_ops, 1)
        adf_body = Node{Float32}(; op=UInt8(1),
                                   l=Node{Float32}(; feature=UInt16(3)),
                                   r=Node{Float32}(; feature=UInt16(4)))
        main = Node{Float32}(; op=UInt8(4),
                              l=Node{Float32}(; feature=UInt16(1)),
                              r=Node{Float32}(; feature=UInt16(2)))
        g = ADFGenome{Float32}(main, [adf_body], 2, aug_ops, n_features, 1)

        X = Float32[1 2 3 4; 5 6 7 8]
        y = Float32[6, 8, 10, 12]  # x1 + x2
        @test evaluate_adf(g, X, y) ≈ 0.0 atol=1e-6

        # Off-by-one target should give MSE = 1.0.
        y_off = y .+ 1.0f0
        @test evaluate_adf(g, X, y_off) ≈ 1.0 atol=1e-6
    end

    @testset "ADF body referencing only ARG0 expands correctly" begin
        # ADF0(a, b) := a (ignores b). Main := ADF0(x1, x2). Expanded := x1.
        base_ops = OperatorEnum(binary_operators=[+], unary_operators=[])
        aug_ops = augmented_operators(base_ops, 1)
        adf_body = Node{Float32}(; feature=UInt16(3))  # ARG0
        main = Node{Float32}(; op=UInt8(2),  # ADF0 (slot 2 = base 1 + ADF 1)
                              l=Node{Float32}(; feature=UInt16(1)),
                              r=Node{Float32}(; feature=UInt16(2)))
        g = ADFGenome{Float32}(main, [adf_body], 2, aug_ops, 2, 1)
        expanded = expand_adfs(g)
        @test expanded.degree == 0
        @test expanded.feature == UInt16(1)

        X = Float32[10 20; 0 0]
        @test evaluate_adf(g, X, Float32[10, 20]) ≈ 0.0 atol=1e-6
    end

    @testset "complexity sums across trees" begin
        rng = MersenneTwister(3)
        base_ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g = initialize_adf(Float32, base_ops, 2, 2; arity=2, max_depth=3, rng=rng)
        c = complexity(g)
        @test c == count_nodes(g.main) + sum(count_nodes(adf) for adf in g.adfs)
    end

    @testset "mutation produces a structurally valid genome" begin
        rng = MersenneTwister(5)
        base_ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
        g = initialize_adf(Float32, base_ops, 2, 2; arity=2, max_depth=3, rng=rng)
        for _ in 1:30
            g2 = mutate(SubtreeMutation(), g, rng)
            @test g2 isa ADFGenome{Float32}
            @test g2.n_features == g.n_features
            @test g2.n_adfs == g.n_adfs
            # Should still be expandable + evaluable on a small dataset.
            X = Float32[1 2; 3 4]
            y = Float32[0, 0]
            f = evaluate_adf(g2, X, y)
            @test isfinite(f) || f == Inf  # Inf is acceptable; NaN is not.
        end
    end

    @testset "crossover preserves structure" begin
        rng = MersenneTwister(7)
        base_ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g1 = initialize_adf(Float32, base_ops, 2, 2; arity=2, max_depth=3, rng=rng)
        g2 = initialize_adf(Float32, base_ops, 2, 2; arity=2, max_depth=3, rng=rng)
        for _ in 1:20
            (c1, c2) = crossover(SubtreeCrossover(), g1, g2, rng)
            @test c1.n_adfs == g1.n_adfs
            @test c2.n_adfs == g2.n_adfs
            @test c1.n_features == g1.n_features
            @test c2.n_features == g2.n_features
        end
    end

    @testset "crossover mismatch errors" begin
        rng = MersenneTwister(11)
        base_ops = OperatorEnum(binary_operators=[+], unary_operators=[])
        g1 = initialize_adf(Float32, base_ops, 2, 2; arity=2, rng=rng)
        g2 = initialize_adf(Float32, base_ops, 2, 1; arity=2, rng=rng)  # different n_adfs
        @test_throws ArgumentError crossover(SubtreeCrossover(), g1, g2, rng)
    end

    @testset "serialize is human-readable" begin
        rng = MersenneTwister(13)
        base_ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        g = initialize_adf(Float32, base_ops, 2, 2; arity=2, max_depth=2, rng=rng)
        s = serialize(g)
        @test occursin("ADFGenome", s)
        @test occursin("MAIN:", s)
        @test occursin("ADF0:", s)
        @test occursin("ADF1:", s)
    end
end
