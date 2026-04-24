using Test
using Arborist
using Random

@testset "Arborist.Benchmarks module" begin
    B = Arborist.Benchmarks

    @testset "nguyen: round-trip shape" begin
        for n in 1:10
            prob = B.nguyen(n)
            @test prob.X isa Matrix{Float32}
            @test prob.y isa Vector{Float32}
            @test size(prob.X, 2) == length(prob.y)
            @test prob.name == "Nguyen-$n"
            @test !isempty(prob.target_expr)
        end
        @test size(B.nguyen(1).X) == (1, 20)
        @test size(B.nguyen(9).X) == (2, 100)
        @test_throws ArgumentError B.nguyen(0)
        @test_throws ArgumentError B.nguyen(11)
    end

    @testset "keijzer: variants produce train + test" begin
        k4 = B.keijzer(:k4)
        @test size(k4.X, 1) == 3
        @test size(k4.X, 2) == length(k4.y)
        @test size(k4.X_test_interior, 2) == length(k4.y_test_interior)
        @test size(k4.X_test_extrapolation, 2) == length(k4.y_test_extrapolation)
        # Interior and extrapolation disjoint by construction.
        @test all(maximum(abs.(col)) <= 1 for col in eachcol(k4.X_test_interior))
        @test all(maximum(abs.(col)) > 1 for col in eachcol(k4.X_test_extrapolation))

        k11 = B.keijzer(:k11)
        @test size(k11.X, 1) == 2
        @test size(k11.X, 2) == length(k11.y)
        @test size(k11.X_test_interior, 2) == 400
        @test_throws ArgumentError B.keijzer(:k99)
    end

    @testset "koza: variants" begin
        for name in [:quartic, :septic, :nonic]
            p = B.koza(name)
            @test size(p.X) == (1, 20)
            @test length(p.y) == 20
        end
        @test_throws ArgumentError B.koza(:invalid)
    end

    @testset "pagie-1 shape" begin
        p = B.pagie()
        @test size(p.X) == (2, 676)
        @test length(p.y) == 676
        # No NaN/Inf in the generated y (protected inverse).
        @test all(isfinite, p.y)
    end

    @testset "iris: stratified split preserves class counts" begin
        ds = B.iris(; test_ratio=0.2)
        @test size(ds.X_train) == (4, 120)
        @test size(ds.X_test) == (4, 30)
        @test length(ds.y_train) == 120
        @test length(ds.y_test) == 30
        # Each class should appear 40 times in train, 10 in test.
        for cls in 1:3
            @test count(==(cls), ds.y_train) == 40
            @test count(==(cls), ds.y_test) == 10
        end
        @test ds.class_names == ["setosa", "versicolor", "virginica"]
        @test ds.n_classes == 3

        # Different test_ratio.
        ds2 = B.iris(; test_ratio=0.4)
        @test size(ds2.X_train) == (4, 90)
        @test size(ds2.X_test) == (4, 60)

        @test_throws ArgumentError B.iris(; test_ratio=0.0)
        @test_throws ArgumentError B.iris(; test_ratio=1.0)
    end

    @testset "two_spirals" begin
        s = B.two_spirals()
        @test size(s.X) == (2, 194)
        @test size(s.y) == (1, 194)
        # ±1 only, balanced classes.
        @test count(==(1.0), s.y) == 97
        @test count(==(-1.0), s.y) == 97
    end

    @testset "multiplexer: 6-bit and 11-bit" begin
        m6 = B.multiplexer(2)
        @test m6.n_inputs == 6
        @test m6.n_rows == 64
        @test size(m6.X) == (6, 64)
        @test length(m6.y) == 64
        # Boolean outputs only.
        @test all(v -> v == 0.0 || v == 1.0, m6.y)

        m11 = B.multiplexer(3)
        @test m11.n_inputs == 11
        @test m11.n_rows == 2048

        @test_throws ArgumentError B.multiplexer(0)
    end

    @testset "parity: 3-bit and 5-bit" begin
        p3 = B.parity(3)
        @test p3.n_inputs == 3
        @test p3.n_rows == 8
        # For parity, y = XOR of inputs.
        for row in 1:8
            expected = Float64(Int(sum(p3.X[:, row])) % 2)
            @test p3.y[row] == expected
        end
        p5 = B.parity(5)
        @test p5.n_rows == 32
        @test_throws ArgumentError B.parity(0)
    end

    @testset "xor_env" begin
        x = B.xor_env()
        @test size(x.input_data) == (2, 4)
        @test size(x.output_data) == (1, 4)
    end

    @testset "cartpole / mountain_car / acrobot / double_pole" begin
        for env in [B.cartpole(), B.mountain_car(), B.acrobot(),
                    B.double_pole(markovian=true),
                    B.double_pole(markovian=false)]
            @test hasproperty(env, :initial_state)
            @test hasproperty(env, :dynamics)
            @test hasproperty(env, :reward)
            @test hasproperty(env, :done)
            @test hasproperty(env, :observe)
            @test hasproperty(env, :decode_action)
            @test env.n_states > 0
            @test env.n_actions > 0
            @test env.max_steps > 0

            # Smoke-test the callables on an initial state.
            rng = MersenneTwister(0)
            s = env.initial_state(rng)
            obs = env.observe(s)
            @test length(obs) == env.n_states
        end
    end

    @testset "sequence_memory / sequence_recall" begin
        sm = B.sequence_memory(length=5)
        @test sm.length == 5
        @test !isempty(sm.sequences)
        @test length(sm.sequences[1]) == 5
        @test length(sm.targets) == length(sm.sequences)

        sr = B.sequence_recall(delay=3)
        @test sr.delay == 3
        @test all(length(s) == 4 for s in sr.sequences)

        @test_throws ArgumentError B.sequence_memory(length=1)
        @test_throws ArgumentError B.sequence_recall(delay=0)
    end

    @testset "canonical_sr_operators helper" begin
        ops_spec = B.canonical_sr_operators(Float32)
        @test length(ops_spec.binary) == 4
        @test length(ops_spec.unary) == 5
    end
end
