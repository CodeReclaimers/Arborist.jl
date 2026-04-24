@testset "protected_operators" begin
    @testset "pdiv guards division by zero" begin
        @test pdiv(1.0, 2.0) ≈ 0.5
        @test pdiv(3.0, 0.0) == 1.0              # fallback to one(a)
        @test pdiv(3.0, 1e-20) == 1.0            # below threshold -> fallback
        @test pdiv(3.0, 1e-5)  ≈ 3.0 / 1e-5      # above threshold -> real div
        @test pdiv(-2.0, -4.0) ≈ 0.5
        # Float32 preserves type through the fallback.
        @test pdiv(1.0f0, 0.0f0) isa Float32
        @test pdiv(1.0f0, 0.0f0) == 1.0f0
    end

    @testset "plog is finite everywhere" begin
        for x in (-100.0, -1.0, -1e-20, 0.0, 1e-20, 1.0, 100.0)
            @test isfinite(plog(x))
        end
        @test plog(exp(1.0)) ≈ log(exp(1.0) + PROTECTED_EPS)
        @test plog(0.0) ≈ log(PROTECTED_EPS)
        @test plog(-5.0) ≈ log(5.0 + PROTECTED_EPS)   # mirror symmetry
    end

    @testset "psqrt is finite everywhere" begin
        for x in (-100.0, -1.0, 0.0, 1.0, 100.0)
            @test isfinite(psqrt(x))
            @test psqrt(x) >= 0.0
        end
        @test psqrt(4.0) ≈ 2.0
        @test psqrt(-9.0) ≈ 3.0                   # mirror symmetry
    end

    @testset "pexp does not overflow" begin
        @test isfinite(pexp(1000.0))              # clamped
        @test isfinite(pexp(-1000.0))             # clamped lower too
        @test pexp(0.0) ≈ 1.0
        @test pexp(1.0) ≈ exp(1.0)
        @test pexp(100.0) ≈ exp(50.0)             # clamp ceiling
        @test pexp(-100.0) ≈ exp(-50.0)           # clamp floor
    end

    @testset "pinv guards division by zero" begin
        @test pinv(2.0)   ≈ 0.5
        @test pinv(0.0)   == 0.0
        @test pinv(1e-20) == 0.0
        @test pinv(-4.0)  ≈ -0.25
        @test pinv(1.0f0) isa Float32
    end

    @testset "type stability" begin
        # The branches in pdiv/pinv produce values that should preserve
        # the input element type; any implicit Float64 promotion would be
        # a regression.
        @test @inferred(pdiv(1.0f0, 2.0f0)) isa Float32
        @test @inferred(pdiv(1.0f0, 0.0f0)) isa Float32
        @test @inferred(pinv(2.0f0)) isa Float32
        @test @inferred(pinv(0.0f0)) isa Float32
        @test @inferred(psqrt(4.0f0)) isa Float32
        # plog and pexp mix in Float64 literals (PROTECTED_EPS, clamp bounds),
        # so they widen to Float64 — documented behavior, not tested as
        # type-stable below the literal boundary.
    end

    @testset "default_protected_function_set contents" begin
        fset = default_protected_function_set()
        @test fset isa Arborist.FunctionSet

        # Flatten to (name, arity) for inclusion testing.
        names = Set((d.name, length(d.args)) for d in fset.funcs)

        # Binary arithmetic for Float32 and Int32.
        for op in (:+, :-, :*)
            @test (op, 2) in names
        end
        # pdiv Float32 only.
        @test (:pdiv, 2) in names
        # Protected unaries.
        for op in (:plog, :psqrt, :pexp, :sin, :cos)
            @test (op, 1) in names
        end
        # Not included by default.
        @test !((:pinv, 1) in names)
    end

    @testset "protected set is usable with TreeGenome OperatorEnum" begin
        # A user passing the raw functions to DynamicExpressions should
        # get a working OperatorEnum. This is the TreeGenome-side contract.
        import DynamicExpressions
        ops = DynamicExpressions.OperatorEnum(
            binary_operators = [+, -, *, pdiv],
            unary_operators  = [plog, psqrt, pexp, sin, cos],
        )
        @test ops isa DynamicExpressions.OperatorEnum
    end
end
