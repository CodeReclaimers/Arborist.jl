@testset "ExprGenome Float32 round-trip" begin

    # GenState's constructor randomizes temp variable types. To keep this
    # round-trip test deterministic regardless of the temp draws, we build
    # bodies that only assign to the output variable `y` (guaranteed
    # Float32 by construction).
    function make_float32_state()
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        return GenState(fset,
                        Dict(:x => Float32),
                        Dict(:y => Float32),
                        2)
    end

    @testset "Float32(literal) call form survives round-trip" begin
        state = make_float32_state()

        # Body with Float32(literal) call expressions -- the form an LLM
        # commonly emits as Julia source. Meta.parse turns
        # "Float32(1.5)" into Expr(:call, :Float32, 1.5), which the
        # default repr() renders back to "Float32(1.5)" -- a form the
        # deserializer's _is_valid_assignment rejects (RHS is a :call,
        # not a bare Number).
        body = Expr[
            :(y = Float32(1.5)),
            :(y = Float32(0.25)),
            :(y = Float32(-3.0)),
        ]
        g = ExprGenome(body, state)

        s = serialize(g)
        @test s isa String
        @test length(s) > 0
        # serialize must emit f0 literal form, not Float32(...) call form.
        @test !occursin("Float32(", s)
        @test occursin("1.5f0", s)
        @test occursin("0.25f0", s)
        @test occursin("-3.0f0", s)

        g2 = deserialize(ExprGenome, s, state)
        @test g2 !== nothing
        @test g2 isa ExprGenome
        @test length(g2.body) == length(g.body)
    end

    @testset "nested Float32(literal) inside binary op" begin
        state = make_float32_state()

        # Float32 calls nested inside other call expressions also need
        # collapsing -- otherwise the outer assignment validates fine
        # but the printed source still says "Float32(1.5)".
        body = Expr[
            :(y = Float32(2.5) + Float32(0.5)),
            :(y = x * Float32(3.0)),
        ]
        g = ExprGenome(body, state)
        s = serialize(g)

        @test !occursin("Float32(", s)
        @test occursin("2.5f0", s)
        @test occursin("0.5f0", s)
        @test occursin("3.0f0", s)

        g2 = deserialize(ExprGenome, s, state)
        @test g2 !== nothing
        @test length(g2.body) == length(g.body)
    end

    @testset "non-literal Float32 conversions are preserved" begin
        # Float32(symbol) -- a real conversion call -- must NOT be
        # collapsed. Only literal arguments (Numbers) should fold.
        state = make_float32_state()
        body = Expr[
            :(y = Float32(x)),
        ]
        g = ExprGenome(body, state)
        s = serialize(g)
        @test occursin("Float32(x)", s)
    end

    @testset "Float32-typed leaves print as f0 literals" begin
        state = make_float32_state()
        body = Expr[
            :(y = $(Float32(1.5))),
        ]
        g = ExprGenome(body, state)
        s = serialize(g)
        @test occursin("1.5f0", s)
        @test !occursin("Float32(", s)

        g2 = deserialize(ExprGenome, s, state)
        @test g2 !== nothing
        @test length(g2.body) == length(g.body)
    end

end
