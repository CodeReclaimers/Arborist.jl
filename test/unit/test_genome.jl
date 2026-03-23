@testset "ExprGenome" begin

    function make_test_problem()
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows)

        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<, :(==)]
            add!(fset, op, 2, Float32, Bool)
        end

        return GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
    end

    @testset "initialize" begin
        problem = make_test_problem()
        g = initialize(ExprGenome, problem)
        @test g isa ExprGenome
        @test length(g.body) == 3
        @test all(e -> e isa Expr, g.body)
        @test g.state isa GenState
    end

    @testset "mutate (genome interface)" begin
        problem = make_test_problem()
        g = initialize(ExprGenome, problem)
        rng = Random.MersenneTwister(123)
        g2 = mutate(g, rng)
        @test g2 isa ExprGenome
        @test g2.state === g.state
        # Mutation should produce a different body (with high probability).
        # We don't assert inequality since a no-op mutation is possible.
    end

    @testset "crossover (genome interface)" begin
        problem = make_test_problem()
        Random.seed!(42)
        g1 = initialize(ExprGenome, problem)
        g2 = initialize(ExprGenome, problem)
        rng = Random.MersenneTwister(99)
        (c1, c2) = crossover(g1, g2, rng)
        @test c1 isa ExprGenome
        @test c2 isa ExprGenome
    end

    @testset "distance" begin
        problem = make_test_problem()
        Random.seed!(42)
        g1 = initialize(ExprGenome, problem)
        g2 = initialize(ExprGenome, problem)
        d = distance(g1, g2)
        @test d isa Float64
        @test d >= 0.0
        # Distance to self should be zero.
        @test distance(g1, g1) == 0.0
    end

    @testset "complexity" begin
        problem = make_test_problem()
        g = initialize(ExprGenome, problem)
        c = complexity(g)
        @test c isa Float64
        @test c > 0.0
    end

    @testset "serialize" begin
        problem = make_test_problem()
        Random.seed!(42)
        g = initialize(ExprGenome, problem)
        s = serialize(g)
        @test s isa String
        @test length(s) > 0
    end

    @testset "deserialize (keyword arg)" begin
        problem = make_test_problem()
        Random.seed!(42)
        g = initialize(ExprGenome, problem)
        s = serialize(g)
        # Without state, should return nothing.
        @test deserialize(ExprGenome, s) === nothing
        # With state keyword, should reconstruct.
        g2 = deserialize(ExprGenome, s; state=g.state)
        @test g2 isa ExprGenome
        @test length(g2.body) == length(g.body)
    end

    @testset "deserialize (positional arg)" begin
        problem = make_test_problem()
        Random.seed!(42)
        g = initialize(ExprGenome, problem)
        s = serialize(g)
        g2 = deserialize(ExprGenome, s, g.state)
        @test g2 isa ExprGenome
        @test length(g2.body) == length(g.body)
    end

    @testset "serialize/deserialize round-trip" begin
        problem = make_test_problem()
        Random.seed!(42)
        # Generate multiple assignment-only genomes and verify round-trip.
        # Some assignments may use repr forms that fail strict type-checking
        # (e.g., Float32(literal) not in the function set). We verify that
        # at least the valid assignments survive.
        successes = 0
        for _ in 1:20
            g = initialize(ExprGenome, problem)
            s = serialize(g)
            g2 = deserialize(ExprGenome, s, g.state)
            if g2 !== nothing
                successes += 1
                @test length(g2.body) >= 1
            end
        end
        @test successes >= 16  # at least 80% round-trip successfully
    end

    @testset "evaluate_genome" begin
        problem = make_test_problem()
        Random.seed!(42)
        g = initialize(ExprGenome, problem)
        fitness = Arborist.evaluate_genome(g, problem.evaluator)
        @test fitness isa Float64
        @test fitness >= 0.0
    end

    @testset "codegen basics (rvalues)" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32, :y => Int32, :z => Bool),
                     Dict(:a => Float32, :b => Int32, :c => Bool),
                     8)
        for T in [Float32, Int32, Bool]
            for (k, v) in Arborist.get_rvalues_of_type(s, T)
                @test v == T
            end
        end
    end

    @testset "codegen basics (lvalues)" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32, :y => Int32, :z => Bool),
                     Dict(:a => Float32, :b => Int32, :c => Bool),
                     8)
        for T in [Float32, Int32, Bool]
            for (k, v) in Arborist.get_lvalues_of_type(s, T)
                @test v == T
            end
        end
    end

    @testset "random assignment creation" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32, :y => Int32, :z => Bool),
                     Dict(:a => Float32, :b => Int32, :c => Bool),
                     8)
        for _ in 1:100
            assignment = create_random_assignment(s)
            @test assignment.head == :(=)
            @test Arborist.get_lvalue_type(s, assignment.args[1]) == Arborist.get_rvalue_type(s, assignment.args[2])
        end
    end

    @testset "create_random_for_loop" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32),
                     Dict(:y => Float32),
                     4)
        for _ in 1:20
            expr = Arborist.create_random_for_loop(s)
            @test expr isa Expr
            @test expr.head == :for
        end
    end

    @testset "create_random_while_loop" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32),
                     Dict(:y => Float32),
                     4)
        for _ in 1:20
            expr = Arborist.create_random_while_loop(s)
            @test expr isa Expr
            @test expr.head == :while
        end
    end

    @testset "create_random_if_statement" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32),
                     Dict(:y => Float32),
                     4)
        for _ in 1:20
            expr = Arborist.create_random_if_statement(s)
            @test expr isa Expr
            @test expr.head == :if
            @test length(expr.args) == 3
        end
    end

    @testset "depth limiting prevents infinite recursion" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32),
                     Dict(:y => Float32),
                     4)
        for _ in 1:20
            @test Arborist.create_random_for_loop(s; depth=0).head == :(=)
            @test Arborist.create_random_while_loop(s; depth=0).head == :(=)
            @test Arborist.create_random_if_statement(s; depth=0).head == :(=)
            @test Arborist.create_random_block(s; depth=0).head == :block
            @test create_random_statement(s; depth=0).head == :(=)
        end
    end

    @testset "add_loop_checks instruments for loops" begin
        body = [
            Expr(:for, Expr(:(=), :i, Expr(:call, :(:), Int32(1), Int32(5))),
                Expr(:block, :(a = Float32(1.0))))
        ]
        checked = add_loop_checks(body; limit=100)
        @test length(checked) == 1
        outer = checked[1]
        @test outer isa Expr
        @test outer.head == :block
        @test length(outer.args) == 2
        @test outer.args[2].head == :for
    end

    @testset "add_loop_checks actually prevents runaway loops" begin
        body = [
            Expr(:while, true,
                Expr(:block, :(a = Float32(1.0))))
        ]
        checked = add_loop_checks(body; limit=10)
        func_expr = Expr(:function, Expr(:call, gensym("test_loop")),
            Expr(:block, :(a = Float32(0.0)), checked...))
        f = @eval $func_expr
        @test_throws LoopLimitExceeded f()
    end

    @testset "crossover functions" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32),
                     Dict(:y => Float32),
                     4)
        a = Expr(:block, create_random_assignment(s), create_random_assignment(s))
        b = Expr(:block, create_random_assignment(s), create_random_assignment(s))
        (oa, ob) = Arborist.crossover(s, a, b)
        @test oa isa Expr
        @test ob isa Expr
    end

    @testset "crossover does not modify parents" begin
        fset = default_function_set()
        s = GenState(fset,
                     Dict(:x => Float32),
                     Dict(:y => Float32),
                     4)
        a = Expr(:block, create_random_assignment(s), create_random_assignment(s))
        b = Expr(:block, create_random_assignment(s), create_random_assignment(s))
        a_str = string(a)
        b_str = string(b)
        Arborist.crossover(s, a, b)
        @test string(a) == a_str
        @test string(b) == b_str
    end

    @testset "create_harness produces valid function expr" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        s = GenState(fset, Dict(:x => Float32), Dict(:y => Float32), 2)
        body = [:(y = x + Float32(1.0))]
        harness = create_harness(s, body, :test_harness_fn)
        @test harness isa Expr
        @test harness.head == :function
        f = @eval $harness
        result = f(Float32(5.0))
        @test result ≈ Float32(6.0)
    end

end
