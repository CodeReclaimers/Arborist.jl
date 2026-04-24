using DynamicExpressions: Node, OperatorEnum

@testset "Graphviz DOT export" begin
    # Treat the output as a plain String and check structural invariants:
    # - starts with `digraph`, ends with `}`
    # - every `node_N` reference has a matching definition
    # - every label we care about appears at least once

    function _dot_is_well_formed(s::AbstractString)
        stripped = strip(s)
        return startswith(stripped, "digraph") && endswith(stripped, "}")
    end

    @testset "TreeGenome" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin, cos])
        # sin(x1 + x2)
        inner = Node(1, Node(Float64; feature=1), Node(Float64; feature=2))
        tree = Node(1, inner)    # unary slot 1 = sin
        g = TreeGenome(tree, ops, 2)

        dot = to_dot(g)
        @test _dot_is_well_formed(dot)
        @test occursin("digraph TreeGenome", dot)
        @test occursin("sin", dot)
        @test occursin("+", dot)
        @test occursin("x1", dot)
        @test occursin("x2", dot)

        # Constant leaves render literally.
        c_tree = TreeGenome(Node(Float64; val=3.14), ops, 0)
        @test occursin("3.14", to_dot(c_tree))
    end

    @testset "ExprGenome" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        s = GenState(fset, Dict(:x => Float32), Dict(:y => Float32), 0)
        g = ExprGenome(Expr[:(y = x + 1.0f0), :(y = y * 2.0f0)], s)
        dot = to_dot(g)
        @test _dot_is_well_formed(dot)
        @test occursin("digraph ExprGenome", dot)
        @test occursin("cluster_stmt_1", dot)
        @test occursin("cluster_stmt_2", dot)
    end

    @testset "AntGenome" begin
        ag = Arborist.AntGenome(
            :(begin; move; left; end),
            [:move, :right, :left],
            [:food_ahead],
            3,
        )
        dot = to_dot(ag)
        @test _dot_is_well_formed(dot)
        @test occursin("digraph AntGenome", dot)
        @test occursin("move", dot)
    end

    @testset "ADFGenome" begin
        aug_ops = OperatorEnum(
            binary_operators=[+, -, *, Arborist._adf_placeholder],
            unary_operators=[sin],
        )
        main = Node(1, Node(Float64; feature=1), Node(Float64; feature=2))
        adf = Node(Float64; feature=3)  # ARG0
        g = Arborist.ADFGenome{Float64}(main, [adf], 2, aug_ops, 2, 1)
        dot = to_dot(g)
        @test _dot_is_well_formed(dot)
        @test occursin("digraph ADFGenome", dot)
        @test occursin("cluster_main", dot)
        @test occursin("cluster_adf_0", dot)
    end

    @testset "GraphGenome" begin
        reset_innovation_counter!()
        rng = Random.MersenneTwister(42)
        g = initialize(GraphGenome, 2, 1, rng)
        dot = to_dot(g)
        @test _dot_is_well_formed(dot)
        @test occursin("digraph GraphGenome", dot)
        @test occursin("rankdir=LR", dot)
        @test occursin("rank=min", dot)
        @test occursin("rank=max", dot)
        # Input / output / bias should each have distinct styles.
        @test occursin("lightblue", dot)    # input fill
        @test occursin("lightcoral", dot)   # output fill
        @test occursin("lightgray", dot)    # bias fill

        # Disabled connections rendered dashed gray.
        # Manually disable one and re-render.
        any_inn = first(keys(g.connections))
        g.connections[any_inn].enabled = false
        dot2 = to_dot(g)
        @test occursin("style=dashed", dot2)
        @test occursin("color=gray", dot2)
    end

    @testset "IO-target form" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        g = TreeGenome(Node(Float64; feature=1), ops, 1)
        io = IOBuffer()
        to_dot(io, g)
        s = String(take!(io))
        @test _dot_is_well_formed(s)
    end

    # If the system has `dot` installed, actually invoke it to make sure the
    # output parses as valid DOT. This is a robustness check against syntax
    # drift beyond simple regex matches.
    @testset "dot binary parses output (if available)" begin
        dot_available = try
            success(run(pipeline(`dot -V`, stderr=devnull, stdout=devnull)))
        catch
            false
        end
        if !dot_available
            @test_skip "Graphviz `dot` binary not on PATH — skipping round-trip parse check"
        else
            # Helper: write the DOT source to a temp file and run
            # `dot -Tsvg tmp -o /dev/null`; return whether `dot` accepted it.
            function _dot_parses(src::AbstractString)
                tmp_path, tmp_io = mktemp()
                write(tmp_io, src); close(tmp_io)
                ok = try
                    success(run(pipeline(`dot -Tsvg $tmp_path -o /dev/null`,
                                         stderr=devnull, stdout=devnull)))
                catch
                    false
                end
                rm(tmp_path; force=true)
                return ok
            end

            reset_innovation_counter!()
            rng = Random.MersenneTwister(42)
            gg = initialize(GraphGenome, 2, 1, rng)
            @test _dot_parses(to_dot(gg))

            ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
            tg = TreeGenome(Node(1, Node(Float64; feature=1),
                                    Node(Float64; val=2.0)), ops, 1)
            @test _dot_parses(to_dot(tg))
        end
    end
end
