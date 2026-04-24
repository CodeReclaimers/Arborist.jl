using DynamicExpressions: Node, OperatorEnum

@testset "Base.show methods" begin
    # Helper: pretty-printed form (what the REPL shows).
    pretty(x) = sprint(show, MIME("text/plain"), x)
    # Compact form (what `repr(x)` / show-in-container uses).
    compact(x) = sprint(show, x)

    @testset "GPResult" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        g = TreeGenome(Node(Float64; feature=1), ops, 1)
        r = GPResult{TreeGenome{Float64}}(
            g, 0.0042, [g, g], [0.1, 0.01, 0.0042],
            [0.5, 0.3, 0.2], 3, 12.5, true,
        )
        @test occursin("GPResult{TreeGenome{Float64}}", compact(r))
        @test occursin("gens=3", compact(r))

        p = pretty(r)
        @test occursin("GPResult", p)
        @test occursin("generations run: 3", p)
        @test occursin("best fitness:    0.0042", p)
        @test occursin("converged:       true", p)
    end

    @testset "NSGAIIResult" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        g = TreeGenome(Node(Float64; feature=1), ops, 1)
        r = NSGAIIResult{TreeGenome{Float64}}(
            [g, g, g],                              # pareto_front
            [[0.1, 1.0], [0.2, 0.5], [0.3, 0.25]],  # pareto_fitnesses
            [g, g, g],                              # population
            [[0.1, 1.0], [0.2, 0.5], [0.3, 0.25]],  # all_fitnesses
            [0.1, 0.3, 0.5],                        # hypervolume_history
            10,                                     # generations_run
            5.0,                                    # wall_time
            ["fitness", "complexity"],              # objective_names
        )
        @test occursin("NSGAIIResult", compact(r))
        @test occursin("front=3", compact(r))

        p = pretty(r)
        @test occursin("front size:      3", p)
        @test occursin("objectives:      fitness, complexity", p)
        @test occursin("final HV:        0.5", p)
    end

    @testset "MAPElitesResult and MAPElitesArchive" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        g = TreeGenome(Node(Float64; feature=1), ops, 1)
        archive = MAPElitesArchive{TreeGenome{Float64}}([5, 5])
        archive.grid[(1, 1)] = Pair(g, 0.1)
        archive.grid[(2, 3)] = Pair(g, 0.2)

        @test occursin("MAPElitesArchive", compact(archive))
        @test occursin("2/25 cells", compact(archive))

        p = pretty(archive)
        @test occursin("filled:     2", p)
        @test occursin("8.0% coverage", p) || occursin("8% coverage", p)

        r = MAPElitesResult{TreeGenome{Float64}}(
            archive, [0.04, 0.08], [0.3, 0.5], g, 0.1, 3.2, 5,
        )
        @test occursin("MAPElitesResult", compact(r))
        pr = pretty(r)
        @test occursin("coverage:", pr)
        @test occursin("QD score:", pr)
    end

    @testset "RunLog and GenerationLog" begin
        g1 = Arborist.GenerationLog(0, 0.5, 0.8, 0.7, 1.2, 3, [5, 3, 2],
                                    Dict(:subtree => 8), Dict(:subtree => 10),
                                    4, 0.5)
        g2 = Arborist.GenerationLog(1, 0.2, 0.4, 0.35, 0.9, 3, [6, 3, 1],
                                    Dict(:subtree => 12), Dict(:subtree => 15),
                                    5, 0.6)
        @test occursin("gen=1", compact(g2))
        @test occursin("best=0.2", compact(g2))

        pg = pretty(g2)
        @test occursin("generation:       1", pg)
        @test occursin("species:", pg)
        @test occursin("operators (ok/try):", pg)

        rl = RunLog()
        push!(rl.entries, g1)
        push!(rl.entries, g2)
        @test occursin("RunLog(2 generations)", compact(rl))

        prl = pretty(rl)
        @test occursin("RunLog: 2 generations", prl)
        @test occursin("final fitness:", prl)
        @test occursin("operator success:", prl)

        # Empty log edge case.
        @test occursin("empty", pretty(RunLog()))
    end

    @testset "Checkpoint" begin
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        g = TreeGenome(Node(Float64; feature=1), ops, 1)
        c = Checkpoint{TreeGenome{Float64}}(
            1, v"0.1.0", v"1.11.5", 100, [g, g, g], [0.1, 0.2, 0.3],
            nothing, g, 0.1, [0.5, 0.3, 0.1], [0.7, 0.5, 0.2], 12.5, UInt64(42),
        )
        @test occursin("Checkpoint{TreeGenome{Float64}}", compact(c))
        @test occursin("gen=100", compact(c))
        @test occursin("pop=3", compact(c))

        p = pretty(c)
        @test occursin("population:       3 genomes", p)
        @test occursin("best fitness:     0.1", p)
    end

    @testset "genome show methods" begin
        # ExprGenome
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        s = GenState(fset, Dict(:x => Float32), Dict(:y => Float32), 0)
        eg = ExprGenome(Expr[:(y = x + 1.0f0)], s)
        @test occursin("ExprGenome(size=", compact(eg))
        peg = pretty(eg)
        @test occursin("ExprGenome", peg)
        @test occursin("y = x", peg)

        # TreeGenome
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        tg = TreeGenome(Node(1, Node(Float64; feature=1),
                                Node(Float64; val=2.0)), ops, 1)
        @test occursin("TreeGenome{Float64}", compact(tg))
        @test occursin("size=", compact(tg))
        ptg = pretty(tg)
        @test occursin("TreeGenome", ptg)

        # AntGenome
        Random.seed!(42)
        ag = Arborist.AntGenome(
            :(begin; move; left; end),
            [:move, :right, :left],
            [:food_ahead],
            3,
        )
        @test occursin("AntGenome(size=", compact(ag))
        pag = pretty(ag)
        @test occursin("primitives:", pag)

        # GraphGenome
        reset_innovation_counter!()
        rng = Random.MersenneTwister(42)
        gg = initialize(GraphGenome, 2, 1, rng)
        @test occursin("GraphGenome(nodes=", compact(gg))
        pgg = pretty(gg)
        @test occursin("inputs:", pgg)
        @test occursin("outputs:", pgg)
        @test occursin("connections:", pgg)

        # ADFGenome
        adf_ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        main = Node(1, Node(Float64; feature=1), Node(Float64; feature=2))
        adf = Node(Float64; feature=3)  # ARG0 in n_features+1 space
        # ADFGenome expects an augmented operator enum with N placeholder slots.
        aug_ops = OperatorEnum(binary_operators=[+, -, *, Arborist._adf_placeholder],
                               unary_operators=[sin])
        ag2 = Arborist.ADFGenome{Float64}(main, [adf], 2, aug_ops, 2, 1)
        @test occursin("ADFGenome{Float64}", compact(ag2))
        @test occursin("adfs=1", compact(ag2))
        pag2 = pretty(ag2)
        @test occursin("main", pag2)
        @test occursin("ADF0", pag2)
    end
end
