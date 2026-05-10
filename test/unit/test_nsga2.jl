using DynamicExpressions: OperatorEnum, Node

@testset "NSGA-II" begin

    # =========================================================================
    # NSGAII construction
    # =========================================================================

    @testset "NSGAII construction" begin
        alg = NSGAII()
        @test alg.pop_size == 100
        @test alg.generations == 200
        @test alg.mutation_rate == 0.3
        @test alg.crossover_rate == 0.3
        @test alg.parallel == true

        alg2 = NSGAII(pop_size=50, generations=30)
        @test alg2.pop_size == 50
        @test alg2.generations == 30

        # pop_size must be even
        @test_throws ArgumentError NSGAII(pop_size=51)
        # pop_size must be >= 4
        @test_throws ArgumentError NSGAII(pop_size=2)
        # rates must sum to <= 1.0
        @test_throws ArgumentError NSGAII(mutation_rate=0.6, crossover_rate=0.6)
        @test_throws ArgumentError NSGAII(generations=-1)
        @test_throws ArgumentError NSGAII(mutation_rate=-0.1)
        @test_throws ArgumentError NSGAII(crossover_rate=-0.1)
        @test_throws ArgumentError NSGAII(mutation_rate=NaN)
        @test_throws ArgumentError NSGAII(crossover_rate=NaN)
    end

    # =========================================================================
    # Non-dominated sorting
    # =========================================================================

    @testset "Non-dominated sorting" begin
        # All non-dominated (each is best in one objective).
        fitnesses = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        ranks = Arborist._nondominated_sort(fitnesses)
        @test all(r == 1 for r in ranks)
        println("  _nondominated_sort: all non-dominated -> ranks=$ranks")

        # Clear hierarchy: a dominates b, b dominates c.
        fitnesses2 = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        ranks2 = Arborist._nondominated_sort(fitnesses2)
        @test ranks2 == [1, 2, 3]
        println("  _nondominated_sort: hierarchy -> ranks=$ranks2")

        # Identical fitnesses: all rank 1 (no domination).
        fitnesses3 = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        ranks3 = Arborist._nondominated_sort(fitnesses3)
        @test all(r == 1 for r in ranks3)
        println("  _nondominated_sort: identical -> ranks=$ranks3")

        # Single individual.
        fitnesses4 = [[5.0, 5.0]]
        ranks4 = Arborist._nondominated_sort(fitnesses4)
        @test ranks4 == [1]

        # Mixed: front 1 = {(1,3), (3,1)}, front 2 = {(4,4)}.
        fitnesses5 = [[1.0, 3.0], [3.0, 1.0], [4.0, 4.0]]
        ranks5 = Arborist._nondominated_sort(fitnesses5)
        @test ranks5[1] == 1  # (1,3) non-dominated
        @test ranks5[2] == 1  # (3,1) non-dominated
        @test ranks5[3] == 2  # (4,4) dominated by both
        println("  _nondominated_sort: mixed -> ranks=$ranks5")
    end

    # =========================================================================
    # Crowding distance
    # =========================================================================

    @testset "Crowding distance" begin
        # Single member: Inf.
        fitnesses = [[1.0, 1.0]]
        cd = Arborist._crowding_distance(fitnesses, [1])
        @test cd == [Inf]

        # Two members: both Inf.
        fitnesses2 = [[1.0, 3.0], [3.0, 1.0]]
        cd2 = Arborist._crowding_distance(fitnesses2, [1, 2])
        @test cd2 == [Inf, Inf]

        # Three members: boundaries Inf, interior finite.
        fitnesses3 = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
        cd3 = Arborist._crowding_distance(fitnesses3, [1, 2, 3])
        @test isinf(cd3[1])
        @test isinf(cd3[3])
        @test cd3[2] > 0.0  # interior, finite
        @test isfinite(cd3[2])
        println("  _crowding_distance: 3 members -> $cd3")

        # Four members: two boundary Inf, two interior finite.
        fitnesses4 = [[1.0, 5.0], [2.0, 3.0], [3.0, 2.0], [5.0, 1.0]]
        cd4 = Arborist._crowding_distance(fitnesses4, [1, 2, 3, 4])
        @test isinf(cd4[1])
        @test isinf(cd4[4])
        @test cd4[2] > 0.0 && isfinite(cd4[2])
        @test cd4[3] > 0.0 && isfinite(cd4[3])
    end

    # =========================================================================
    # NSGA-II tournament selection
    # =========================================================================

    @testset "NSGA-II tournament selection" begin
        rng = Random.MersenneTwister(42)

        # Different ranks: lower rank should win more often.
        ranks = [1, 2, 3]
        crowding = [1.0, 1.0, 1.0]
        wins = zeros(Int, 3)
        for _ in 1:1000
            idx = Arborist._nsga2_tournament_select(ranks, crowding, rng)
            wins[idx] += 1
        end
        # Rank 1 should win most often.
        @test wins[1] > wins[2]
        @test wins[1] > wins[3]
        println("  Tournament selection (rank): wins=$wins (rank 1 should dominate)")

        # Same rank, different crowding: higher crowding should win more often.
        ranks2 = [1, 1, 1]
        crowding2 = [10.0, 1.0, 0.1]
        wins2 = zeros(Int, 3)
        for _ in 1:1000
            idx = Arborist._nsga2_tournament_select(ranks2, crowding2, rng)
            wins2[idx] += 1
        end
        @test wins2[1] > wins2[3]
        println("  Tournament selection (crowding): wins=$wins2 (high crowding should dominate)")
    end

    # =========================================================================
    # Hypervolume 2D
    # =========================================================================

    @testset "Hypervolume 2D" begin
        # Single point at origin, ref at (1,1) -> area = 1.0.
        hv = Arborist._hypervolume_2d([[0.0, 0.0]], [1.0, 1.0])
        @test hv ≈ 1.0
        println("  Hypervolume: single point (0,0), ref (1,1) -> $hv")

        # Two points forming an L-shape.
        hv2 = Arborist._hypervolume_2d([[0.0, 0.5], [0.5, 0.0]], [1.0, 1.0])
        @test hv2 ≈ 0.75  # 1.0 - 0.5*0.5 = 0.75
        println("  Hypervolume: L-shape -> $hv2")

        # Empty front.
        hv3 = Arborist._hypervolume_2d(Vector{Float64}[], [1.0, 1.0])
        @test hv3 == 0.0

        # Points beyond reference point are excluded.
        hv4 = Arborist._hypervolume_2d([[0.0, 0.0], [2.0, 2.0]], [1.0, 1.0])
        @test hv4 ≈ 1.0  # only the (0,0) point counts
    end

    # =========================================================================
    # Hypervolume N-D (HSO recursion, Phase F follow-up)
    # =========================================================================

    @testset "Hypervolume N-D" begin
        # 1-D closed form: ref - min.
        hv1 = Arborist._hypervolume_nd([[0.25], [0.75]], [1.0])
        @test hv1 ≈ 0.75
        println("  Hypervolume 1-D: min(0.25,0.75) vs ref 1.0 -> $hv1")

        # 3-D unit cube from origin: ref (1,1,1) -> volume 1.
        hv3_unit = Arborist._hypervolume_nd([[0.0, 0.0, 0.0]], [1.0, 1.0, 1.0])
        @test hv3_unit ≈ 1.0

        # 4-D unit hypercube from origin: ref (1,1,1,1) -> volume 1.
        hv4_unit = Arborist._hypervolume_nd([[0.0, 0.0, 0.0, 0.0]],
                                             [1.0, 1.0, 1.0, 1.0])
        @test hv4_unit ≈ 1.0

        # 3-D dominated point contributes nothing (origin already covers
        # everything inside ref (1,1,1)).
        hv3_dom = Arborist._hypervolume_nd([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                                            [1.0, 1.0, 1.0])
        @test hv3_dom ≈ 1.0

        # 3-D hand-computed front against ref (3.3, 4.4, 5.5):
        #   slab z in [2,3): HV_2d([(3,1)], (3.3,4.4)) = 0.3*3.4 = 1.02
        #   slab z in [3,5): HV_2d([(3,1),(2,2)], (3.3,4.4)) = 3.42
        #   slab z in [5,5.5]: HV_2d(all three, (3.3,4.4))   = 3.82
        # total = 1*1.02 + 2*3.42 + 0.5*3.82 = 9.77
        hv3_hand = Arborist._hypervolume_nd([[1.0, 4.0, 5.0],
                                              [2.0, 2.0, 3.0],
                                              [3.0, 1.0, 2.0]],
                                             [3.3, 4.4, 5.5])
        @test hv3_hand ≈ 9.77 atol=1e-9
        println("  Hypervolume 3-D: hand-computed slab sum -> $hv3_hand")

        # Empty front and points beyond reference both produce zero.
        @test Arborist._hypervolume_nd(Vector{Float64}[], [1.0, 1.0, 1.0]) == 0.0
        @test Arborist._hypervolume_nd([[2.0, 2.0, 2.0]], [1.0, 1.0, 1.0]) == 0.0

        # Regression: _compute_hypervolume must be non-zero for a non-degenerate
        # 3-objective front. Prior to the HSO fix it always returned 0.0 for
        # n_objectives != 2.
        fits3 = [[1.0, 4.0, 5.0], [2.0, 2.0, 3.0], [3.0, 1.0, 2.0]]
        ranks3 = [1, 1, 1]
        hv3_dispatch = Arborist._compute_hypervolume(fits3, ranks3)
        @test hv3_dispatch > 0.0
        println("  _compute_hypervolume 3-objective dispatch -> $hv3_dispatch")

        # 4-objective dispatch sanity.
        fits4 = [[1.0, 4.0, 5.0, 2.0], [2.0, 2.0, 3.0, 1.0]]
        ranks4 = [1, 1]
        @test Arborist._compute_hypervolume(fits4, ranks4) > 0.0
    end

    # =========================================================================
    # ParsimonyEvaluator
    # =========================================================================

    @testset "ParsimonyEvaluator" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
        xs = Float32.(range(-1, 1, length=10))
        X = reshape(xs, 1, :)
        y = xs .^ 2

        inner = TreeFitnessEvaluator(X, y, ops)
        pe = ParsimonyEvaluator(inner)

        @test pe isa AbstractMultiObjectiveEvaluator
        @test pe isa AbstractEvaluator
        @test objective_names(pe) == ["fitness", "complexity"]
        @test input_signature(pe) == input_signature(inner)
        @test output_signature(pe) == output_signature(inner)

        # evaluate_multi returns [fitness, complexity].
        tree = Node{Float32}(; feature=UInt16(1))  # just x1
        genome = TreeGenome{Float32}(tree, ops, 1)
        result = evaluate_multi(pe, genome)
        @test length(result) == 2
        @test result[1] >= 0.0  # MSE
        @test result[2] == complexity(genome)
        println("  ParsimonyEvaluator: evaluate_multi -> fitness=$(round(result[1], digits=4)), complexity=$(result[2])")
    end

    # =========================================================================
    # Solve with TreeGenome
    # =========================================================================

    @testset "NSGA-II solve with TreeGenome" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
        xs = Float32.(range(-1, 1, length=20))
        X = reshape(xs, 1, :)
        y = xs .^ 2 .+ xs

        inner = TreeFitnessEvaluator(X, y, ops)
        evaluator = ParsimonyEvaluator(inner)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        algorithm = NSGAII(pop_size=50, generations=20,
                           mutation_rate=0.4, crossover_rate=0.3,
                           parallel=false)

        result = solve(problem, algorithm; verbose=false)

        @test result isa NSGAIIResult{TreeGenome{Float32}}
        @test length(result.pareto_front) > 0
        @test length(result.pareto_front) == length(result.pareto_fitnesses)
        @test length(result.population) == 50
        @test length(result.all_fitnesses) == 50
        @test length(result.hypervolume_history) == 20
        @test result.generations_run == 20
        @test result.wall_time > 0.0
        @test result.objective_names == ["fitness", "complexity"]

        # All Pareto front members should have 2-element fitness vectors.
        for f in result.pareto_fitnesses
            @test length(f) == 2
            @test f[1] >= 0.0  # MSE
            @test f[2] >= 1.0  # at least 1 node
        end

        # Pareto front members should be mutually non-dominated.
        for i in 1:length(result.pareto_front)
            for j in (i+1):length(result.pareto_front)
                fi = result.pareto_fitnesses[i]
                fj = result.pareto_fitnesses[j]
                @test !Arborist._dominates(fi, fj) || !Arborist._dominates(fj, fi)
            end
        end

        println("  NSGA-II TreeGenome solve: $(length(result.pareto_front)) Pareto front members")
        println("  Best MSE: $(round(result.pareto_fitnesses[1][1], digits=6))")
        println("  Simplest: complexity=$(result.pareto_fitnesses[end][2])")
    end

    # =========================================================================
    # RunLog wired with NSGA-II hypervolume + front-size metrics
    # =========================================================================

    @testset "NSGA-II RunLog metrics" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
        xs = Float32.(range(-1, 1, length=20))
        X = reshape(xs, 1, :)
        y = xs .^ 2 .+ xs

        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        algorithm = NSGAII(pop_size=20, generations=8,
                           mutation_rate=0.4, crossover_rate=0.3,
                           parallel=false)

        log = RunLog()
        result = solve(problem, algorithm; verbose=false, log=log)

        # One entry per generation.
        @test length(log) == algorithm.generations

        # Hypervolume populated and consistent with NSGAIIResult.
        for (i, entry) in enumerate(log)
            @test !isnan(entry.hypervolume)
            @test entry.hypervolume == result.hypervolume_history[i]
        end

        # Front sizes are non-empty and sum to the population size.
        for entry in log
            @test !isempty(entry.front_sizes)
            @test sum(entry.front_sizes) == algorithm.pop_size
            @test all(s -> s > 0, entry.front_sizes)
        end

        # Final-generation hypervolume should be > 0 for this 2-objective
        # parsimony problem (regression: pre-fix this stayed at 0.0 for
        # ≥3 objectives; the assertion guards the populated path generally).
        @test log[end].hypervolume > 0.0

        # Show methods surface the new fields.
        io = IOBuffer()
        show(io, MIME("text/plain"), log[end])
        s = String(take!(io))
        @test occursin("hypervolume:", s)
        @test occursin("Pareto fronts:", s)

        println("  NSGA-II RunLog: gens=$(length(log)), final hv=$(round(log[end].hypervolume, digits=4)), final fronts=$(log[end].front_sizes)")
    end

    # =========================================================================
    # Backwards-compat: GenerationLog 11-arg constructor still works
    # =========================================================================

    @testset "GenerationLog backward-compat constructor" begin
        # 11-arg positional call (pre-NSGA-II-metrics signature).
        g = Arborist.GenerationLog(0, 0.5, 0.8, 0.7, 1.2, 3, [5, 3, 2],
                                    Dict(:m => 8), Dict(:m => 10), 4, 0.5)
        @test isnan(g.hypervolume)
        @test isempty(g.front_sizes)

        # Single-objective record! call leaves the new fields at their
        # sentinels (NaN / Int[]).
        log = RunLog()
        # Use a minimal stand-in for genomes (only counted via objectid).
        Arborist.record!(log, 1, [0.5, 0.7], [Ref(1), Ref(2)], 0.1)
        @test length(log) == 1
        @test isnan(log[1].hypervolume)
        @test isempty(log[1].front_sizes)
    end

    # =========================================================================
    # Solve with ExprGenome
    # =========================================================================

    @testset "NSGA-II solve with ExprGenome" begin
        # Simple regression: y = x^2.
        inputs = Dict(:x => Float32)
        outputs = Dict(:y => Float32)
        input_rows = [Dict{Symbol,Any}(:x => Float32(v)) for v in range(-2, 2, length=10)]
        output_rows = [Dict{Symbol,Any}(:y => Float32(v)^2) for v in range(-2, 2, length=10)]
        fe = TableFitnessEvaluator(inputs, outputs, input_rows, output_rows)

        evaluator = ParsimonyEvaluator(fe)
        fset = default_function_set()
        problem = GPProblem(evaluator, ExprGenome;
                           function_set=fset, num_temps=2, seed=42)
        algorithm = NSGAII(pop_size=20, generations=10,
                           mutation_rate=0.3, crossover_rate=0.3,
                           parallel=false)

        result = solve(problem, algorithm; verbose=false)

        @test result isa NSGAIIResult{ExprGenome}
        @test length(result.pareto_front) > 0
        @test length(result.population) == 20
        @test length(result.hypervolume_history) == 10
        @test result.generations_run == 10
        @test result.wall_time > 0.0

        println("  NSGA-II ExprGenome solve: $(length(result.pareto_front)) Pareto front members")
    end

    # =========================================================================
    # Verbose output and callback
    # =========================================================================

    @testset "NSGA-II verbose and callback" begin
        ops = OperatorEnum(; binary_operators=[+, -], unary_operators=[abs])
        xs = Float32.(range(-1, 1, length=10))
        X = reshape(xs, 1, :)
        y = xs .^ 2

        evaluator = ParsimonyEvaluator(TreeFitnessEvaluator(X, y, ops))
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=99)
        algorithm = NSGAII(pop_size=20, generations=5, parallel=false)

        callback_log = Tuple{Int, Int, Float64}[]
        cb = (gen, front_size, hv) -> push!(callback_log, (gen, front_size, hv))

        result = solve(problem, algorithm; verbose=false, callback=cb)
        @test length(callback_log) == 5
        @test callback_log[1][1] == 1
        @test callback_log[end][1] == 5
        for (gen, fs, hv) in callback_log
            @test fs > 0
            @test hv >= 0.0
        end
        println("  Callback: $(length(callback_log)) generations logged")
    end

    # =========================================================================
    # GraphGenome + NSGA-II
    # =========================================================================

    @testset "NSGA-II with GraphGenome (Pareto fitness vs complexity)" begin
        # XOR as the target. Pareto front should span a range of connection
        # counts — with ParsimonyEvaluator, the second objective is the number
        # of enabled connections (from `complexity(::GraphGenome)`).
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = ParsimonyEvaluator(GraphEvaluator(input_data, output_data))

        problem = GPProblem(evaluator, GraphGenome; seed=17)
        ops = neat_defaults()
        algorithm = NSGAII(
            pop_size=40, generations=20,
            mutation_rate=0.5, crossover_rate=0.3, parallel=false,
            mutation_ops=ops.mutation_ops,
            crossover_ops=ops.crossover_ops,
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa NSGAIIResult{GraphGenome}
        @test result.objective_names == ["fitness", "complexity"]
        @test !isempty(result.pareto_front)
        @test all(g isa GraphGenome for g in result.pareto_front)

        # Pareto front should include at least one non-trivial point.
        front_fitness     = [f[1] for f in result.pareto_fitnesses]
        front_complexity  = [f[2] for f in result.pareto_fitnesses]
        @test all(isfinite, front_fitness)
        @test all(c -> c >= 0.0, front_complexity)

        # Hypervolume should be positive (reference point is derived from
        # the worst finite objective values scaled by 1.1).
        @test any(h -> h > 0.0, result.hypervolume_history)

        println("  NSGA-II GraphGenome XOR: front=$(length(result.pareto_front)), " *
                "fitness range=[$(round(minimum(front_fitness), digits=4)), " *
                "$(round(maximum(front_fitness), digits=4))], " *
                "complexity range=[$(minimum(front_complexity)), " *
                "$(maximum(front_complexity))]")
        flush(stdout)
    end

    @testset "NSGA-II with GraphGenome rejects bad ops" begin
        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = ParsimonyEvaluator(GraphEvaluator(input_data, output_data))
        problem = GPProblem(evaluator, GraphGenome; seed=1)
        # Default NSGAII ops are ExprGenome-flavored; validation must catch it.
        algorithm = NSGAII(pop_size=10, generations=2)
        @test_throws ArgumentError solve(problem, algorithm; verbose=false)
    end

    @testset "NSGA-II with GraphGenome rejects non-GraphEvaluator" begin
        # ParsimonyEvaluator wrapping a TableFitnessEvaluator should fail the
        # init check — GraphGenome needs a GraphEvaluator inside.
        inputs  = Dict{Symbol, DataType}(:x => Float64)
        outputs = Dict{Symbol, DataType}(:y => Float64)
        input_rows  = [Dict{Symbol, Any}(:x => 1.0), Dict{Symbol, Any}(:x => 2.0)]
        output_rows = [Dict{Symbol, Any}(:y => 2.0), Dict{Symbol, Any}(:y => 4.0)]
        inner = TableFitnessEvaluator(inputs, outputs, input_rows, output_rows)
        eval_bad = ParsimonyEvaluator(inner)
        problem_bad = GPProblem(eval_bad, GraphGenome; seed=1)
        ops = neat_defaults()
        alg = NSGAII(pop_size=10, generations=2,
                     mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops)
        @test_throws ErrorException solve(problem_bad, alg; verbose=false)
    end

end  # @testset "NSGA-II"
