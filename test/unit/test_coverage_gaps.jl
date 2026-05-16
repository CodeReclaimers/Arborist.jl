# Tests for previously untested functionality identified by coverage audit.
using Test
using Arborist
using Random

struct AlwaysBadMutation <: AbstractMutationOperator end
Arborist.mutate(::AlwaysBadMutation, g::ExprGenome, rng::AbstractRNG) =
    ExprGenome([:(out = x + 10.0)], g.state)

@testset "Coverage gap tests" begin

    # =========================================================================
    # Operator rate validation
    # =========================================================================

    @testset "GeneticProgramming rejects rates summing > 1.0" begin
        @test_throws ArgumentError GeneticProgramming(crossover_rate=0.8, mutation_rate=0.5)
        @test_throws ArgumentError GeneticProgramming(crossover_rate=0.6, mutation_rate=0.6)
        @test_throws ArgumentError GeneticProgramming(crossover_rate=1.0, mutation_rate=0.01)
        @test_throws ArgumentError GeneticProgramming(crossover_rate=-0.1)
        @test_throws ArgumentError GeneticProgramming(mutation_rate=-0.1)
        @test_throws ArgumentError GeneticProgramming(crossover_rate=NaN)
        @test_throws ArgumentError GeneticProgramming(mutation_rate=NaN)
        # Boundary: exactly 1.0 is allowed (no reproduction, but valid)
        alg = GeneticProgramming(crossover_rate=0.7, mutation_rate=0.3)
        @test alg.crossover_rate == 0.7
        @test alg.mutation_rate == 0.3
        # Default rates are valid
        alg2 = GeneticProgramming()
        @test alg2.crossover_rate + alg2.mutation_rate <= 1.0
        println("  Operator rate validation: invalid rates rejected, valid rates accepted")
        flush(stdout)
    end

    @testset "GeneticProgramming validates public scalar config" begin
        @test_throws ArgumentError GeneticProgramming(pop_size=0)
        @test_throws ArgumentError GeneticProgramming(pop_size=-1)
        @test_throws ArgumentError GeneticProgramming(generations=-1)
        @test_throws ArgumentError GeneticProgramming(elitism=-1)
        @test_throws ArgumentError GeneticProgramming(pop_size=4, elitism=5)
        @test_throws ArgumentError GeneticProgramming(bloat_penalty=-0.1)
        @test_throws ArgumentError IslandModel(n_islands=0)
        @test_throws ArgumentError IslandModel(migration_interval=0)
        @test_throws ArgumentError IslandModel(migration_size=-1)

        alg = GeneticProgramming(;
            pop_size=4,
            elitism=4,
            generations=0,
            mutation_rate=1.0,
            crossover_rate=0.0,
            crossover_ops=AbstractCrossoverOperator[])
        @test alg.generations == 0
        @test isempty(alg.crossover_ops)
    end

    @testset "_validate_ops respects zero operator rates" begin
        muts = AbstractMutationOperator[WeightPerturbMutation()]
        xos = AbstractCrossoverOperator[NEATCrossover()]

        @test Arborist._validate_ops(muts, AbstractCrossoverOperator[], GraphGenome;
                                     mutation_rate=1.0, crossover_rate=0.0) === nothing
        @test Arborist._validate_ops(AbstractMutationOperator[], xos, GraphGenome;
                                     mutation_rate=0.0, crossover_rate=1.0) === nothing
        @test_throws ArgumentError Arborist._validate_ops(
            muts, AbstractCrossoverOperator[], GraphGenome;
            mutation_rate=1.0, crossover_rate=0.1)
        @test_throws ArgumentError Arborist._validate_ops(
            AbstractMutationOperator[], xos, GraphGenome;
            mutation_rate=0.1, crossover_rate=1.0)
    end

    @testset "_validate_ops requires every operator to dispatch (not just one)" begin
        # The breed loop samples operators uniformly via rand(rng, ops), so a
        # single non-dispatching operator in a mixed list will eventually
        # crash mid-run with a MethodError. Validation must reject the
        # mixed list at configuration time.
        good_mut = AbstractMutationOperator[SubtreeMutation()]                 # dispatches on ExprGenome
        bad_mut_for_expr = AbstractMutationOperator[WeightPerturbMutation()]   # GraphGenome-only
        mixed_mut = AbstractMutationOperator[SubtreeMutation(),
                                              WeightPerturbMutation()]

        good_xo = AbstractCrossoverOperator[SubtreeCrossover()]                # dispatches on ExprGenome
        bad_xo_for_expr = AbstractCrossoverOperator[NEATCrossover()]           # GraphGenome-only
        mixed_xo = AbstractCrossoverOperator[SubtreeCrossover(), NEATCrossover()]

        # All-good lists still pass (regression guard).
        @test Arborist._validate_ops(good_mut, good_xo, ExprGenome) === nothing

        # All-bad lists fail (covered by other tests too — confirm here for parity).
        @test_throws ArgumentError Arborist._validate_ops(bad_mut_for_expr, good_xo, ExprGenome)
        @test_throws ArgumentError Arborist._validate_ops(good_mut, bad_xo_for_expr, ExprGenome)

        # The new contract: mixed lists are rejected, and the error names the
        # offending operator type.
        err_m = try
            Arborist._validate_ops(mixed_mut, good_xo, ExprGenome)
        catch e; e end
        @test err_m isa ArgumentError
        @test occursin("WeightPerturbMutation", sprint(showerror, err_m))

        err_x = try
            Arborist._validate_ops(good_mut, mixed_xo, ExprGenome)
        catch e; e end
        @test err_x isa ArgumentError
        @test occursin("NEATCrossover", sprint(showerror, err_x))
    end

    # (The legacy evolve! rate-validation testset was removed in 0.1.0
    # when the Individual / Population / evolve! API was retired; the
    # equivalent validation now lives entirely inside GeneticProgramming
    # and is covered by the testset above.)

    # =========================================================================
    # Callback mechanism
    # =========================================================================

    @testset "solve() callback fires every generation" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = GeneticProgramming(pop_size=20, generations=5,
                                        mutation_rate=0.4, crossover_rate=0.2)

        callback_log = Tuple{Int, Float64}[]
        callback_genomes = AbstractGenome[]
        cb = (gen, best_fit, best_genome) -> begin
            push!(callback_log, (gen, best_fit))
            push!(callback_genomes, best_genome)
        end

        result = solve(problem, algorithm; verbose=false, callback=cb)

        # Callback should fire once per generation.
        @test length(callback_log) == 5
        # Generation numbers should be 1..5.
        @test [t[1] for t in callback_log] == [1, 2, 3, 4, 5]
        # Best fitness should be non-negative and monotonically non-increasing.
        for i in 2:length(callback_log)
            @test callback_log[i][2] <= callback_log[i-1][2] + 1e-10
        end
        # Callback should receive actual ExprGenome objects.
        @test all(g -> g isa ExprGenome, callback_genomes)

        println("  callback: $(length(callback_log)) calls, final best=$(callback_log[end][2])")
    end

    @testset "solve() reports best from initial population if later generations regress" begin
        input_cols = Dict(:x => Float64)
        output_cols = Dict(:out => Float64)
        input_rows = [Dict{Symbol,Any}(:x => 1.0), Dict{Symbol,Any}(:x => 2.0)]
        output_rows = [Dict{Symbol,Any}(:out => 1.0), Dict{Symbol,Any}(:out => 2.0)]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = default_function_set()
        rng = Random.MersenneTwister(12)
        state = GenState(rng, fset, input_cols, output_cols, 1)
        initial_population = [
            ExprGenome([:(out = x)], state),
            ExprGenome([:(out = x + 5.0)], state),
        ]
        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=1, seed=12)
        algorithm = GeneticProgramming(;
            pop_size=2,
            generations=1,
            elitism=0,
            mutation_rate=1.0,
            crossover_rate=0.0,
            mutation_ops=AbstractMutationOperator[AlwaysBadMutation()],
            parallel=false)

        result = solve(problem, algorithm; initial_population=initial_population)

        @test result.best_fitness == 0.0
        @test only(result.best_genome.body) == :(out = x)
        @test minimum(result.fitness_history) == 0.0
    end

    # =========================================================================
    # Convergence threshold
    # =========================================================================

    @testset "convergence_threshold sets converged flag" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        # Trivial problem: y = x (identity).
        xs = Float32[1.0, 2.0, 3.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)

        # Very high threshold: should always converge.
        alg_easy = GeneticProgramming(pop_size=50, generations=20,
                                       convergence_threshold=1e6)
        result_easy = solve(problem, alg_easy; verbose=false)
        @test result_easy.converged == true
        println("  convergence (easy threshold): converged=$(result_easy.converged), best=$(result_easy.best_fitness)")

        # Impossibly low threshold: should not converge.
        alg_hard = GeneticProgramming(pop_size=20, generations=5,
                                       convergence_threshold=-1.0)
        result_hard = solve(problem, alg_hard; verbose=false)
        @test result_hard.converged == false
        println("  convergence (impossible threshold): converged=$(result_hard.converged), best=$(result_hard.best_fitness)")
    end

    @testset "default convergence_threshold does not mark finite runs converged" begin
        input_cols = Dict(:x => Float64)
        output_cols = Dict(:out => Float64)
        input_rows = [Dict{Symbol,Any}(:x => 1.0)]
        output_rows = [Dict{Symbol,Any}(:out => 1.0)]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        problem = GPProblem(fe, ExprGenome; function_set=default_function_set(),
                            num_temps=1, seed=1)
        alg = GeneticProgramming(pop_size=4, generations=1, parallel=false)
        result = solve(problem, alg)

        @test isfinite(result.best_fitness)
        @test result.converged == false
    end

    # =========================================================================
    # Boolean function set and boolean operators
    # =========================================================================

    @testset "boolean_function_set and gp_nand/gp_nor" begin
        bfset = boolean_function_set()
        @test bfset isa FunctionSet
        @test length(bfset.funcs) >= 6

        # Verify gp_nand truth table.
        @test Arborist.gp_nand(false, false) == true
        @test Arborist.gp_nand(false, true) == true
        @test Arborist.gp_nand(true, false) == true
        @test Arborist.gp_nand(true, true) == false

        # Verify gp_nor truth table.
        @test Arborist.gp_nor(false, false) == true
        @test Arborist.gp_nor(false, true) == false
        @test Arborist.gp_nor(true, false) == false
        @test Arborist.gp_nor(true, true) == false

        println("  boolean operators: gp_nand and gp_nor truth tables verified")
    end

    @testset "boolean_function_set generates valid assignments" begin
        bfset = boolean_function_set()
        rng = Random.MersenneTwister(42)
        s = GenState(rng, bfset, Dict(:a => Bool, :b => Bool), Dict(:c => Bool), 2)
        for _ in 1:50
            assignment = create_random_assignment(s)
            @test assignment isa Expr
            @test assignment.head == :(=)
        end
        println("  boolean function set: 50 random assignments generated successfully")
    end

    # (The legacy Individual / Population / evolve! / evaluate_individual! /
    # tournament_select / mutate_individual / crossover_individuals tests
    # were removed in 0.1.0 when the legacy imperative API was retired in
    # favor of the Problem/Algorithm/Solve path. The canonical API is
    # covered by test_operators.jl, test_genome.jl, and the integration
    # tests in test/integration/.)

    # =========================================================================
    # _is_valid_call (was broken by field name typo, now fixed)
    # =========================================================================

    @testset "deserialize standalone function calls" begin
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        add!(fset, :sin, 1, Float32, Float32)
        rng = Random.MersenneTwister(42)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        # A standalone function call should be accepted as a valid statement.
        s = "sin(x)"
        g = deserialize(ExprGenome, s, state)
        @test g !== nothing
        @test length(g.body) >= 1
        @test any(e -> e isa Expr && e.head == :call, g.body)
        println("  standalone function call: deserialized successfully")
    end

    # =========================================================================
    # wrap_rvalue edge case (now returns value unchanged for unsupported arity)
    # =========================================================================

    @testset "wrap_rvalue returns value for unsupported arity" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        s = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        # wrap_rvalue should always return a non-nothing value.
        for _ in 1:20
            result = Arborist.wrap_rvalue(s, :x)
            @test result !== nothing
        end
    end

    # =========================================================================
    # GPResult field access
    # =========================================================================

    @testset "GPResult fields are all populated" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = GeneticProgramming(pop_size=20, generations=5)
        result = solve(problem, algorithm; verbose=false)

        @test result.best_genome isa ExprGenome
        @test result.best_fitness isa Float64
        @test result.best_fitness >= 0.0
        @test result.population isa Vector{ExprGenome}
        @test length(result.population) == 20
        @test result.fitness_history isa Vector{Float64}
        @test length(result.fitness_history) == 5
        @test result.mean_history isa Vector{Float64}
        @test length(result.mean_history) == 5
        @test result.generations_run == 5
        @test result.wall_time > 0.0
        @test result.converged isa Bool
        println("  GPResult: all fields populated, wall_time=$(round(result.wall_time, digits=3))s")
    end

    # =========================================================================
    # IslandModel callback
    # =========================================================================

    @testset "IslandModel solve() callback fires every generation" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        island_alg = GeneticProgramming(pop_size=15, generations=5,
                                         mutation_rate=0.4, crossover_rate=0.2)
        algorithm = IslandModel(island_algorithm=island_alg, n_islands=3, migration_interval=2)

        callback_count = Ref(0)
        cb = (gen, best_fit, best_genome) -> (callback_count[] += 1)

        result = solve(problem, algorithm; verbose=false, callback=cb)
        @test callback_count[] == 5
        @test result isa GPResult{ExprGenome}
        println("  IslandModel callback: $(callback_count[]) calls")
    end

    # =========================================================================
    # BehavioralSpeciation integration with solve
    # =========================================================================

    @testset "BehavioralSpeciation integrates with solve" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end

        # Use a simple fingerprint based on body length.
        fp_fn = g -> length(g.body)
        dist_fn = (a, b) -> Float64(abs(a - b))

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = GeneticProgramming(
            pop_size=30, generations=10,
            mutation_rate=0.4, crossover_rate=0.2,
            speciation=BehavioralSpeciation(fingerprint_fn=fp_fn, distance_fn=dist_fn,
                                             threshold=2.0)
        )

        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        @test length(result.fitness_history) == 10
        @test result.best_fitness >= 0.0
        println("  BehavioralSpeciation + solve: best=$(round(result.best_fitness, digits=4))")
    end

    # =========================================================================
    # perturb_literal and get_random_literal for all types
    # =========================================================================

    @testset "perturb_literal handles all types" begin
        rng = Random.MersenneTwister(42)

        # Bool
        for _ in 1:10
            result = Arborist.perturb_literal(rng, true)
            @test result isa Bool
        end

        # Int32
        for _ in 1:10
            result = Arborist.perturb_literal(rng, Int32(5))
            @test result isa Int32
            @test abs(result - 5) <= 2
        end

        # Float32
        for _ in 1:10
            result = Arborist.perturb_literal(rng, Float32(1.0))
            @test result isa Float32
        end

        println("  perturb_literal: Bool, Int32, Float32 all handled")
    end

    @testset "get_random_literal handles all types" begin
        rng = Random.MersenneTwister(42)

        for _ in 1:10
            @test Arborist.get_random_literal(rng, Bool) isa Bool
            @test Arborist.get_random_literal(rng, Int32) isa Int32
            @test Arborist.get_random_literal(rng, Float32) isa Float32
        end
    end

    # =========================================================================
    # Elitism edge cases
    # =========================================================================

    @testset "elitism=0 works" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0, 2.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=1, seed=42)
        algorithm = GeneticProgramming(pop_size=10, generations=3, elitism=0)
        result = solve(problem, algorithm; verbose=false)
        @test result isa GPResult{ExprGenome}
        println("  elitism=0: completed, best=$(round(result.best_fitness, digits=4))")
    end

    # =========================================================================
    # Solve with verbose=true (smoke test for println path)
    # =========================================================================

    @testset "verbose=true does not crash" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=1, seed=42)
        algorithm = GeneticProgramming(pop_size=10, generations=2)
        result = solve(problem, algorithm; verbose=true)
        @test result isa GPResult{ExprGenome}
    end

end
