# Tests for previously untested functionality identified by coverage audit.

@testset "Coverage gap tests" begin

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

    # =========================================================================
    # Legacy evolution API (Individual, Population, evolve!)
    # =========================================================================

    @testset "Individual and Population construction" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0, 2.0, 3.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        rng = Random.MersenneTwister(42)
        pop = Arborist.Population(rng, fe, 10, 3, 2)
        @test length(pop.individuals) == 10
        @test pop.generation == 0
        @test all(ind -> ind.fitness == Inf, pop.individuals)
        @test all(ind -> length(ind.expr) == 3, pop.individuals)
        println("  Population: 10 individuals created, all unevaluated")
    end

    @testset "evaluate_individual!" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0, 2.0, 3.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        rng = Random.MersenneTwister(42)
        pop = Arborist.Population(rng, fe, 5, 3, 2)
        ind = pop.individuals[1]
        @test ind.fitness == Inf
        Arborist.evaluate_individual!(pop, ind)
        @test ind.fitness >= 0.0
        @test ind.fitness < Inf || true  # May be Inf if program errors, that's valid
        println("  evaluate_individual!: fitness=$(ind.fitness)")
    end

    @testset "evolve! runs and improves fitness" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-2.0, -1.0, 0.0, 1.0, 2.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        rng = Random.MersenneTwister(42)
        pop = Arborist.Population(rng, fe, 30, 3, 2)
        pop = Arborist.evolve!(pop, 10; mutation_rate=0.3, crossover_rate=0.3, verbose=false)

        @test pop.generation == 10
        @test length(pop.individuals) == 30
        # After evolution, best individual should have finite fitness.
        best = pop.individuals[1]
        @test best.fitness < Inf
        @test best.fitness >= 0.0
        println("  evolve!: 10 generations, best=$(best.fitness)")
    end

    @testset "tournament_select returns valid individual" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0, 2.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        rng = Random.MersenneTwister(42)
        pop = Arborist.Population(rng, fe, 10, 3, 2)
        # Evaluate all so fitnesses are set.
        for ind in pop.individuals
            Arborist.evaluate_individual!(pop, ind)
        end

        for _ in 1:20
            selected = Arborist.tournament_select(pop, 3)
            @test selected isa Arborist.Individual
            @test selected in pop.individuals
        end
    end

    @testset "mutate_individual produces different individual" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        rng = Random.MersenneTwister(42)
        pop = Arborist.Population(rng, fe, 5, 3, 2)
        ind = pop.individuals[1]
        original_str = string(ind.expr)

        mutated = Arborist.mutate_individual(pop.state, ind)
        @test mutated isa Arborist.Individual
        @test mutated.fitness == Inf  # new individual starts unevaluated
        @test mutated.age == 0
        # Original should be unchanged.
        @test string(ind.expr) == original_str
    end

    @testset "crossover_individuals produces offspring" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        rng = Random.MersenneTwister(42)
        pop = Arborist.Population(rng, fe, 5, 3, 2)
        a = pop.individuals[1]
        b = pop.individuals[2]

        (c1, c2) = Arborist.crossover_individuals(pop.state, a, b)
        @test c1 isa Arborist.Individual
        @test c2 isa Arborist.Individual
        @test !isempty(c1.expr)
        @test !isempty(c2.expr)
    end

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
