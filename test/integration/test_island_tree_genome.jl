# IslandModel + TreeGenome integration tests.
#
# Covers the three IslandModel execution modes (sequential, sync
# distributed, async distributed) plus the migration round-trip path
# that `to_migrant`/`from_migrant` provide for cross-process transfer.

using Distributed
using DynamicExpressions

@testset "IslandModel + TreeGenome" begin

    # ------------------------------------------------------------------
    # Migration round-trip — unit-level check that to_migrant/from_migrant
    # preserve tree structure and evaluation semantics.
    # ------------------------------------------------------------------
    @testset "to_migrant/from_migrant round-trip" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
        # Build tree: x1 * x1 + x1  (op indices: binary + is op=1, binary * is op=3)
        x1   = Node{Float32}(; feature=1)
        sq   = Node{Float32}(; op=3, l=x1, r=x1)
        tree = Node{Float32}(; op=1, l=sq, r=x1)
        g    = TreeGenome{Float32}(tree, ops, 1)

        m = to_migrant(g, 0.125)
        @test m isa MigrantGenome
        @test m.genome_type === :TreeGenome
        @test m.fitness == 0.125

        # A fresh TreeGenomeContext mimics the destination island's state.
        ctx = Arborist.TreeGenomeContext{Float32}(Random.MersenneTwister(0), ops, 1)
        g2 = from_migrant(m, ctx)
        @test g2 isa TreeGenome{Float32}
        @test g2.operators === ops
        @test g2.n_features == 1

        # Evaluate both trees on a common matrix and confirm predictions match.
        X = reshape(Float32[-2.0, -1.0, 0.0, 1.0, 2.0], 1, :)
        p1 = g.tree(X, ops)
        p2 = g2.tree(X, ops)
        @test p1 == p2

        # Mutating the reconstructed tree must not affect the source —
        # from_migrant must deep-copy.
        @test g.tree !== g2.tree
    end

    # ------------------------------------------------------------------
    # Sequential IslandModel + TreeGenome: convergence + determinism.
    # ------------------------------------------------------------------
    @testset "Sequential IslandModel (3 islands)" begin
        ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
        X = reshape(Float32.(range(-2, 2, length=30)), 1, :)
        y = X[1, :].^2 .+ X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)

        inner = GeneticProgramming(
            pop_size=30, generations=20,
            crossover_rate=0.7, mutation_rate=0.2,
            elitism=2,
            selection=TournamentSelection(3),
            mutation_ops=[SubtreeMutation()],
            crossover_ops=[SubtreeCrossover()],
            speciation=NoSpeciation(),
        )

        island_alg = IslandModel(
            n_islands=3,
            island_algorithm=inner,
            migration_interval=5,
            migration_size=2,
            topology=RingTopology(),
            distributed=false,
        )

        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        result = solve(problem, island_alg; verbose=false)

        @test result isa GPResult{TreeGenome{Float32}}
        @test result.best_fitness < 1.0               # loose — not requiring exact convergence
        @test result.best_fitness >= 0.0
        @test length(result.fitness_history) == 20
        @test length(result.population) == 3 * 30
        @test result.best_genome isa TreeGenome{Float32}
        @test Arborist.count_nodes(result.best_genome.tree) >= 1

        # Determinism: a second solve with the same seed must match bit-for-bit.
        problem2 = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        result2 = solve(problem2, island_alg; verbose=false)
        @test result.fitness_history == result2.fitness_history
        @test result.best_fitness == result2.best_fitness
    end

    # ------------------------------------------------------------------
    # Unsupported evaluator: IslandModel+TreeGenome requires TreeFitnessEvaluator.
    # ------------------------------------------------------------------
    @testset "Unsupported evaluator type errors" begin
        # Build a TableFitnessEvaluator (meant for ExprGenome) and try to
        # drive it through the new TreeGenome init path.
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        input_rows = [Dict{Symbol,Any}(:x => 1.0f0)]
        output_rows = [Dict{Symbol,Any}(:y => 1.0f0)]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        problem = GPProblem(fe, TreeGenome{Float32}; seed=42)
        inner = GeneticProgramming(pop_size=10, generations=5)
        island_alg = IslandModel(n_islands=2, island_algorithm=inner, distributed=false)

        @test_throws ErrorException solve(problem, island_alg; verbose=false)
    end

    # ------------------------------------------------------------------
    # Sync distributed IslandModel + TreeGenome: 2 workers, 2 islands.
    # ------------------------------------------------------------------
    @testset "Sync distributed IslandModel (2 workers)" begin
        added = addprocs(2; exeflags="--project=$(Base.active_project())")
        @everywhere using Arborist
        @everywhere using DynamicExpressions

        try
            ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
            X = reshape(Float32.(range(-2, 2, length=30)), 1, :)
            y = X[1, :].^2 .+ X[1, :]
            evaluator = TreeFitnessEvaluator(X, y, ops)

            inner = GeneticProgramming(
                pop_size=20, generations=15,
                crossover_rate=0.7, mutation_rate=0.2,
                elitism=2,
                selection=TournamentSelection(3),
                mutation_ops=[SubtreeMutation()],
                crossover_ops=[SubtreeCrossover()],
                speciation=NoSpeciation(),
                parallel=false,
            )

            island_alg = IslandModel(
                n_islands=2,
                island_algorithm=inner,
                migration_interval=5,
                migration_size=2,
                topology=RingTopology(),
                distributed=true,
                async=false,
            )

            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
            result = solve(problem, island_alg; verbose=false)

            @test result isa GPResult{TreeGenome{Float32}}
            @test result.best_fitness >= 0.0
            @test result.best_fitness < Inf
            @test length(result.fitness_history) == 15
            @test length(result.population) == 2 * 20
            @test result.best_genome isa TreeGenome{Float32}
        finally
            rmprocs(added)
        end
    end

    # ------------------------------------------------------------------
    # Async distributed IslandModel + TreeGenome: 2 workers, 2 islands.
    # Non-deterministic — only convergence/validity is asserted.
    # ------------------------------------------------------------------
    @testset "Async distributed IslandModel (2 workers)" begin
        added = addprocs(2; exeflags="--project=$(Base.active_project())")
        @everywhere using Arborist
        @everywhere using DynamicExpressions

        try
            ops = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
            X = reshape(Float32.(range(-2, 2, length=30)), 1, :)
            y = X[1, :].^2 .+ X[1, :]
            evaluator = TreeFitnessEvaluator(X, y, ops)

            inner = GeneticProgramming(
                pop_size=20, generations=15,
                crossover_rate=0.7, mutation_rate=0.2,
                elitism=2,
                selection=TournamentSelection(3),
                mutation_ops=[SubtreeMutation()],
                crossover_ops=[SubtreeCrossover()],
                speciation=NoSpeciation(),
                parallel=false,
            )

            island_alg = IslandModel(
                n_islands=2,
                island_algorithm=inner,
                migration_interval=5,
                migration_size=2,
                topology=RingTopology(),
                distributed=true,
                async=true,
            )

            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
            result = solve(problem, island_alg; verbose=false)

            @test result isa GPResult{TreeGenome{Float32}}
            @test result.best_fitness >= 0.0
            @test result.best_fitness < Inf
            @test length(result.population) == 2 * 20
            @test !isempty(result.fitness_history)
            @test result.wall_time > 0.0
        finally
            rmprocs(added)
        end
    end
end
