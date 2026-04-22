using Test
using Arborist
using Random

@testset "RunLog" begin
    @testset "empty and basic shape" begin
        log = RunLog()
        @test length(log) == 0
        @test isempty(log)
        @test entries(log) === log.entries
    end

    @testset "record! aggregates correctly with mixed finite/Inf fitnesses" begin
        log = RunLog()
        # Dummy genomes for structure counting — use strings wrapped in a minimal
        # AbstractGenome? Instead, exercise the serialize-hash path via real
        # TreeGenome objects later. Here we test fitness aggregation only.
        fitnesses = Float64[1.0, 2.0, 3.0, Inf, Inf]
        # Use Int as stand-ins for genomes — serialize() will fail -> objectid fallback.
        genomes = [1, 2, 3, 4, 5]
        record!(log, 1, fitnesses, genomes, 0.5)
        @test length(log) == 1
        e = log[1]
        @test e.generation == 1
        @test e.best_fitness == 1.0
        @test e.worst_fitness == 3.0
        @test e.mean_fitness ≈ 2.0
        @test e.median_fitness == 2.0
        @test e.wall_time == 0.5
        @test e.n_species == 1
        @test e.species_sizes == [5]
        @test isempty(e.operator_success)
        @test isempty(e.operator_attempted)
    end

    @testset "all-Inf fitness -> all Inf summary" begin
        log = RunLog()
        record!(log, 1, Float64[Inf, Inf], [1, 2], 0.1)
        e = log[1]
        @test e.best_fitness == Inf
        @test e.mean_fitness == Inf
        @test e.median_fitness == Inf
        @test e.worst_fitness == Inf
    end

    @testset "median of even-length vector" begin
        log = RunLog()
        record!(log, 1, Float64[1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4], 0.0)
        @test log[1].median_fitness == 2.5
    end

    @testset "SpeciationSnapshot populates species fields" begin
        log = RunLog()
        snap = SpeciationSnapshot()
        snap.n_species = 3
        snap.sizes = [4, 3, 2]
        record!(log, 1, Float64[1.0, 2.0, 3.0], [1, 2, 3], 0.0; snapshot=snap)
        e = log[1]
        @test e.n_species == 3
        @test e.species_sizes == [4, 3, 2]
    end

    @testset "RunLog populated through GP solve path (TreeGenome)" begin
        using DynamicExpressions
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        X = reshape(collect(Float32, 0.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
        alg = GeneticProgramming(pop_size=20, generations=5, parallel=false)
        log = RunLog()
        result = solve(problem, alg; log=log)
        @test length(log) == 5
        @test log[1].generation == 1
        @test log[5].generation == 5
        @test log[5].wall_time >= log[1].wall_time
        # structural diversity should be at least 1 (the population has content)
        @test all(e -> e.unique_structures >= 1, entries(log))
    end

    @testset "log=nothing is zero-overhead (identical result)" begin
        using DynamicExpressions
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        X = reshape(collect(Float32, 0.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        evaluator = TreeFitnessEvaluator(X, y, ops)
        problem1 = GPProblem(evaluator, TreeGenome{Float32}; seed=7)
        problem2 = GPProblem(evaluator, TreeGenome{Float32}; seed=7)
        alg = GeneticProgramming(pop_size=15, generations=5, parallel=false)
        r1 = solve(problem1, alg)
        r2 = solve(problem2, alg; log=RunLog())
        # Fitness trajectory must match regardless of logging.
        @test r1.fitness_history == r2.fitness_history
        @test r1.best_fitness == r2.best_fitness
    end
end
