# Koza symbolic regression suite — uses TreeGenome (DynamicExpressions.jl)
# for fast vectorized evaluation without @eval overhead.
#
# Migrated from ExprGenome in Phase 4. TreeGenome provides dramatic speedup
# over ExprGenome for pure function approximation problems.

using DynamicExpressions
const _KozaDynExt = Base.get_extension(Arborist, :DynExprExt)
const _KozaTreeGenome = _KozaDynExt.TreeGenome
const _KozaTreeEval = _KozaDynExt.TreeFitnessEvaluator

@testset "Koza symbolic regression suite (TreeGenome)" begin
    operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])

    algorithm = GeneticProgramming(
        pop_size=100,
        generations=300,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2,
        tournament_size=3
    )

    xs = Float32.(range(-1, 1, length=20))
    X = reshape(xs, 1, :)

    @testset "Koza-1: x^4 + x^3 + x^2 + x" begin
        y = xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs
        evaluator = _KozaTreeEval(X, y, operators)

        successes = map(1:5) do seed
            problem = GPProblem(evaluator, _KozaTreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            result.best_fitness < 0.1
        end

        n_success = count(successes)
        println("  Koza-1 (TreeGenome): $n_success/5 seeds converged (fitness < 0.1)")
        flush(stdout)
        @test n_success >= 3
    end

    @testset "Koza-2: x^5 - 2x^3 + x" begin
        y = xs .^ 5 .- 2 .* xs .^ 3 .+ xs
        evaluator = _KozaTreeEval(X, y, operators)

        successes = map(1:5) do seed
            problem = GPProblem(evaluator, _KozaTreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            result.best_fitness < 0.1
        end

        n_success = count(successes)
        println("  Koza-2 (TreeGenome): $n_success/5 seeds converged (fitness < 0.1)")
        flush(stdout)
        @test n_success >= 3
    end

    @testset "Koza-3: x^6 - 2x^4 + x^2" begin
        y = xs .^ 6 .- 2 .* xs .^ 4 .+ xs .^ 2
        evaluator = _KozaTreeEval(X, y, operators)

        successes = map(1:5) do seed
            problem = GPProblem(evaluator, _KozaTreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            result.best_fitness < 0.1
        end

        n_success = count(successes)
        println("  Koza-3 (TreeGenome): $n_success/5 seeds converged (fitness < 0.1)")
        flush(stdout)
        @test n_success >= 3
    end
end
