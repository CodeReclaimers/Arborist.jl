# Nguyen symbolic regression benchmark suite — Nguyen-1 through Nguyen-10.
#
# Uytterhoeven (2008) / Nguyen et al. benchmarks as codified in
# McDermott et al. (2012) "Genetic Programming Needs Better Benchmarks."
# These are the de-facto standard beyond Koza for symbolic regression
# comparisons in modern GP papers.
#
# Dataset generation sourced from `Arborist.Benchmarks.nguyen(n)`.

using DynamicExpressions
using Random

@testset "Nguyen symbolic regression suite (TreeGenome)" begin
    # Canonical Nguyen operator set: +, -, *, pdiv / sin, cos, exp, plog, psqrt.
    # Nguyen-flavor protected operators (threshold 1e-6, not the library's 1e-10)
    # live in Arborist.Benchmarks to preserve historical fitness curves.
    operators = OperatorEnum(;
        binary_operators=[+, -, *, Arborist.Benchmarks.nguyen_pdiv],
        unary_operators=[sin, cos, exp,
                         Arborist.Benchmarks.nguyen_plog,
                         Arborist.Benchmarks.nguyen_psqrt]
    )

    algorithm = GeneticProgramming(
        pop_size=100,
        generations=300,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    # run_one: standard 5-seed eval with a common gate.  Returns n_success.
    function run_one(prob; gate_fitness::Float64 = 0.01, n_seeds_required::Int = 3)
        evaluator = TreeFitnessEvaluator(prob.X, prob.y, operators)
        successes = map(1:5) do seed
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
            result = solve(problem, algorithm; verbose=false)
            result.best_fitness < gate_fitness
        end
        n_success = count(successes)
        println("  $(prob.name): $n_success/5 seeds converged (fitness < $gate_fitness)")
        flush(stdout)
        @test n_success >= n_seeds_required
    end

    @testset "Nguyen-1: x^3 + x^2 + x" begin
        run_one(Arborist.Benchmarks.nguyen(1))
    end
    @testset "Nguyen-2: x^4 + x^3 + x^2 + x" begin
        run_one(Arborist.Benchmarks.nguyen(2))
    end
    @testset "Nguyen-3: x^5 + x^4 + x^3 + x^2 + x" begin
        run_one(Arborist.Benchmarks.nguyen(3))
    end
    @testset "Nguyen-4: x^6 + x^5 + x^4 + x^3 + x^2 + x" begin
        run_one(Arborist.Benchmarks.nguyen(4))
    end
    @testset "Nguyen-5: sin(x^2) * cos(x) - 1" begin
        run_one(Arborist.Benchmarks.nguyen(5))
    end
    @testset "Nguyen-6: sin(x) + sin(x + x^2)" begin
        run_one(Arborist.Benchmarks.nguyen(6))
    end
    @testset "Nguyen-7: log(x+1) + log(x^2+1)" begin
        # Notoriously discriminating — relaxed gate (2/5 seeds).
        run_one(Arborist.Benchmarks.nguyen(7); n_seeds_required=2)
    end
    @testset "Nguyen-8: sqrt(x)" begin
        run_one(Arborist.Benchmarks.nguyen(8))
    end
    @testset "Nguyen-9: sin(x) + sin(y^2)" begin
        run_one(Arborist.Benchmarks.nguyen(9))
    end
    @testset "Nguyen-10: 2 * sin(x) * cos(y)" begin
        run_one(Arborist.Benchmarks.nguyen(10))
    end
end
