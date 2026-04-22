using Test
using Arborist
using Random
using DynamicExpressions

# Lexicase selection smoke-test on a modal regression target.
#
# Target: y = x > 0 ? 2x : 3x  on  x ∈ [-2, 2], 41 points.
# Operators: +, -, *  (no conditional).
#
# Purpose: exercise LexicaseSelection and EpsilonLexicaseSelection end-to-end
# through the TreeGenome solve path. This is a test-only benchmark (per Phase
# F D7) — the plan anticipated lexicase might not universally beat tournament
# on smooth regression, and empirically it doesn't: tournament's averaging
# behavior suits piecewise-linear targets well when paired with only arithmetic
# operators. Lexicase's specialist preservation is most valuable on program-
# synthesis and deceptive modal targets with sharper structure.
#
# Gate: both methods finish with finite best_fitness. Numeric comparison is
# printed for human review but not gated.

@testset "Lexicase vs tournament on modal regression" begin
    ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[])
    xs = collect(Float32, -2.0:0.1:2.0)
    ys = Float32[x > 0 ? 2*x : 3*x for x in xs]
    X = reshape(xs, 1, :)

    pop_size = 60
    generations = 30
    n_seeds = 5

    function _run(sel, seed)
        evaluator = TreeFitnessEvaluator(X, ys, ops)
        problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
        alg = GeneticProgramming(
            pop_size=pop_size, generations=generations,
            mutation_rate=0.4, crossover_rate=0.4,
            parallel=false, selection=sel,
        )
        result = solve(problem, alg)
        return result.best_fitness
    end

    tour_fits = Float64[]
    lex_fits  = Float64[]
    elex_fits = Float64[]
    for seed in 1:n_seeds
        tour = _run(TournamentSelection(3), seed)
        lex  = _run(LexicaseSelection(), seed)
        elex = _run(EpsilonLexicaseSelection(), seed)
        push!(tour_fits, tour)
        push!(lex_fits,  lex)
        push!(elex_fits, elex)
        println("  modal-regression seed=$seed: " *
                "tour=$(round(tour, digits=4)), " *
                "lex=$(round(lex, digits=4)), " *
                "elex=$(round(elex, digits=4))")
        flush(stdout)
    end
    println("  means: tour=$(round(sum(tour_fits)/n_seeds, digits=4)), " *
            "lex=$(round(sum(lex_fits)/n_seeds, digits=4)), " *
            "elex=$(round(sum(elex_fits)/n_seeds, digits=4))")
    flush(stdout)

    # Forward-progress gate: both lexicase variants complete and produce
    # finite fitnesses across all seeds. Comparative performance is printed
    # (not gated) — lexicase's advantage is problem-class-dependent.
    @test all(isfinite, lex_fits)
    @test all(isfinite, elex_fits)
    @test all(isfinite, tour_fits)
end
