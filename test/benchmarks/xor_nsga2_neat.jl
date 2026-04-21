# NSGA-II GraphGenome infrastructure benchmark.
#
# XOR with ParsimonyEvaluator (fitness vs. number of enabled connections).
# This demonstrates NSGA-II dispatch for GraphGenome but is *not* a solve
# benchmark — vanilla NSGA-II + parsimony on XOR has a well-known attractor:
# the zero-connection network always achieves MSE=0.25 at complexity=0, which
# Pareto-dominates every higher-complexity network that hasn't yet found a
# sub-0.25 solution. The population typically converges to the trivial
# extreme unless you add speciation, niching, or a separate diversity
# mechanism to protect structural innovations long enough to cross the
# fitness plateau. Stanley & Miikkulainen (2002) recommend fitness sharing
# for exactly this reason.
#
# What this benchmark asserts is that the NSGA-II + GraphGenome wiring
# itself is correct: the solver returns an `NSGAIIResult{GraphGenome}`,
# records a hypervolume history, and produces a non-empty Pareto front.
# For a problem where parsimony + NEAT actually succeeds, a future benchmark
# can mirror this one with speciation enabled or a different task.

@testset "XOR NSGA-II NEAT infrastructure (GraphGenome Pareto front)" begin
    reset_innovation_counter!()

    input_data = Float64[0 0 1 1; 0 1 0 1]
    output_data = Float64[0 1 1 0]
    evaluator = ParsimonyEvaluator(GraphEvaluator(input_data, output_data))

    ops = neat_defaults()
    algorithm = NSGAII(
        pop_size=100, generations=50,
        mutation_rate=0.5, crossover_rate=0.3, parallel=false,
        mutation_ops=ops.mutation_ops,
        crossover_ops=ops.crossover_ops,
    )

    problem = GPProblem(evaluator, GraphGenome; seed=2026)
    result = solve(problem, algorithm; verbose=false)

    @test result isa NSGAIIResult{GraphGenome}
    @test result.objective_names == ["fitness", "complexity"]
    @test !isempty(result.pareto_front)
    @test length(result.hypervolume_history) == algorithm.generations
    @test all(g isa GraphGenome for g in result.pareto_front)

    # Fitness values on the front must be finite (no stale Infs leaking through).
    for f in result.pareto_fitnesses
        @test all(isfinite, f)
    end

    # Hypervolume was computed (non-trivial in at least one generation).
    @test any(h -> h > 0.0, result.hypervolume_history)

    # Summary print for human review.
    front_fitness    = [f[1] for f in result.pareto_fitnesses]
    front_complexity = [f[2] for f in result.pareto_fitnesses]
    println("  XOR NSGA-II Pareto front ($(length(result.pareto_front)) genomes):")
    println("    fitness    : min=$(round(minimum(front_fitness), digits=4))  " *
            "max=$(round(maximum(front_fitness), digits=4))")
    println("    complexity : min=$(Int(minimum(front_complexity)))  " *
            "max=$(Int(maximum(front_complexity)))")
    println("    hypervolume[end] = $(round(result.hypervolume_history[end], digits=4))")
    flush(stdout)
end
