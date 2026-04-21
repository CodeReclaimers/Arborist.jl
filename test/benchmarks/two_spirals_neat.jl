# Two-spirals classification benchmark using GraphGenome (NEAT-style).
#
# Lang & Witbrock (1989) two-spirals dataset. 194 points = 97 per spiral.
# Canonical hard-for-gradient benchmark: the two classes interleave in a
# spiral pattern that cannot be separated by any polynomial boundary of
# low degree. NEAT (via speciation-protected structural growth) is
# historically one of the gradient-free methods that solves it.
#
# Encoding: 2 inputs (x, y), 1 output with ±1 target (tanh-friendly).

@testset "Two-spirals classification NEAT benchmark (GraphGenome)" begin
    n_per_spiral = 97
    n_total = 2 * n_per_spiral

    input_data = zeros(Float64, 2, n_total)
    output_data = zeros(Float64, 1, n_total)

    # Lang & Witbrock parameterization: angle = k·π/16, radius = 6.5·(104-k)/104.
    for k in 0:(n_per_spiral - 1)
        angle = k * π / 16.0
        radius = 6.5 * (104 - k) / 104.0
        x = radius * sin(angle)
        y = radius * cos(angle)
        # Spiral 1 (class +1) at column 2k+1.
        input_data[1, 2k + 1] = x
        input_data[2, 2k + 1] = y
        output_data[1, 2k + 1] = 1.0
        # Spiral 2 (class -1) at column 2k+2.
        input_data[1, 2k + 2] = -x
        input_data[2, 2k + 2] = -y
        output_data[1, 2k + 2] = -1.0
    end

    evaluator = GraphEvaluator(input_data, output_data)
    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=200,
        generations=500,
        mutation_rate=0.5,
        crossover_rate=0.3,
        elitism=2,
        mutation_ops=ops.mutation_ops,
        crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=20)
    )

    # !!! Two-spirals is genuinely hard for default NEAT hyperparameters.
    # Baseline MSE (zero output vs ±1 targets) is 1.0. At this test-suite
    # budget (pop=200, gen=500), we have measured that best fitness across 5
    # seeds clusters around 0.90–0.92 — NEAT finds *some* improvement over the
    # trivial predictor but is nowhere near "solving" the problem. The
    # literature suggests pop≥500 and gen≥3000 for real convergence, which is
    # too slow for a test-suite benchmark. This test therefore asserts a weak
    # threshold (< 0.95 on ≥3/5 seeds) — just enough to catch a regression to
    # "makes no progress at all." Future work: add a benchmark variant with
    # NEAT hyperparameters tuned for this problem.
    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        converged = result.best_fitness < 0.95
        println("    Two-spirals seed=$seed: fitness=$(round(result.best_fitness, digits=6)), " *
                "nodes=$(length(result.best_genome.nodes))")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  Two-spirals NEAT: $n_success/5 seeds converged (fitness < 0.95)")
    flush(stdout)
    @test n_success >= 3
end
