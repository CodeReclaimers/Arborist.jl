# NSGA-II GraphGenome benchmark on two-spirals.
#
# Intended as a structurally-different NSGA-II solve demo to complement
# `xor_nsga2_neat.jl` (XOR infrastructure-only). Empirically, default NEAT
# hyperparameters + vanilla NSGA-II + parsimony objective exhibit the same
# zero-connection-attractor pathology on two-spirals as they do on XOR: the
# complexity=0 network (MSE=1.0 on ±1 targets) Pareto-dominates all
# higher-complexity networks that haven't yet crossed the fitness plateau.
# Best measured fitness on the front hovers at 0.95–0.98 at the test-suite
# budget, so this benchmark is — like `xor_nsga2_neat.jl` — infrastructure
# coverage with a weak progress check, not a solve demonstration.
#
# Future work: add a variant with NEAT speciation/niching specifically
# tuned to preserve structural innovations through the MSE=1.0 plateau,
# or with fitness sharing as Stanley & Miikkulainen (2002) prescribe.

@testset "Two-spirals NSGA-II NEAT benchmark (GraphGenome Pareto solve)" begin
    reset_innovation_counter!()

    n_per_spiral = 97
    n_total = 2 * n_per_spiral

    input_data = zeros(Float64, 2, n_total)
    output_data = zeros(Float64, 1, n_total)
    for k in 0:(n_per_spiral - 1)
        angle = k * π / 16.0
        radius = 6.5 * (104 - k) / 104.0
        x = radius * sin(angle)
        y = radius * cos(angle)
        input_data[1, 2k + 1] = x
        input_data[2, 2k + 1] = y
        output_data[1, 2k + 1] = 1.0
        input_data[1, 2k + 2] = -x
        input_data[2, 2k + 2] = -y
        output_data[1, 2k + 2] = -1.0
    end

    evaluator = ParsimonyEvaluator(GraphEvaluator(input_data, output_data))

    ops = neat_defaults()
    algorithm = NSGAII(
        pop_size=150, generations=300,
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

    for f in result.pareto_fitnesses
        @test all(isfinite, f)
    end

    @test any(h -> h > 0.0, result.hypervolume_history)

    # Weak progress-over-baseline assertion: at least one Pareto member
    # must beat the zero-connection trivial baseline (MSE=1.0 on ±1
    # targets). In practice default NEAT + vanilla NSGA-II on two-spirals
    # exhibits the same zero-connection-attractor pathology as the XOR
    # NSGA-II benchmark — the front is pulled toward complexity=0 before
    # enough structural innovation accumulates to cross the fitness plateau.
    # Measured best-fit is typically 0.95–0.98 at this budget. Asserting
    # < 1.0 catches regressions without depending on seed-specific luck.
    front_fitness = [f[1] for f in result.pareto_fitnesses]
    front_complexity = [f[2] for f in result.pareto_fitnesses]
    best_fit_on_front = minimum(front_fitness)
    @test best_fit_on_front < 1.0

    println("  Two-spirals NSGA-II Pareto front ($(length(result.pareto_front)) genomes):")
    println("    fitness    : min=$(round(best_fit_on_front, digits=4))  " *
            "max=$(round(maximum(front_fitness), digits=4))")
    println("    complexity : min=$(Int(minimum(front_complexity)))  " *
            "max=$(Int(maximum(front_complexity)))")
    println("    hypervolume[end] = $(round(result.hypervolume_history[end], digits=4))")
    flush(stdout)
end
