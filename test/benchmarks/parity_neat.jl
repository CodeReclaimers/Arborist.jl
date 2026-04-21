# N-bit parity benchmark using GraphGenome (NEAT-style).
#
# Structural-search scaling of XOR: N=3 and N=5. Canonical in the NEAT
# literature because it exposes whether speciation protects the structural
# innovations needed to grow a network deep enough to solve the task.
#
# Target: even-parity (output = 1 when an odd number of inputs are 1,
# matching XOR convention for N=2). Encoded as Float64 for GraphEvaluator.

@testset "3-bit parity NEAT benchmark (GraphGenome)" begin
    n_bits = 3
    n_cases = 2^n_bits

    # input_data is n_bits × n_cases (matches XOR benchmark layout).
    input_data = zeros(Float64, n_bits, n_cases)
    output_data = zeros(Float64, 1, n_cases)
    for bits in 0:(n_cases - 1)
        n_true = 0
        for i in 1:n_bits
            val = (bits >> (i - 1)) & 1
            input_data[i, bits + 1] = Float64(val)
            n_true += val
        end
        output_data[1, bits + 1] = Float64(n_true % 2)
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

    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        converged = result.best_fitness < 0.05
        println("    Parity-3 seed=$seed: fitness=$(round(result.best_fitness, digits=6)), " *
                "nodes=$(length(result.best_genome.nodes))")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  Parity-3 NEAT: $n_success/5 seeds converged (fitness < 0.05)")
    flush(stdout)
    @test n_success >= 4
end

@testset "5-bit parity NEAT benchmark (GraphGenome)" begin
    n_bits = 5
    n_cases = 2^n_bits

    input_data = zeros(Float64, n_bits, n_cases)
    output_data = zeros(Float64, 1, n_cases)
    for bits in 0:(n_cases - 1)
        n_true = 0
        for i in 1:n_bits
            val = (bits >> (i - 1)) & 1
            input_data[i, bits + 1] = Float64(val)
            n_true += val
        end
        output_data[1, bits + 1] = Float64(n_true % 2)
    end

    evaluator = GraphEvaluator(input_data, output_data)
    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=300,
        generations=800,
        mutation_rate=0.5,
        crossover_rate=0.3,
        elitism=2,
        mutation_ops=ops.mutation_ops,
        crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=25)
    )

    # 5-bit parity is a genuinely hard structural-search problem for NEAT.
    # Threshold is intentionally lenient: at least 1/5 seeds under 0.15 MSE
    # (trivial baseline is 0.25 from predicting the mean).
    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        converged = result.best_fitness < 0.15
        println("    Parity-5 seed=$seed: fitness=$(round(result.best_fitness, digits=6)), " *
                "nodes=$(length(result.best_genome.nodes))")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  Parity-5 NEAT: $n_success/5 seeds converged (fitness < 0.15)")
    flush(stdout)
    @test n_success >= 1
end
