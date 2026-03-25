# XOR benchmark using GraphGenome (NEAT-style).
# This replicates the canonical NEAT benchmark (Stanley & Miikkulainen, 2002).

@testset "XOR NEAT benchmark (GraphGenome)" begin
    reset_innovation_counter!()

    input_data = Float64[0 0 1 1; 0 1 0 1]
    output_data = Float64[0 1 1 0]
    evaluator = GraphEvaluator(input_data, output_data)

    algorithm = GeneticProgramming(
        pop_size=150,
        generations=150,
        mutation_rate=0.5,
        crossover_rate=0.3,
        elitism=2,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=15)
    )

    # Majority convergence: 4/5 seeds should reach fitness < 0.01.
    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        converged = result.best_fitness < 0.01
        println("    XOR seed=$seed: fitness=$(round(result.best_fitness, digits=6)), nodes=$(length(result.best_genome.nodes))")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  XOR NEAT: $n_success/5 seeds converged (fitness < 0.01)")
    flush(stdout)
    @test n_success >= 4
end
