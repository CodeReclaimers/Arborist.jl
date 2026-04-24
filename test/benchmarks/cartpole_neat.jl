# Single-pole cart-pole balancing benchmark using GraphGenome + EpisodicEvaluator.
#
# Barto, Sutton & Anderson (1983) dynamics sourced from
# `Arborist.Benchmarks.cartpole()`. Canonical NEAT control benchmark.

@testset "Cart-pole NEAT benchmark (GraphGenome + EpisodicEvaluator)" begin
    cp = Arborist.Benchmarks.cartpole()

    # 5 episodes averaged per fitness evaluation.
    evaluator = EpisodicEvaluator(
        cp.n_states, cp.n_actions,
        cp.initial_state, cp.dynamics,
        cp.reward, cp.done, cp.observe, cp.decode_action;
        max_steps=cp.max_steps,
        n_episodes=5,
        episode_seed_base=1000,
        allow_recurrent=false,   # feedforward suffices with full state observation
    )

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=100, generations=60,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=20),
    )

    # Success: >=4/5 seeds produce a champion that balances an average of
    # >=195 steps across the 5 evaluation episodes (fitness <= -195).
    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        converged = result.best_fitness <= -195.0
        println("    CartPole seed=$seed: fitness=$(round(result.best_fitness, digits=2)), " *
                "nodes=$(length(result.best_genome.nodes)), " *
                "conns=$(count(c.enabled for c in values(result.best_genome.connections)))")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  Cart-pole NEAT: $n_success/5 seeds balanced >=195 steps (fitness <= -195)")
    flush(stdout)
    @test n_success >= 4
end
