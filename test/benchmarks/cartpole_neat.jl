# Single-pole cart-pole balancing benchmark using GraphGenome + EpisodicEvaluator.
#
# Barto, Sutton & Anderson (1983) "Neuronlike adaptive elements..." dynamics.
# Canonical NEAT control benchmark — easiest closed-loop task, historically
# solved by NEAT in ~10–30 generations. Demonstrates the EpisodicEvaluator
# wiring on a standard problem before the harder double-pole case.

const _CP_GRAVITY    = 9.8
const _CP_MASSCART   = 1.0
const _CP_MASSPOLE   = 0.1
const _CP_LENGTH     = 0.5    # half the pole length
const _CP_FORCE_MAG  = 10.0
const _CP_TAU        = 0.02   # seconds between state updates
const _CP_X_LIMIT    = 2.4
const _CP_THETA_LIMIT = π / 15.0  # 12 degrees

function _cartpole_initial_state(rng)
    # Standard start: all four components uniform in [-0.05, 0.05].
    return (x         = 0.1 * (rand(rng) - 0.5),
            xdot      = 0.1 * (rand(rng) - 0.5),
            theta     = 0.1 * (rand(rng) - 0.5),
            theta_dot = 0.1 * (rand(rng) - 0.5))
end

function _cartpole_dynamics(s, a)
    force = a * _CP_FORCE_MAG
    total_mass = _CP_MASSCART + _CP_MASSPOLE
    polemass_length = _CP_MASSPOLE * _CP_LENGTH
    costh = cos(s.theta)
    sinth = sin(s.theta)
    temp = (force + polemass_length * s.theta_dot^2 * sinth) / total_mass
    theta_acc = (_CP_GRAVITY * sinth - costh * temp) /
                (_CP_LENGTH * (4.0/3.0 - _CP_MASSPOLE * costh^2 / total_mass))
    x_acc = temp - polemass_length * theta_acc * costh / total_mass
    return (x         = s.x + _CP_TAU * s.xdot,
            xdot      = s.xdot + _CP_TAU * x_acc,
            theta     = s.theta + _CP_TAU * s.theta_dot,
            theta_dot = s.theta_dot + _CP_TAU * theta_acc)
end

_cartpole_reward(s, a, sp) = 1.0
_cartpole_done(s) = abs(s.x) > _CP_X_LIMIT || abs(s.theta) > _CP_THETA_LIMIT
_cartpole_obs(s) = Float64[s.x, s.xdot, s.theta, s.theta_dot]
# Output node activation defaults to :sigmoid (values ∈ (0,1)); threshold at
# the sigmoid midpoint 0.5 so both actions are reachable from the raw output.
_cartpole_decode(y) = y[1] > 0.5 ? 1 : -1

@testset "Cart-pole NEAT benchmark (GraphGenome + EpisodicEvaluator)" begin
    # 5 episodes averaged per fitness evaluation.  max_steps=200 is the
    # canonical Barto-Sutton-Anderson horizon (4 seconds at 50 Hz).
    evaluator = EpisodicEvaluator(
        4, 1,
        _cartpole_initial_state,
        _cartpole_dynamics,
        _cartpole_reward,
        _cartpole_done,
        _cartpole_obs,
        _cartpole_decode;
        max_steps=200,
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
