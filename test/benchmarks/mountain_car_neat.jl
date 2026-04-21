# Mountain Car benchmark using GraphGenome + EpisodicEvaluator.
#
# Canonical sparse-reward reinforcement-learning task: an underpowered car
# sits at the bottom of a valley between two hills and must reach the goal
# at position 0.5 by building momentum through oscillation. Reward = -1
# per step, max 200 steps, so the "trivial" fitness is -200 (never
# reaches the goal). Escape from this local optimum requires the
# controller to accept short-term penalty (moving downhill) in service
# of long-term gain (reaching the uphill goal) — exactly the kind of
# hard-to-greedy problem speciation-protected NEAT is supposed to beat.

const _MC_POS_MIN = -1.2
const _MC_POS_MAX = 0.6
const _MC_VEL_MIN = -0.07
const _MC_VEL_MAX = 0.07
const _MC_GOAL    = 0.5
const _MC_FORCE   = 0.001
const _MC_GRAVITY = 0.0025
const _MC_MAX_STEPS = 200

function _mc_initial_state(rng)
    pos = -0.6 + 0.2 * rand(rng)   # uniform in [-0.6, -0.4]
    return (pos=pos, vel=0.0)
end

function _mc_dynamics(s, a)
    # a ∈ {-1, 0, +1}
    new_vel = s.vel + _MC_FORCE * a - _MC_GRAVITY * cos(3 * s.pos)
    new_vel = clamp(new_vel, _MC_VEL_MIN, _MC_VEL_MAX)
    new_pos = s.pos + new_vel
    if new_pos < _MC_POS_MIN
        return (pos=_MC_POS_MIN, vel=0.0)   # hit left wall, velocity zeroed
    elseif new_pos > _MC_POS_MAX
        return (pos=_MC_POS_MAX, vel=new_vel)
    else
        return (pos=new_pos, vel=new_vel)
    end
end

_mc_reward(s, a, sp) = -1.0
_mc_done(s) = s.pos >= _MC_GOAL
_mc_obs(s) = Float64[s.pos, s.vel]
# 3 output nodes. Because default output activation is :sigmoid (all > 0),
# argmax works cleanly. Map index 1→-1, 2→0, 3→+1.
function _mc_decode(y)
    i = argmax(y)
    return i == 1 ? -1 : (i == 2 ? 0 : 1)
end

@testset "Mountain Car NEAT benchmark (GraphGenome + EpisodicEvaluator)" begin
    evaluator = EpisodicEvaluator(
        2, 3,
        _mc_initial_state, _mc_dynamics,
        _mc_reward, _mc_done,
        _mc_obs, _mc_decode;
        max_steps=_MC_MAX_STEPS, n_episodes=5,
        episode_seed_base=3000,
        allow_recurrent=false,
    )

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=150, generations=100,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=25),
    )

    # Success: ≥2/5 seeds produce a champion that reaches the goal at
    # least sometimes — equivalent to fitness strictly better than the
    # always-fail baseline of -200 per episode (mean reward > -200 means
    # at least one of 5 eval episodes ended early via goal).
    # Fitness = -mean_reward = -(-episode_steps) = +episode_steps on
    # average; a goal reach in episode i gives rewards(i) > -200.
    # So fitness < 200 indicates at least partial success.
    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        # best_fitness < 200.0 ⇒ at least one episode reached the goal.
        converged = result.best_fitness < 200.0
        n_enabled = count(c.enabled for c in values(result.best_genome.connections))
        println("    MountainCar seed=$seed: fitness=$(round(result.best_fitness, digits=2)), " *
                "nodes=$(length(result.best_genome.nodes)), conns=$n_enabled")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  Mountain Car NEAT: $n_success/5 seeds reached goal at least once (fitness < 200)")
    flush(stdout)
    @test n_success >= 2
end
