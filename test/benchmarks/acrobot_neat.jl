# Acrobot swing-up benchmark using GraphGenome + EpisodicEvaluator.
#
# Sutton (1996) "Generalization in Reinforcement Learning: Successful
# Examples using Sparse Coarse Coding."  Two-link under-actuated
# pendulum — a torque motor at the joint between the two links swings
# the system up so the tip rises above the pivot.  Canonical follow-on
# to cart-pole that RL benchmarks use to separate easy controllers
# from ones that must learn a swing-up strategy.
#
# State (4D): θ1 (first link angle from straight down), θ2 (second link
# angle relative to first), ω1, ω2 (angular velocities).  Action ∈
# {-1, 0, +1} applied as joint torque.  Termination: tip height above
# one link length (−cos(θ1) − cos(θ1+θ2) > 1.0), or 500 steps.  Reward
# = −1 per step, so shorter episodes = higher reward.
#
# Dynamics integrated with RK4 at dt=0.2 (matching Gym's Acrobot-v1).

const _AC_M1 = 1.0    # link 1 mass
const _AC_M2 = 1.0    # link 2 mass
const _AC_L1 = 1.0    # link 1 length
const _AC_LC1 = 0.5   # link 1 center of mass
const _AC_LC2 = 0.5   # link 2 center of mass
const _AC_I1 = 1.0    # link 1 moment of inertia
const _AC_I2 = 1.0    # link 2 moment of inertia
const _AC_G  = 9.8
const _AC_DT = 0.2
const _AC_MAX_VEL_1 = 4π
const _AC_MAX_VEL_2 = 9π
const _AC_MAX_STEPS = 500

# ds/dt given state and torque action.
function _acrobot_ddt(theta1, theta2, omega1, omega2, a)
    m1, m2 = _AC_M1, _AC_M2
    l1, lc1, lc2 = _AC_L1, _AC_LC1, _AC_LC2
    I1, I2 = _AC_I1, _AC_I2
    g = _AC_G

    d1 = m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(theta2)) + I1 + I2
    d2 = m2*(lc2^2 + l1*lc2*cos(theta2)) + I2
    phi2 = m2*lc2*g*cos(theta1 + theta2 - π/2)
    phi1 = (-m2*l1*lc2*omega2^2*sin(theta2)
            - 2*m2*l1*lc2*omega2*omega1*sin(theta2)
            + (m1*lc1 + m2*l1)*g*cos(theta1 - π/2) + phi2)

    ddtheta2 = (a + d2/d1*phi1 - m2*l1*lc2*omega1^2*sin(theta2) - phi2) /
               (m2*lc2^2 + I2 - d2^2/d1)
    ddtheta1 = -(d2*ddtheta2 + phi1) / d1

    return (omega1, omega2, ddtheta1, ddtheta2)
end

# One RK4 step in (theta1, theta2, omega1, omega2) space.
function _acrobot_rk4(theta1, theta2, omega1, omega2, a, dt)
    k1 = _acrobot_ddt(theta1, theta2, omega1, omega2, a)
    k2 = _acrobot_ddt(theta1 + 0.5dt*k1[1], theta2 + 0.5dt*k1[2],
                      omega1 + 0.5dt*k1[3], omega2 + 0.5dt*k1[4], a)
    k3 = _acrobot_ddt(theta1 + 0.5dt*k2[1], theta2 + 0.5dt*k2[2],
                      omega1 + 0.5dt*k2[3], omega2 + 0.5dt*k2[4], a)
    k4 = _acrobot_ddt(theta1 + dt*k3[1], theta2 + dt*k3[2],
                      omega1 + dt*k3[3], omega2 + dt*k3[4], a)

    th1n = theta1 + dt*(k1[1] + 2k2[1] + 2k3[1] + k4[1])/6
    th2n = theta2 + dt*(k1[2] + 2k2[2] + 2k3[2] + k4[2])/6
    w1n  = omega1 + dt*(k1[3] + 2k2[3] + 2k3[3] + k4[3])/6
    w2n  = omega2 + dt*(k1[4] + 2k2[4] + 2k3[4] + k4[4])/6

    # Wrap angles and clamp velocities per Gym convention.
    th1n = mod(th1n + π, 2π) - π
    th2n = mod(th2n + π, 2π) - π
    w1n  = clamp(w1n, -_AC_MAX_VEL_1, _AC_MAX_VEL_1)
    w2n  = clamp(w2n, -_AC_MAX_VEL_2, _AC_MAX_VEL_2)

    return (th1n, th2n, w1n, w2n)
end

function _acrobot_initial_state(rng)
    # Canonical Gym init: uniform in [-0.1, 0.1] for all four components.
    return (theta1 = 0.2 * (rand(rng) - 0.5),
            theta2 = 0.2 * (rand(rng) - 0.5),
            omega1 = 0.2 * (rand(rng) - 0.5),
            omega2 = 0.2 * (rand(rng) - 0.5))
end

function _acrobot_dynamics(s, a)
    th1, th2, w1, w2 = _acrobot_rk4(s.theta1, s.theta2, s.omega1, s.omega2,
                                    Float64(a), _AC_DT)
    return (theta1=th1, theta2=th2, omega1=w1, omega2=w2)
end

_acrobot_reward(s, a, sp) = -1.0
# Tip height above pivot: h = -cos(θ1) - cos(θ1 + θ2).  Terminate when h > 1.
_acrobot_done(s) = -cos(s.theta1) - cos(s.theta1 + s.theta2) > 1.0
_acrobot_obs(s) = Float64[s.theta1, s.theta2, s.omega1, s.omega2]
# 3 outputs, argmax → action ∈ {-1, 0, +1}.  Same decoder as Mountain Car.
function _acrobot_decode(y)
    i = argmax(y)
    return i == 1 ? -1 : (i == 2 ? 0 : 1)
end

@testset "Acrobot swing-up NEAT benchmark (GraphGenome + EpisodicEvaluator)" begin
    evaluator = EpisodicEvaluator(
        4, 3,
        _acrobot_initial_state, _acrobot_dynamics,
        _acrobot_reward, _acrobot_done,
        _acrobot_obs, _acrobot_decode;
        max_steps=_AC_MAX_STEPS, n_episodes=5,
        episode_seed_base=5000,
        allow_recurrent=false,
    )

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=150, generations=150,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=25),
    )

    # fitness = -mean_reward.  Baseline (controller never swings up):
    # 500 steps × −1 reward = −500 per episode, fitness = +500.
    # NEAT with speciation solves this reliably in under 100 steps
    # average once the swing-up strategy is discovered, so the gate
    # distinguishes "found a swing-up policy" (fitness < 150) from
    # "no swing-up" (fitness = 500).  Initial calibration at fitness
    # < 400 was far too loose — all 5 seeds hit ~65 during development.
    successes = map(1:5) do seed
        reset_innovation_counter!()
        problem = GPProblem(evaluator, GraphGenome; seed=seed)
        result = solve(problem, algorithm; verbose=false)
        converged = result.best_fitness < 150.0
        n_enabled = count(c.enabled for c in values(result.best_genome.connections))
        println("    Acrobot seed=$seed: fitness=$(round(result.best_fitness, digits=2)), " *
                "nodes=$(length(result.best_genome.nodes)), conns=$n_enabled")
        flush(stdout)
        converged
    end

    n_success = count(successes)
    println("  Acrobot NEAT: $n_success/5 seeds swung up in <150 mean steps (fitness < 150)")
    flush(stdout)
    @test n_success >= 3
end
