# Control Tasks: Cart-Pole with EpisodicEvaluator

This tutorial solves the Barto/Sutton/Anderson cart-pole balancing
problem end-to-end using `GraphGenome` + `EpisodicEvaluator`. It is
the canonical closed-loop control benchmark, historically solved by
NEAT in 10–30 generations.

## EpisodicEvaluator overview

`EpisodicEvaluator` is a *declarative* closed-loop evaluator: you
hand it six callables describing the environment, and it runs one or
more rollouts per fitness evaluation, averaging the returned reward.
The callables are:

| Callable | Signature | Role |
|---|---|---|
| `initial_state` | `(rng) -> s₀` | Sample a starting state using the thread-local RNG |
| `dynamics` | `(s, a) -> s'` | One step of the environment transition |
| `reward` | `(s, a, s') -> Float64` | Per-step scalar reward |
| `done` | `(s) -> Bool` | Terminate episode if true |
| `observe` | `(s) -> Vector{Float64}` | Network inputs for this state |
| `decode_action` | `(y::Vector{Float64}) -> a` | Map network output to action |

Arborist convention: **lower fitness is better**. `EpisodicEvaluator`
returns `-mean_reward` so higher cumulative reward ⇒ lower fitness.

## Dynamics

Barto-Sutton-Anderson constants:

```julia
using Arborist

const _CP_GRAVITY     = 9.8
const _CP_MASSCART    = 1.0
const _CP_MASSPOLE    = 0.1
const _CP_LENGTH      = 0.5         # half-length
const _CP_FORCE_MAG   = 10.0
const _CP_TAU         = 0.02        # seconds per step
const _CP_X_LIMIT     = 2.4
const _CP_THETA_LIMIT = π / 15.0    # ≈12°

function _initial(rng)
    (x = 0.1*(rand(rng) - 0.5),
     xdot = 0.1*(rand(rng) - 0.5),
     theta = 0.1*(rand(rng) - 0.5),
     theta_dot = 0.1*(rand(rng) - 0.5))
end

function _dynamics(s, a)
    force = a * _CP_FORCE_MAG
    total_mass = _CP_MASSCART + _CP_MASSPOLE
    polemass_length = _CP_MASSPOLE * _CP_LENGTH
    costh, sinth = cos(s.theta), sin(s.theta)
    temp = (force + polemass_length * s.theta_dot^2 * sinth) / total_mass
    theta_acc = (_CP_GRAVITY * sinth - costh * temp) /
                (_CP_LENGTH * (4/3 - _CP_MASSPOLE * costh^2 / total_mass))
    x_acc = temp - polemass_length * theta_acc * costh / total_mass
    return (x = s.x + _CP_TAU * s.xdot,
            xdot = s.xdot + _CP_TAU * x_acc,
            theta = s.theta + _CP_TAU * s.theta_dot,
            theta_dot = s.theta_dot + _CP_TAU * theta_acc)
end

_reward(s, a, sp) = 1.0
_done(s) = abs(s.x) > _CP_X_LIMIT || abs(s.theta) > _CP_THETA_LIMIT
_obs(s)  = Float64[s.x, s.xdot, s.theta, s.theta_dot]
```

## Decoding the network output

The output node's default activation is sigmoid, so raw outputs live in
`(0, 1)`. Threshold at the sigmoid midpoint so both actions are
reachable:

```julia
_decode(y) = y[1] > 0.5 ? 1 : -1
```

A common early mistake is thresholding at `0`, which makes the first
action unreachable under sigmoid activation. The same story holds
for `tanh` (threshold at 0, since tanh outputs span (−1, 1)) and
`identity` (user-defined range).

## Assemble the evaluator

```julia
evaluator = EpisodicEvaluator(
    4, 1,                  # n_inputs, n_outputs
    _initial, _dynamics, _reward, _done, _obs, _decode;
    max_steps        = 200,
    n_episodes       = 5,  # rollouts averaged per fitness call
    episode_seed_base = 1000,
    allow_recurrent  = false,
)
```

`episode_seed_base` keeps the 5 rollouts deterministic within a run —
episode `k` uses seed `episode_seed_base + k`.

## Solve

```julia
reset_innovation_counter!()
ops       = neat_defaults()
algorithm = GeneticProgramming(
    pop_size       = 100,
    generations    = 60,
    mutation_rate  = 0.5,
    crossover_rate = 0.3,
    elitism        = 2,
    mutation_ops   = ops.mutation_ops,
    crossover_ops  = ops.crossover_ops,
    speciation     = ThresholdSpeciation(
        threshold        = 3.0,
        min_species_size = 2,
        stagnation_limit = 20,
    ),
)

problem = GPProblem(evaluator, GraphGenome; seed=42)
result  = solve(problem, algorithm; verbose=true)
```

Success is `result.best_fitness ≤ -195`, i.e. the champion balances
an average of ≥ 195 steps across the 5 evaluation episodes. Across
5 seeds, at least 4 usually converge.

## Harder control tasks

The same pattern generalizes:

- `test/benchmarks/double_pole_neat.jl` — Markovian two-pole cart.
- `test/benchmarks/mountain_car_neat.jl` — sparse-reward mountain car.
- `test/benchmarks/acrobot_neat.jl` — under-actuated acrobot swing-up.

For non-Markovian versions (e.g. double-pole without velocity
observations), add `allow_recurrent=true` with `relaxation_passes=N`
so the network can carry state across steps.
