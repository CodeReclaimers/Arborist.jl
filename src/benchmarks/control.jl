# Closed-loop control benchmark generators.
#
# All control benchmarks return the full set of callables needed by
# `Arborist.EpisodicEvaluator`: `initial_state`, `dynamics`, `reward`,
# `done`, `observe`, `decode_action`. Plus metadata (`n_states`,
# `n_actions`, `max_steps`, `name`, `target_expr`).
#
# The `decode_action` callbacks assume the default `EpisodicEvaluator`
# output-node activation `:sigmoid` with output range (0, 1). Callers
# using `:tanh` or `:identity` output activations should supply their
# own decoder.

# =============================================================================
# Cart-pole (Barto, Sutton & Anderson 1983)
# =============================================================================

const _CP_GRAVITY     = 9.8
const _CP_MASSCART    = 1.0
const _CP_MASSPOLE    = 0.1
const _CP_LENGTH      = 0.5     # half the pole length
const _CP_FORCE_MAG   = 10.0
const _CP_TAU         = 0.02    # seconds between state updates
const _CP_X_LIMIT     = 2.4
const _CP_THETA_LIMIT = π / 15.0  # 12 degrees

function _cp_initial_state(rng)
    return (x         = 0.1 * (rand(rng) - 0.5),
            xdot      = 0.1 * (rand(rng) - 0.5),
            theta     = 0.1 * (rand(rng) - 0.5),
            theta_dot = 0.1 * (rand(rng) - 0.5))
end

function _cp_dynamics(s, a)
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

_cp_reward(s, a, sp) = 1.0
_cp_done(s) = abs(s.x) > _CP_X_LIMIT || abs(s.theta) > _CP_THETA_LIMIT
_cp_obs(s) = Float64[s.x, s.xdot, s.theta, s.theta_dot]
# Default sigmoid output: threshold at 0.5 so both actions reachable.
_cp_decode(y) = y[1] > 0.5 ? 1 : -1

"""
    cartpole() -> NamedTuple

Single-pole cart-pole balancing (Barto, Sutton & Anderson 1983).
4 continuous state variables (x, xdot, theta, theta_dot), binary
action (±1). Success criterion: balance for 200 steps (= 4 seconds
at 50 Hz). Reward = 1 per step until pole falls or cart leaves bounds.

Returns `(; initial_state, dynamics, reward, done, observe,
decode_action, n_states=4, n_actions=1, max_steps=200, name,
target_expr)`.
"""
function cartpole()
    return (;
        initial_state = _cp_initial_state,
        dynamics      = _cp_dynamics,
        reward        = _cp_reward,
        done          = _cp_done,
        observe       = _cp_obs,
        decode_action = _cp_decode,
        n_states = 4, n_actions = 1,
        max_steps = 200,
        name = "Cart-Pole",
        target_expr = "balance pole for 200 steps")
end

# =============================================================================
# Mountain Car (Moore 1990; Sutton 1996)
# =============================================================================

const _MC_POS_MIN = -1.2
const _MC_POS_MAX = 0.6
const _MC_VEL_MIN = -0.07
const _MC_VEL_MAX = 0.07
const _MC_GOAL    = 0.5
const _MC_FORCE   = 0.001
const _MC_GRAVITY = 0.0025
const _MC_MAX_STEPS = 200

function _mc_initial_state(rng)
    pos = -0.6 + 0.2 * rand(rng)
    return (pos=pos, vel=0.0)
end

function _mc_dynamics(s, a)
    new_vel = s.vel + _MC_FORCE * a - _MC_GRAVITY * cos(3 * s.pos)
    new_vel = clamp(new_vel, _MC_VEL_MIN, _MC_VEL_MAX)
    new_pos = s.pos + new_vel
    if new_pos < _MC_POS_MIN
        return (pos=_MC_POS_MIN, vel=0.0)
    elseif new_pos > _MC_POS_MAX
        return (pos=_MC_POS_MAX, vel=new_vel)
    else
        return (pos=new_pos, vel=new_vel)
    end
end

_mc_reward(s, a, sp) = -1.0
_mc_done(s) = s.pos >= _MC_GOAL
_mc_obs(s) = Float64[s.pos, s.vel]
# 3 outputs → categorical; argmax indexes action from {-1, 0, +1}.
function _mc_decode(y)
    i = argmax(y)
    return i == 1 ? -1 : (i == 2 ? 0 : 1)
end

"""
    mountain_car() -> NamedTuple

Sparse-reward under-powered car escape task (Moore 1990, Sutton 1996).
2 state variables (pos, vel), 3 discrete actions (left / no-op / right).
Reward = -1 per step; must reach `pos >= 0.5`. Baseline fitness (never
reaching goal) is 200.
"""
function mountain_car()
    return (;
        initial_state = _mc_initial_state,
        dynamics      = _mc_dynamics,
        reward        = _mc_reward,
        done          = _mc_done,
        observe       = _mc_obs,
        decode_action = _mc_decode,
        n_states = 2, n_actions = 3,
        max_steps = _MC_MAX_STEPS,
        name = "Mountain Car",
        target_expr = "reach goal position 0.5 within 200 steps")
end

# =============================================================================
# Acrobot (Sutton 1996)
# =============================================================================

const _AC_M1 = 1.0
const _AC_M2 = 1.0
const _AC_L1 = 1.0
const _AC_LC1 = 0.5
const _AC_LC2 = 0.5
const _AC_I1 = 1.0
const _AC_I2 = 1.0
const _AC_G = 9.8
const _AC_DT = 0.2
const _AC_MAX_STEPS = 500
const _AC_GOAL_HEIGHT = 1.0   # y-coordinate of tip (tip_y > 1 solves)

function _ac_initial_state(rng)
    return (theta1 = 0.1 * (rand(rng) - 0.5),
            theta2 = 0.1 * (rand(rng) - 0.5),
            dtheta1 = 0.0, dtheta2 = 0.0)
end

function _ac_dsdt(s, torque)
    d1 = _AC_M1 * _AC_LC1^2 + _AC_M2 * (_AC_L1^2 + _AC_LC2^2 + 2 * _AC_L1 * _AC_LC2 * cos(s.theta2)) + _AC_I1 + _AC_I2
    d2 = _AC_M2 * (_AC_LC2^2 + _AC_L1 * _AC_LC2 * cos(s.theta2)) + _AC_I2
    phi2 = _AC_M2 * _AC_LC2 * _AC_G * cos(s.theta1 + s.theta2 - π/2)
    phi1 = -_AC_M2 * _AC_L1 * _AC_LC2 * s.dtheta2^2 * sin(s.theta2) -
            2 * _AC_M2 * _AC_L1 * _AC_LC2 * s.dtheta2 * s.dtheta1 * sin(s.theta2) +
            (_AC_M1 * _AC_LC1 + _AC_M2 * _AC_L1) * _AC_G * cos(s.theta1 - π/2) + phi2
    ddtheta2 = (torque + d2 / d1 * phi1 - _AC_M2 * _AC_L1 * _AC_LC2 * s.dtheta1^2 * sin(s.theta2) - phi2) /
               (_AC_M2 * _AC_LC2^2 + _AC_I2 - d2^2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return (ddtheta1 = ddtheta1, ddtheta2 = ddtheta2)
end

function _ac_dynamics(s, a)
    torque = float(a)
    # RK4 over dt.
    h = _AC_DT
    state_vec(state) = (state.theta1, state.theta2, state.dtheta1, state.dtheta2)
    as_nt(v) = (theta1=v[1], theta2=v[2], dtheta1=v[3], dtheta2=v[4])

    v1 = state_vec(s)
    a1 = _ac_dsdt(s, torque)
    k1 = (v1[3], v1[4], a1.ddtheta1, a1.ddtheta2)

    v2 = v1 .+ (h/2) .* k1
    s2 = as_nt(v2)
    a2 = _ac_dsdt(s2, torque)
    k2 = (v2[3], v2[4], a2.ddtheta1, a2.ddtheta2)

    v3 = v1 .+ (h/2) .* k2
    s3 = as_nt(v3)
    a3 = _ac_dsdt(s3, torque)
    k3 = (v3[3], v3[4], a3.ddtheta1, a3.ddtheta2)

    v4 = v1 .+ h .* k3
    s4 = as_nt(v4)
    a4 = _ac_dsdt(s4, torque)
    k4 = (v4[3], v4[4], a4.ddtheta1, a4.ddtheta2)

    vnew = v1 .+ (h/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    return as_nt(vnew)
end

_ac_reward(s, a, sp) = -1.0
_ac_done(s) = (-cos(s.theta1) - cos(s.theta2 + s.theta1)) > _AC_GOAL_HEIGHT
_ac_obs(s) = Float64[cos(s.theta1), sin(s.theta1), cos(s.theta2), sin(s.theta2),
                     s.dtheta1, s.dtheta2]
function _ac_decode(y)
    i = argmax(y)
    return i == 1 ? -1.0 : (i == 2 ? 0.0 : 1.0)
end

"""
    acrobot() -> NamedTuple

Acrobot swing-up (Sutton 1996). Two-link under-actuated pendulum; torque
applied only at the joint. 6-dim observation (cos/sin of both angles +
angular velocities), 3 discrete torque actions. RK4 integration at
dt=0.2. Goal: tip height > 1.0 within 500 steps.
"""
function acrobot()
    return (;
        initial_state = _ac_initial_state,
        dynamics      = _ac_dynamics,
        reward        = _ac_reward,
        done          = _ac_done,
        observe       = _ac_obs,
        decode_action = _ac_decode,
        n_states = 6, n_actions = 3,
        max_steps = _AC_MAX_STEPS,
        name = "Acrobot",
        target_expr = "swing tip above height 1.0 within 500 steps")
end

# =============================================================================
# Double-pole balancing (markovian variant)
# =============================================================================

const _DP_GRAVITY    = -9.8
const _DP_MASSCART   = 1.0
const _DP_M1         = 0.1
const _DP_M2         = 0.01
const _DP_L1         = 0.5   # half
const _DP_L2         = 0.05  # half (shorter pole)
const _DP_FORCE_MAG  = 10.0
const _DP_TAU        = 0.01
const _DP_X_LIMIT    = 2.4
const _DP_THETA_LIMIT = π / 5.0  # 36 degrees
const _DP_MAX_STEPS  = 1000

function _dp_initial_state(rng)
    return (x = 0.1 * (rand(rng) - 0.5),
            xdot = 0.0,
            theta1 = 0.07,  # 4 degrees, canonical NEAT start
            theta1_dot = 0.0,
            theta2 = 0.0,
            theta2_dot = 0.0)
end

function _dp_forces(s, force)
    # Compute accelerations given force on cart and pole angles.
    function pole_force(theta, theta_dot, mi, li)
        mli2 = mi * li
        ml2_theta_dot2_sin = mli2 * theta_dot^2 * sin(theta)
        return (mli2 * cos(theta) * _DP_GRAVITY * cos(theta) / li,
                mli2 * cos(theta),
                ml2_theta_dot2_sin)
    end
    # Use canonical effective-mass formulation.
    # (This is a simplified version; the full NEAT double-pole dynamics
    #  are in Gomez & Miikkulainen 1999.)
    meff1 = _DP_M1 * (1 - 0.75 * cos(s.theta1)^2)
    meff2 = _DP_M2 * (1 - 0.75 * cos(s.theta2)^2)
    f_pole1 = _DP_M1 * _DP_L1 * s.theta1_dot^2 * sin(s.theta1) +
              0.75 * _DP_M1 * cos(s.theta1) * (_DP_GRAVITY * sin(s.theta1))
    f_pole2 = _DP_M2 * _DP_L2 * s.theta2_dot^2 * sin(s.theta2) +
              0.75 * _DP_M2 * cos(s.theta2) * (_DP_GRAVITY * sin(s.theta2))
    x_acc = (force + f_pole1 + f_pole2) / (_DP_MASSCART + meff1 + meff2)
    theta1_acc = -0.75 * (x_acc * cos(s.theta1) + _DP_GRAVITY * sin(s.theta1)) / _DP_L1
    theta2_acc = -0.75 * (x_acc * cos(s.theta2) + _DP_GRAVITY * sin(s.theta2)) / _DP_L2
    return (x_acc, theta1_acc, theta2_acc)
end

function _dp_dynamics(s, a)
    force = a * _DP_FORCE_MAG
    x_acc, th1_acc, th2_acc = _dp_forces(s, force)
    return (x          = s.x + _DP_TAU * s.xdot,
            xdot       = s.xdot + _DP_TAU * x_acc,
            theta1     = s.theta1 + _DP_TAU * s.theta1_dot,
            theta1_dot = s.theta1_dot + _DP_TAU * th1_acc,
            theta2     = s.theta2 + _DP_TAU * s.theta2_dot,
            theta2_dot = s.theta2_dot + _DP_TAU * th2_acc)
end

_dp_reward(s, a, sp) = 1.0
_dp_done(s) = abs(s.x) > _DP_X_LIMIT ||
              abs(s.theta1) > _DP_THETA_LIMIT ||
              abs(s.theta2) > _DP_THETA_LIMIT

_dp_obs_markov(s) = Float64[s.x, s.xdot, s.theta1, s.theta1_dot,
                             s.theta2, s.theta2_dot]
_dp_obs_no_vel(s) = Float64[s.x, s.theta1, s.theta2]
_dp_decode(y) = y[1] > 0.5 ? 1 : -1

"""
    double_pole(; markovian=true) -> NamedTuple

Double-pole balancing, Markovian variant (Gomez & Miikkulainen 1999).
When `markovian=true`, observation includes all 6 state components
(including velocities) — a relatively easy NEAT benchmark. When
`markovian=false`, velocities are withheld — the classic hard non-
Markovian variant requiring recurrence to estimate dt.
"""
function double_pole(; markovian::Bool=true)
    obs = markovian ? _dp_obs_markov : _dp_obs_no_vel
    n_states = markovian ? 6 : 3
    name = markovian ? "Double-pole (Markovian)" : "Double-pole (no-velocity)"
    return (;
        initial_state = _dp_initial_state,
        dynamics      = _dp_dynamics,
        reward        = _dp_reward,
        done          = _dp_done,
        observe       = obs,
        decode_action = _dp_decode,
        n_states = n_states, n_actions = 1,
        max_steps = _DP_MAX_STEPS,
        name = name,
        target_expr = "balance both poles for 1000 steps")
end
