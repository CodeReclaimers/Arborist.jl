#!/usr/bin/env julia
# bench/benchmarks.jl
#
# Entry points for benchstone (https://github.com/.../benchstone) integration.
# Invoked as:
#   julia --project=. --threads=8 bench/benchmarks.jl \
#       --entry=<name> --config=<config_path> --output=<output_path>
#
# Each entry point reads a JSON InvocationConfig from --config, runs the
# existing benchmark logic (wrapped, not reimplemented), and writes a
# JSON ProjectResult to --output. Stdout is progress-only and not parsed.
#
# Wrapping discipline (CLAUDE.md): the entry points call into the project's
# existing benchmark surfaces without tuning them. The editable surface
# remains examples/nsga2_tuning_config.jl; this file is protocol plumbing.

using Arborist
using Random
using Statistics
using DynamicExpressions
using TOML

# Include existing benchmark surfaces. Both files guard their top-level
# main() behind `abspath(PROGRAM_FILE) == @__FILE__` or have no top-level
# side-effects, so include() is side-effect-free here.
include(joinpath(@__DIR__, "..", "examples", "bin_packing.jl"))
include(joinpath(@__DIR__, "..", "examples", "nsga2_tuning_config.jl"))

# ---------------------------------------------------------------------------
# Minimal JSON I/O
#
# The benchstone contract is JSON. Julia has no stdlib JSON package and the
# benchmark environment would like to avoid an added runtime dependency on
# Arborist's Project.toml, so the narrow shapes we exchange are parsed and
# emitted here directly. Input config is a flat ASCII object of primitives
# plus an optional null; output is structured but built from Dicts we own.
# ---------------------------------------------------------------------------

function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'
            write(buf, "\\\"")
        elseif c == '\\'
            write(buf, "\\\\")
        elseif c == '\n'
            write(buf, "\\n")
        elseif c == '\r'
            write(buf, "\\r")
        elseif c == '\t'
            write(buf, "\\t")
        elseif UInt32(c) < 0x20
            write(buf, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else
            write(buf, c)
        end
    end
    return String(take!(buf))
end

function _json_write(io::IO, v)
    if v === nothing
        print(io, "null")
    elseif v isa Bool
        print(io, v ? "true" : "false")
    elseif v isa Integer
        print(io, v)
    elseif v isa AbstractFloat
        isfinite(v) || error("non-finite JSON value: $v")
        print(io, v)
    elseif v isa AbstractString
        print(io, '"', _json_escape(v), '"')
    elseif v isa AbstractDict
        print(io, '{')
        ks = sort!(collect(keys(v)), by=string)
        for (i, k) in enumerate(ks)
            i > 1 && print(io, ',')
            print(io, '"', _json_escape(string(k)), '"', ':')
            _json_write(io, v[k])
        end
        print(io, '}')
    elseif v isa AbstractVector
        print(io, '[')
        for (i, x) in enumerate(v)
            i > 1 && print(io, ',')
            _json_write(io, x)
        end
        print(io, ']')
    else
        error("unsupported JSON type: $(typeof(v))")
    end
end

mutable struct _JP
    s::String
    i::Int
end

function _jp_peek(p::_JP)
    while p.i <= sizeof(p.s)
        c = p.s[p.i]
        if c == ' ' || c == '\t' || c == '\n' || c == '\r'
            p.i = nextind(p.s, p.i)
        else
            return c
        end
    end
    return '\0'
end

function _jp_consume!(p::_JP, c::Char)
    got = _jp_peek(p)
    got == c || error("JSON parse: expected '$c' got '$got' at byte $(p.i)")
    p.i = nextind(p.s, p.i)
end

function _jp_parse_string(p::_JP)
    _jp_consume!(p, '"')
    buf = IOBuffer()
    while p.i <= sizeof(p.s)
        c = p.s[p.i]
        if c == '"'
            p.i = nextind(p.s, p.i)
            return String(take!(buf))
        elseif c == '\\'
            p.i = nextind(p.s, p.i)
            esc = p.s[p.i]
            if esc == 'n'
                write(buf, '\n')
            elseif esc == 't'
                write(buf, '\t')
            elseif esc == 'r'
                write(buf, '\r')
            elseif esc == '"'
                write(buf, '"')
            elseif esc == '\\'
                write(buf, '\\')
            elseif esc == '/'
                write(buf, '/')
            elseif esc == 'u'
                hex = p.s[nextind(p.s, p.i):nextind(p.s, p.i, 4)]
                write(buf, Char(parse(UInt32, hex, base=16)))
                p.i = nextind(p.s, p.i, 4)
            else
                error("JSON parse: unknown escape \\$esc")
            end
            p.i = nextind(p.s, p.i)
        else
            write(buf, c)
            p.i = nextind(p.s, p.i)
        end
    end
    error("JSON parse: unterminated string")
end

function _jp_parse_number(p::_JP)
    start = p.i
    while p.i <= sizeof(p.s)
        c = p.s[p.i]
        if c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' ||
           ('0' <= c <= '9')
            p.i = nextind(p.s, p.i)
        else
            break
        end
    end
    text = p.s[start:prevind(p.s, p.i)]
    if occursin('.', text) || occursin('e', text) || occursin('E', text)
        return parse(Float64, text)
    else
        return parse(Int64, text)
    end
end

function _jp_parse_value(p::_JP)
    c = _jp_peek(p)
    if c == '"'
        return _jp_parse_string(p)
    elseif c == 't'
        p.i = nextind(p.s, p.i, 4)
        return true
    elseif c == 'f'
        p.i = nextind(p.s, p.i, 5)
        return false
    elseif c == 'n'
        p.i = nextind(p.s, p.i, 4)
        return nothing
    elseif c == '{'
        return _jp_parse_object(p)
    elseif c == '['
        return _jp_parse_array(p)
    else
        return _jp_parse_number(p)
    end
end

function _jp_parse_object(p::_JP)
    _jp_consume!(p, '{')
    d = Dict{String,Any}()
    if _jp_peek(p) == '}'
        p.i = nextind(p.s, p.i)
        return d
    end
    while true
        key = _jp_parse_string(p)
        _jp_consume!(p, ':')
        d[key] = _jp_parse_value(p)
        c = _jp_peek(p)
        if c == ','
            p.i = nextind(p.s, p.i)
        elseif c == '}'
            p.i = nextind(p.s, p.i)
            return d
        else
            error("JSON parse: expected ',' or '}' at byte $(p.i)")
        end
    end
end

function _jp_parse_array(p::_JP)
    _jp_consume!(p, '[')
    arr = Any[]
    if _jp_peek(p) == ']'
        p.i = nextind(p.s, p.i)
        return arr
    end
    while true
        push!(arr, _jp_parse_value(p))
        c = _jp_peek(p)
        if c == ','
            p.i = nextind(p.s, p.i)
        elseif c == ']'
            p.i = nextind(p.s, p.i)
            return arr
        else
            error("JSON parse: expected ',' or ']' at byte $(p.i)")
        end
    end
end

_json_parse(text::AbstractString) = _jp_parse_value(_JP(String(text), 1))

# ---------------------------------------------------------------------------
# Entry point: NSGA-II bin packing, mean-fitness-on-test
# ---------------------------------------------------------------------------

function _bp_evaluate_best_on_test(result, test_seed::Int, template::BinPackingEvaluator)
    best_idx = argmin(first.(result.pareto_fitnesses))
    best_genome = result.pareto_front[best_idx]

    test_eval = BinPackingEvaluator(
        n_episodes=template.n_episodes,
        n_items=template.n_items,
        capacity=template.capacity,
        item_dist=template.item_dist,
        rng_seed=test_seed,
    )

    fname = gensym("bp_bench_test")
    checked_body = Arborist.add_loop_checks(best_genome.body; limit=1000)
    harness = Arborist.create_harness(best_genome.state, checked_body, fname)
    f_test = @eval $harness
    return Arborist.evaluate(test_eval, f_test)
end

function nsga2_binpack_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_episodes = Int(spec["n_episodes"])
    n_items = Int(spec["n_items"])
    capacity = Float32(spec["capacity"])
    item_dist = Symbol(String(spec["item_dist"]))
    test_offset = Int(spec["test_seed_offset"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])
    _ensure_bp_states()

    inner_eval = BinPackingEvaluator(
        n_episodes=n_episodes, n_items=n_items, capacity=capacity,
        item_dist=item_dist, rng_seed=seed,
    )
    evaluator = BPMultiObjectiveEvaluator(inner_eval)
    fset = bin_packing_function_set()
    algorithm, num_temps = build_tuning_algorithm(generations=generations)
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                 function_set=fset, num_temps=num_temps,
                                 seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    test_fit = _bp_evaluate_best_on_test(result, seed + test_offset, inner_eval)
    train_best = minimum(first.(result.pareto_fitnesses))

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(test_fit),
        "metric_components" => Dict{String,Any}(
            "test_fitness" => Float64(test_fit),
            "train_best_fitness" => Float64(train_best),
            "pareto_front_size" => length(result.pareto_front),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "generations" => generations,
            "pool_size" => parse(Int, get(ENV, "BP_POOL_SIZE", "10000")),
            "notes" => "metric = held-out test-seed fitness of lowest-primary-fitness Pareto member",
        ),
    )
end

# ---------------------------------------------------------------------------
# Entry point: Koza-1 symbolic regression, best_fitness
# ---------------------------------------------------------------------------

function koza_regression_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    target = String(spec["target"])
    x_min = Float64(spec["x_min"])
    x_max = Float64(spec["x_max"])
    n_points = Int(spec["n_points"])
    generations = Int(spec["generations"])
    pop_size = Int(spec["pop_size"])

    # Only Koza-1 is wired; other Koza targets would need distinct corpora.
    target == "x^4 + x^3 + x^2 + x" ||
        error("koza corpus target $(repr(target)) is not wired in this entry point")

    seed = Int(cfg["seed"])

    operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])
    xs = Float32.(range(x_min, x_max, length=n_points))
    X = reshape(xs, 1, :)
    y = xs .^ 4 .+ xs .^ 3 .+ xs .^ 2 .+ xs
    evaluator = TreeFitnessEvaluator(X, y, operators)

    algorithm = GeneticProgramming(
        pop_size=pop_size,
        generations=generations,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2,
    )

    problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "target" => target,
            "n_points" => n_points,
            "notes" => "metric = final best_fitness (MSE-like) on frozen xs",
        ),
    )
end

# ---------------------------------------------------------------------------
# Entry point: 3-bit parity NEAT, best_fitness
# ---------------------------------------------------------------------------

function parity3_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_bits = Int(spec["n_bits"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

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
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=20),
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
            "best_node_count" => length(result.best_genome.nodes),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_bits" => n_bits,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = final best_fitness (MSE) on enumerated 2^n_bits boolean cases",
        ),
    )
end

# ---------------------------------------------------------------------------
# Two-spirals data generation (shared between single- and multi-objective entry points)
# ---------------------------------------------------------------------------

function _two_spirals_dataset(n_per_spiral::Int)
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
    return input_data, output_data
end

# ---------------------------------------------------------------------------
# Entry point: Two-spirals classification NEAT, best_fitness
# ---------------------------------------------------------------------------

function two_spirals_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_per_spiral = Int(spec["n_per_spiral"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

    input_data, output_data = _two_spirals_dataset(n_per_spiral)
    evaluator = GraphEvaluator(input_data, output_data)
    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=20),
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
            "best_node_count" => length(result.best_genome.nodes),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_per_spiral" => n_per_spiral,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = final best_fitness (MSE) on 2*n_per_spiral Lang-Witbrock points",
        ),
    )
end

# ---------------------------------------------------------------------------
# Entry point: NSGA-II two-spirals, best Pareto-front fitness
# ---------------------------------------------------------------------------

function two_spirals_nsga2_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_per_spiral = Int(spec["n_per_spiral"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

    input_data, output_data = _two_spirals_dataset(n_per_spiral)
    evaluator = ParsimonyEvaluator(GraphEvaluator(input_data, output_data))
    ops = neat_defaults()
    algorithm = NSGAII(
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, parallel=false,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    front_fitness = [f[1] for f in result.pareto_fitnesses]
    front_complexity = [f[2] for f in result.pareto_fitnesses]
    best_fit = minimum(front_fitness)

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(best_fit),
        "metric_components" => Dict{String,Any}(
            "best_fit_on_front" => Float64(best_fit),
            "pareto_front_size" => length(result.pareto_front),
            "min_complexity" => Float64(minimum(front_complexity)),
            "max_complexity" => Float64(maximum(front_complexity)),
            "hypervolume_final" => Float64(result.hypervolume_history[end]),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_per_spiral" => n_per_spiral,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = best (lowest) fitness on Pareto front; parsimony vs MSE",
        ),
    )
end

# ---------------------------------------------------------------------------
# Cart-pole dynamics (shared by Phase B control benchmarks)
# ---------------------------------------------------------------------------

const _CP_GRAVITY    = 9.8
const _CP_MASSCART   = 1.0
const _CP_MASSPOLE   = 0.1
const _CP_LENGTH     = 0.5
const _CP_FORCE_MAG  = 10.0
const _CP_TAU        = 0.02
const _CP_X_LIMIT    = 2.4
const _CP_THETA_LIMIT = π / 15.0

function _cartpole_initial_state(rng)
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
_cartpole_decode(y) = y[1] > 0.5 ? 1 : -1

# ---------------------------------------------------------------------------
# Entry point: Single-pole cart-pole NEAT, mean balance duration
# ---------------------------------------------------------------------------

function cartpole_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_episodes = Int(spec["n_episodes"])
    max_steps = Int(spec["max_steps"])
    episode_seed_base = Int(spec["episode_seed_base"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

    evaluator = EpisodicEvaluator(
        4, 1,
        _cartpole_initial_state, _cartpole_dynamics,
        _cartpole_reward, _cartpole_done,
        _cartpole_obs, _cartpole_decode;
        max_steps=max_steps, n_episodes=n_episodes,
        episode_seed_base=episode_seed_base,
        allow_recurrent=false,
    )

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=20),
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
            "best_node_count" => length(result.best_genome.nodes),
            "best_enabled_conns" => count(c.enabled for c in values(result.best_genome.connections)),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_episodes" => n_episodes,
            "max_steps" => max_steps,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = -mean_balance_steps; lower is better; -200 = max-horizon balance",
        ),
    )
end

# ---------------------------------------------------------------------------
# Double-pole (Wieland two-pole) dynamics
# ---------------------------------------------------------------------------

const _DP_GRAVITY     = 9.8
const _DP_MASSCART    = 1.0
const _DP_MASS_LONG   = 0.1
const _DP_MASS_SHORT  = 0.01
const _DP_LENGTH_LONG  = 0.5
const _DP_LENGTH_SHORT = 0.05
const _DP_FORCE_MAG   = 10.0
const _DP_TAU         = 0.01
const _DP_X_LIMIT     = 2.4
const _DP_THETA_LIMIT = π / 15.0
const _DP_MU_C        = 0.0005
const _DP_MU_P        = 0.000002

function _doublepole_initial_state(rng)
    return (x         = 0.0,
            xdot      = 0.0,
            theta1    = 0.07 * (rand(rng) - 0.5),
            theta1dot = 0.0,
            theta2    = 0.07 * (rand(rng) - 0.5),
            theta2dot = 0.0)
end

function _dp_pole_f(theta, theta_dot, m, l)
    costh = cos(theta); sinth = sin(theta)
    return m * l * theta_dot^2 * sinth +
           0.75 * m * costh * (_DP_MU_P * theta_dot / (m * l) +
                               _DP_GRAVITY * sinth)
end

_dp_pole_m(theta, m) = m * (1.0 - 0.75 * cos(theta)^2)

function _doublepole_dynamics(s, a)
    force = a * _DP_FORCE_MAG
    f1 = _dp_pole_f(s.theta1, s.theta1dot, _DP_MASS_LONG, _DP_LENGTH_LONG)
    f2 = _dp_pole_f(s.theta2, s.theta2dot, _DP_MASS_SHORT, _DP_LENGTH_SHORT)
    m1 = _dp_pole_m(s.theta1, _DP_MASS_LONG)
    m2 = _dp_pole_m(s.theta2, _DP_MASS_SHORT)
    x_acc = (force - _DP_MU_C * sign(s.xdot) + f1 + f2) /
            (_DP_MASSCART + m1 + m2)
    theta1_acc = -0.75 / _DP_LENGTH_LONG *
                 (x_acc * cos(s.theta1) + _DP_GRAVITY * sin(s.theta1) +
                  _DP_MU_P * s.theta1dot / (_DP_MASS_LONG * _DP_LENGTH_LONG))
    theta2_acc = -0.75 / _DP_LENGTH_SHORT *
                 (x_acc * cos(s.theta2) + _DP_GRAVITY * sin(s.theta2) +
                  _DP_MU_P * s.theta2dot / (_DP_MASS_SHORT * _DP_LENGTH_SHORT))
    return (x         = s.x + _DP_TAU * s.xdot,
            xdot      = s.xdot + _DP_TAU * x_acc,
            theta1    = s.theta1 + _DP_TAU * s.theta1dot,
            theta1dot = s.theta1dot + _DP_TAU * theta1_acc,
            theta2    = s.theta2 + _DP_TAU * s.theta2dot,
            theta2dot = s.theta2dot + _DP_TAU * theta2_acc)
end

_doublepole_reward(s, a, sp) = 1.0
function _doublepole_done(s)
    abs(s.x) > _DP_X_LIMIT && return true
    abs(s.theta1) > _DP_THETA_LIMIT && return true
    abs(s.theta2) > _DP_THETA_LIMIT && return true
    return false
end
_doublepole_obs(s) = Float64[s.x, s.xdot, s.theta1, s.theta1dot, s.theta2, s.theta2dot]
_doublepole_decode(y) = y[1] > 0.5 ? 1 : -1

# ---------------------------------------------------------------------------
# Entry point: Double-pole Markovian NEAT, mean balance duration
# ---------------------------------------------------------------------------

function double_pole_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_episodes = Int(spec["n_episodes"])
    max_steps = Int(spec["max_steps"])
    episode_seed_base = Int(spec["episode_seed_base"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

    evaluator = EpisodicEvaluator(
        6, 1,
        _doublepole_initial_state, _doublepole_dynamics,
        _doublepole_reward, _doublepole_done,
        _doublepole_obs, _doublepole_decode;
        max_steps=max_steps, n_episodes=n_episodes,
        episode_seed_base=episode_seed_base,
        allow_recurrent=false,
    )

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=25),
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
            "best_node_count" => length(result.best_genome.nodes),
            "best_enabled_conns" => count(c.enabled for c in values(result.best_genome.connections)),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_episodes" => n_episodes,
            "max_steps" => max_steps,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = -mean_balance_steps; Markovian 6-D obs; Wieland two-pole dynamics",
        ),
    )
end

# ---------------------------------------------------------------------------
# Mountain Car dynamics
# ---------------------------------------------------------------------------

const _MC_POS_MIN = -1.2
const _MC_POS_MAX = 0.6
const _MC_VEL_MIN = -0.07
const _MC_VEL_MAX = 0.07
const _MC_GOAL    = 0.5
const _MC_FORCE   = 0.001
const _MC_GRAVITY = 0.0025

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
function _mc_decode(y)
    i = argmax(y)
    return i == 1 ? -1 : (i == 2 ? 0 : 1)
end

# ---------------------------------------------------------------------------
# Entry point: Mountain Car NEAT, mean reward (negated)
# ---------------------------------------------------------------------------

function mountain_car_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_episodes = Int(spec["n_episodes"])
    max_steps = Int(spec["max_steps"])
    episode_seed_base = Int(spec["episode_seed_base"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

    evaluator = EpisodicEvaluator(
        2, 3,
        _mc_initial_state, _mc_dynamics,
        _mc_reward, _mc_done,
        _mc_obs, _mc_decode;
        max_steps=max_steps, n_episodes=n_episodes,
        episode_seed_base=episode_seed_base,
        allow_recurrent=false,
    )

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=25),
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
            "best_node_count" => length(result.best_genome.nodes),
            "best_enabled_conns" => count(c.enabled for c in values(result.best_genome.connections)),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_episodes" => n_episodes,
            "max_steps" => max_steps,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = -mean_reward; lower is better; fitness < 200 ⇒ at least one goal reach",
        ),
    )
end

# ---------------------------------------------------------------------------
# Retina left-and-right dataset construction
# ---------------------------------------------------------------------------

function _retina_dataset(n_input_bits_per_side::Int)
    side_size = 2^n_input_bits_per_side
    n_patterns = side_size * side_size
    input_data  = zeros(Float64, 2 * n_input_bits_per_side, n_patterns)
    output_data = zeros(Float64, 1, n_patterns)
    col = 0
    for left in 0:(side_size - 1)
        left_obj = count_ones(left) == 1
        for right in 0:(side_size - 1)
            col += 1
            right_obj = count_ones(right) == 1
            for i in 1:n_input_bits_per_side
                input_data[i, col] = Float64((left >> (i - 1)) & 1)
                input_data[i + n_input_bits_per_side, col] =
                    Float64((right >> (i - 1)) & 1)
            end
            output_data[1, col] = (left_obj && right_obj) ? 1.0 : 0.0
        end
    end
    return input_data, output_data
end

# ---------------------------------------------------------------------------
# Entry point: Retina classification NEAT, best_fitness
# ---------------------------------------------------------------------------

function retina_mean_fitness(cfg::Dict, corpus_path::AbstractString)
    spec = TOML.parsefile(joinpath(corpus_path, "corpus_spec.toml"))
    n_bits_side = Int(spec["n_input_bits_per_side"])
    pop_size = Int(spec["pop_size"])
    generations = Int(spec["generations"])

    seed = Int(cfg["seed"])

    input_data, output_data = _retina_dataset(n_bits_side)
    evaluator = GraphEvaluator(input_data, output_data)

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=pop_size, generations=generations,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=25),
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=seed)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=false)
    wall = time() - t0

    return Dict{String,Any}(
        "status" => "ok",
        "metric" => Float64(result.best_fitness),
        "metric_components" => Dict{String,Any}(
            "best_fitness" => Float64(result.best_fitness),
            "generations_run" => result.generations_run,
            "converged" => result.converged,
            "best_node_count" => length(result.best_genome.nodes),
            "best_enabled_conns" => count(c.enabled for c in values(result.best_genome.connections)),
        ),
        "wall_clock_seconds" => Float64(wall),
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "threads_actual" => Threads.nthreads(),
            "n_input_bits_per_side" => n_bits_side,
            "pop_size" => pop_size,
            "generations" => generations,
            "notes" => "metric = final best_fitness (MSE) on 2^(2*n_bits) enumerated retina patterns",
        ),
    )
end

const _ENTRY_POINTS = Dict{String,Function}(
    "nsga2_binpack_mean_fitness" => nsga2_binpack_mean_fitness,
    "koza_regression_mean_fitness" => koza_regression_mean_fitness,
    "parity3_mean_fitness" => parity3_mean_fitness,
    "two_spirals_mean_fitness" => two_spirals_mean_fitness,
    "two_spirals_nsga2_mean_fitness" => two_spirals_nsga2_mean_fitness,
    "cartpole_mean_fitness" => cartpole_mean_fitness,
    "double_pole_mean_fitness" => double_pole_mean_fitness,
    "mountain_car_mean_fitness" => mountain_car_mean_fitness,
    "retina_mean_fitness" => retina_mean_fitness,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function _parse_args(args)
    entry = config = output = nothing
    for a in args
        if startswith(a, "--entry=")
            entry = String(a[length("--entry=")+1:end])
        elseif startswith(a, "--config=")
            config = String(a[length("--config=")+1:end])
        elseif startswith(a, "--output=")
            output = String(a[length("--output=")+1:end])
        else
            error("unknown argument: $a")
        end
    end
    return entry, config, output
end

function _write_result(output::AbstractString, result::Dict)
    open(output, "w") do io
        _json_write(io, result)
    end
    return nothing
end

function main()
    entry, config_path, output_path = _parse_args(ARGS)
    (entry === nothing || config_path === nothing || output_path === nothing) &&
        error("usage: bench/benchmarks.jl --entry=X --config=PATH --output=PATH")

    fn = get(_ENTRY_POINTS, entry, nothing)
    if fn === nothing
        _write_result(output_path, Dict{String,Any}(
            "status" => "error",
            "message" => "unknown entry point: $entry",
        ))
        return 1
    end

    cfg = _json_parse(read(config_path, String))
    corpus_path = String(cfg["corpus_path"])

    println("# bench/benchmarks.jl entry=$entry seed=$(cfg["seed"]) rep=$(cfg["repetition_index"])/$(cfg["repetition_total"])")
    flush(stdout)

    t0 = time()
    local result
    try
        result = fn(cfg, corpus_path)
    catch e
        io = IOBuffer()
        showerror(io, e)
        println(io)
        Base.show_backtrace(io, catch_backtrace())
        result = Dict{String,Any}(
            "status" => "error",
            "message" => String(take!(io)),
            "wall_clock_seconds" => Float64(time() - t0),
            "metadata" => Dict{String,Any}(
                "julia_version" => string(VERSION),
                "threads_actual" => Threads.nthreads(),
            ),
        )
    end
    if !haskey(result, "wall_clock_seconds")
        result["wall_clock_seconds"] = Float64(time() - t0)
    end
    _write_result(output_path, result)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
