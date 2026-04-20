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

const _ENTRY_POINTS = Dict{String,Function}(
    "nsga2_binpack_mean_fitness" => nsga2_binpack_mean_fitness,
    "koza_regression_mean_fitness" => koza_regression_mean_fitness,
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
