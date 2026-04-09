#!/usr/bin/env julia
# examples/bin_packing.jl — Online 1D bin packing via evolved heuristics.
#
# Demonstrates evolving a bin packing heuristic using the Arborist.jl
# ExprGenome with a side-effectful simulator, following the AntGenome pattern.
#
# Usage:
#   julia --project -t auto examples/bin_packing.jl
#   julia --project -t 16 examples/bin_packing.jl --generations=100 --pop_size=100

using Arborist
using Dates
using Random
using Statistics

# Downloads.jl (stdlib) is used for LLM experiments — no extra install needed.
using Downloads

# =============================================================================
# Bin packing simulator (module-level mutable state, following AntGenome pattern)
# =============================================================================

mutable struct BinPackingState
    bins::Vector{Float32}      # remaining capacity of each open bin
    n_bins::Int32              # number of bins opened so far
    items_placed::Int32        # items successfully placed
    total_waste::Float32       # accumulated wasted space
    capacity::Float32          # bin capacity (always 1.0f0)
    current_item::Float32      # size of the current item being placed
    placed::Bool               # whether current item was placed this step
end

# Thread-local state: one BinPackingState per thread for parallel evaluation.
const _bp_states = BinPackingState[
    BinPackingState(Float32[], Int32(0), Int32(0), 0.0f0, 1.0f0, 0.0f0, false)
]

function _ensure_bp_states()
    n = Threads.nthreads()
    while length(_bp_states) < n
        push!(_bp_states, BinPackingState(Float32[], Int32(0), Int32(0), 0.0f0, 1.0f0, 0.0f0, false))
    end
end

@inline function _get_bp_state()
    return @inbounds _bp_states[Threads.threadid()]
end

function reset_bp_state!(capacity::Float32=1.0f0)
    s = _get_bp_state()
    empty!(s.bins)
    s.n_bins = Int32(0)
    s.items_placed = Int32(0)
    s.total_waste = 0.0f0
    s.capacity = capacity
    s.current_item = 0.0f0
    s.placed = false
    return nothing
end

# =============================================================================
# Primitives callable from @eval'd evolved programs
# =============================================================================

"""Number of currently open bins."""
function bp_n_bins()::Int32
    return _get_bp_state().n_bins
end

"""Remaining capacity of bin i (1-indexed, clamped to valid range).
Returns 0.0f0 for invalid or out-of-range indices."""
function bp_bin_remaining(i::Int32)::Float32
    s = _get_bp_state()
    s.n_bins == Int32(0) && return 0.0f0
    idx = clamp(Int(i), 1, Int(s.n_bins))
    return s.bins[idx]
end

"""Size of the current item being placed."""
function bp_item_size()::Float32
    return _get_bp_state().current_item
end

"""Bin capacity (always 1.0f0)."""
function bp_capacity()::Float32
    return _get_bp_state().capacity
end

"""Place current item in bin i. Returns true if successful.
If i > n_bins, opens new bins up to i. Returns false if item doesn't fit."""
function bp_place_in_bin(i::Int32)::Bool
    s = _get_bp_state()
    s.placed && return false  # already placed this step
    idx = Int(i)
    idx < 1 && return false

    # Open new bins if needed (cap at n_bins + 1 to avoid runaway allocation)
    target = min(idx, Int(s.n_bins) + 1)
    while target > Int(s.n_bins)
        push!(s.bins, s.capacity)
        s.n_bins += Int32(1)
    end
    idx = target

    # Check if item fits
    if s.bins[idx] >= s.current_item
        s.bins[idx] -= s.current_item
        s.items_placed += Int32(1)
        s.placed = true
        return true
    end
    return false
end

# =============================================================================
# BinPackingEvaluator
# =============================================================================

struct BinPackingEvaluator <: Arborist.AbstractEvaluator
    n_episodes::Int
    n_items::Int
    capacity::Float32
    item_dist::Symbol      # :uniform or :bimodal
    rng_seed::Int
end

function BinPackingEvaluator(;
        n_episodes::Int=20,
        n_items::Int=200,
        capacity::Float32=1.0f0,
        item_dist::Symbol=:uniform,
        rng_seed::Int=42)
    BinPackingEvaluator(n_episodes, n_items, capacity, item_dist, rng_seed)
end

Arborist.input_signature(::BinPackingEvaluator) = Dict{Symbol,DataType}()
Arborist.output_signature(::BinPackingEvaluator) = Dict(:result => Bool)

function generate_items(rng::AbstractRNG, n::Int, dist::Symbol)
    if dist == :uniform
        return Float32.(rand(rng, n))  # uniform in (0, 1)
    elseif dist == :bimodal
        items = Vector{Float32}(undef, n)
        for i in 1:n
            if rand(rng, Bool)
                items[i] = Float32(0.1 + 0.2 * rand(rng))  # small: 0.1-0.3
            else
                items[i] = Float32(0.6 + 0.3 * rand(rng))  # large: 0.6-0.9
            end
        end
        return items
    else
        error("Unknown distribution: $dist")
    end
end

"""Lower bound on bins needed: ceil(sum(items) / capacity)."""
function lower_bound(items::Vector{Float32}, capacity::Float32)
    return ceil(Int, sum(items) / capacity)
end

function Arborist.evaluate(e::BinPackingEvaluator, f::Function)
    total_ratio = 0.0
    for ep in 1:e.n_episodes
        ep_rng = Random.MersenneTwister(e.rng_seed + ep)
        items = generate_items(ep_rng, e.n_items, e.item_dist)
        lb = lower_bound(items, e.capacity)
        lb == 0 && continue

        reset_bp_state!(e.capacity)
        s = _get_bp_state()

        for item in items
            s.current_item = item
            s.placed = false

            try
                Base.invokelatest(f)
            catch
                # evolved program failed
            end

            # Fallback: if not placed, open a new bin
            if !s.placed
                push!(s.bins, s.capacity - item)
                s.n_bins += Int32(1)
                s.items_placed += Int32(1)
            end
        end

        total_ratio += Float64(s.n_bins) / Float64(lb)
    end

    return total_ratio / e.n_episodes  # mean ratio, lower is better
end

# =============================================================================
# ExprGenome integration: evaluate_genome override
# =============================================================================

"""Compile an ExprGenome to a callable function. Returns nothing on failure."""
function _bp_compile(g::Arborist.ExprGenome)
    fname = gensym("bp_evolved")
    try
        checked_body = Arborist.add_loop_checks(g.body; limit=1000)
        harness = Arborist.create_harness(g.state, checked_body, fname)
        return @eval $harness
    catch
        return nothing
    end
end

"""Compile and evaluate an ExprGenome for bin packing."""
function Arborist.evaluate_genome(g::Arborist.ExprGenome, e::BinPackingEvaluator)
    f = _bp_compile(g)
    f === nothing && return Inf
    try
        return Arborist.evaluate(e, f)
    catch
        return Inf
    end
end

"""Compile genomes sequentially, then evaluate in parallel.
@eval requires the global compilation lock so must be sequential;
evaluation uses thread-local BinPackingState and is fully parallel."""
function _bp_parallel_evaluate!(fitnesses::Vector{Float64},
                                genomes::Vector{Arborist.ExprGenome},
                                evaluator::BinPackingEvaluator,
                                bp::Float64,
                                indices)
    _ensure_bp_states()

    # Phase 1: compile sequentially (@eval is module-global)
    compiled = Vector{Any}(undef, length(genomes))
    for i in indices
        compiled[i] = _bp_compile(genomes[i])
    end

    # Phase 2: evaluate in parallel (thread-safe with thread-local state)
    Threads.@threads for i in collect(indices)
        f = compiled[i]
        if f === nothing
            fitnesses[i] = Inf
        else
            try
                raw = Arborist.evaluate(evaluator, f)
                if bp > 0.0 && isfinite(raw)
                    raw += bp * Arborist.complexity(genomes[i])
                end
                fitnesses[i] = raw
            catch
                fitnesses[i] = Inf
            end
        end
    end
end

# =============================================================================
# Behavioral fingerprinting for BehavioralSpeciation
#
# A behavioral fingerprint records which bin a program chooses for each item
# in a fixed probe sequence. Two programs with identical placement decisions
# get distance 0.0 regardless of AST structure.
# =============================================================================

struct BinPackingFingerprint
    choices::Vector{Int32}   # bin index chosen for each probe item (-1 = fallback)
end

struct BehavioralProbe
    probe_items::Vector{Float32}
    n_probe_bins::Int
    probe_seed::Int
    n_items::Int
end

function BehavioralProbe(; n_items::Int=50, n_probe_bins::Int=20, probe_seed::Int=999)
    rng = Random.MersenneTwister(probe_seed)
    items = Float32.(rand(rng, n_items))
    BehavioralProbe(items, n_probe_bins, probe_seed, n_items)
end

"""Compute behavioral fingerprint: run compiled function on each probe item
and record which bin was chosen."""
function _bp_fingerprint_from_fn(f, probe::BehavioralProbe)::BinPackingFingerprint
    _ensure_bp_states()
    reset_bp_state!(1.0f0)
    s = _get_bp_state()
    # Pre-open bins so the program has bins to choose from immediately
    for _ in 1:probe.n_probe_bins
        push!(s.bins, 1.0f0)
        s.n_bins += Int32(1)
    end

    choices = Vector{Int32}(undef, probe.n_items)
    for (idx, item) in enumerate(probe.probe_items)
        s.current_item = item
        s.placed = false
        n_bins_before = s.n_bins
        bins_before = copy(s.bins)

        try
            Base.invokelatest(f)
        catch
        end

        if s.placed
            # Determine which bin was modified by comparing bins arrays
            choice = Int32(-1)
            for j in 1:min(Int(n_bins_before), length(bins_before))
                if j <= length(s.bins) && s.bins[j] != bins_before[j]
                    choice = Int32(j)
                    break
                end
            end
            # Check newly opened bins
            if choice == Int32(-1) && s.n_bins > n_bins_before
                choice = s.n_bins
            end
            choices[idx] = choice
        else
            choices[idx] = Int32(-1)
            # Apply fallback so state is consistent for next item
            push!(s.bins, 1.0f0 - item)
            s.n_bins += Int32(1)
        end
    end
    return BinPackingFingerprint(choices)
end

"""Compute behavioral fingerprint for an ExprGenome."""
function compute_bp_fingerprint(g::Arborist.ExprGenome, probe::BehavioralProbe)::BinPackingFingerprint
    f = _bp_compile(g)
    if f === nothing
        return BinPackingFingerprint(fill(Int32(-1), probe.n_items))
    end
    return _bp_fingerprint_from_fn(f, probe)
end

"""Hamming distance between two fingerprints: fraction of items where
the programs made different bin choices."""
function behavioral_distance(a::BinPackingFingerprint, b::BinPackingFingerprint)::Float64
    n = length(a.choices)
    @assert n == length(b.choices)
    n == 0 && return 0.0
    mismatches = sum(a.choices[i] != b.choices[i] for i in 1:n)
    return mismatches / n
end

# =============================================================================
# TrackedMutation — wrapper for tracking LLM operator call metrics
# =============================================================================

mutable struct TrackedMutation <: Arborist.AbstractMutationOperator
    inner::Any          # the wrapped operator (e.g., LLMMutationOperator)
    n_calls::Int        # total mutate() calls
    n_slow::Int         # calls > 0.5s (likely successful LLM inference)
    total_latency::Float64
end

TrackedMutation(inner) = TrackedMutation(inner, 0, 0, 0.0)

function Arborist.mutate(op::TrackedMutation, genome::Arborist.ExprGenome, rng::AbstractRNG)
    t0 = time()
    result = Arborist.mutate(op.inner, genome, rng)
    dt = time() - t0
    op.n_calls += 1
    op.total_latency += dt
    if dt > 0.5  # LLM inference takes seconds; SubtreeMutation takes microseconds
        op.n_slow += 1
    end
    return result
end

# =============================================================================
# LLM system prompt for bin packing (used by Experiment A)
# =============================================================================

const BP_LLM_SYSTEM_PROMPT = """
You are a genetic programming mutation operator for an online bin
packing heuristic written in Julia.

The program runs once per item to be packed. It has access to these
primitives:
  bp_n_bins()::Int32          -- number of currently open bins
  bp_bin_remaining(i::Int32)::Float32  -- remaining capacity of bin i (1-indexed)
  bp_item_size()::Float32     -- size of current item (between 0 and 1)
  bp_capacity()::Float32      -- bin capacity (always 1.0)
  bp_place_in_bin(i::Int32)::Bool  -- place item in bin i, returns true if successful

The program uses these temp variables:
  __temp_1, __temp_2, __temp_3 :: Int32  (loop counters, bin indices)
  __temp_4, __temp_5, __temp_6 :: Float32  (scores, remainders)
  result :: Bool  (output, set by bp_place_in_bin)

Rules:
- Return ONLY valid Julia assignment statements and control flow
- Use only the variables and primitives listed above
- Do not import anything or define functions
- The goal is to place the item in the bin that minimizes wasted space
  (Best Fit: find the bin with least remaining capacity that still fits)
- A while loop scanning from bin 1 to bp_n_bins() with a conditional
  tracking the best bin found so far is the key structure to discover

Respond with only the Julia statements, nothing else.
"""

# =============================================================================
# Shared overnight experiment helpers
# =============================================================================

const OVERNIGHT_SEEDS = [42, 123, 456, 789, 1337]

function _classical_ops()
    [Arborist.SubtreeMutation(), Arborist.PointMutation(),
     Arborist.HoistMutation(), Arborist.ExpansionMutation()]
end

function _common_kwargs(; seed::Int=42, generations::Int=100, pop_size::Int=200)
    Dict{Symbol,Any}(
        :pop_size => pop_size, :generations => generations,
        :mutation_rate => 0.4, :crossover_rate => 0.3,
        :elitism => 3,
        :bloat_penalty => 0.0005,
        :n_episodes => 20, :n_items => 200, :rng_seed => seed,
        :item_dist => :uniform,
    )
end

function _check_ollama()
    try
        output = IOBuffer()
        Downloads.request(
            "http://localhost:11434/v1/chat/completions";
            method="POST",
            headers=["Content-Type" => "application/json"],
            input=IOBuffer("""{"model":"qwen3-coder:30b","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":5}"""),
            output=output,
            timeout=30
        )
        println("Ollama check: OK")
        flush(stdout)
        return true
    catch e
        println("ERROR: Ollama not reachable: $e")
        flush(stdout)
        return false
    end
end

function _make_tracked_llm()
    # LLMMutationOperator was migrated from a weakdep extension into the
    # core module when HTTP.jl was replaced by Downloads.jl (stdlib).
    llm_op = Arborist.LLMMutationOperator(
        endpoint    = "http://localhost:11434/v1/chat/completions",
        model       = "qwen3-coder:30b",
        api_key_env = "",
        system_prompt = BP_LLM_SYSTEM_PROMPT,
        temperature = 0.7,
        max_tokens  = 256,
        timeout_seconds = 60.0,
        fallback_op = Arborist.SubtreeMutation()
    )
    return TrackedMutation(llm_op)
end

function _ensure_data_dir()
    d = joinpath(@__DIR__, "data")
    isdir(d) || mkpath(d)
    return d
end

"""Write seed results to a TSV data file for combine_results."""
function _write_seed_data(filename::String, rows::Vector)
    data_dir = _ensure_data_dir()
    path = joinpath(data_dir, filename)
    open(path, "w") do io
        println(io, "seed\ttrain\ttest\twall_time\tff_test\tbf_test\tllm_calls\tllm_ok\tllm_latency")
        for r in rows
            println(io, join([r.seed, round(r.train, digits=6), round(r.test, digits=6),
                              round(r.wall_time, digits=1), round(r.ff_test, digits=6),
                              round(r.bf_test, digits=6), r.llm_calls, r.llm_ok,
                              round(r.llm_latency, digits=2)], "\t"))
        end
    end
    println("Data written to: $path")
    flush(stdout)
end

"""Read TSV data file written by _write_seed_data."""
function _read_seed_data(filename::String)
    data_dir = _ensure_data_dir()
    path = joinpath(data_dir, filename)
    isfile(path) || return nothing
    rows = NamedTuple[]
    for line in readlines(path)
        startswith(line, "seed") && continue
        parts = split(line, "\t")
        length(parts) >= 6 || continue
        push!(rows, (
            seed = round(Int, parse(Float64, parts[1])),
            train = parse(Float64, parts[2]),
            test = parse(Float64, parts[3]),
            wall_time = parse(Float64, parts[4]),
            ff_test = parse(Float64, parts[5]),
            bf_test = parse(Float64, parts[6]),
            llm_calls = length(parts) >= 7 ? round(Int, parse(Float64, parts[7])) : 0,
            llm_ok = length(parts) >= 8 ? round(Int, parse(Float64, parts[8])) : 0,
            llm_latency = length(parts) >= 9 ? parse(Float64, parts[9]) : 0.0,
        ))
    end
    return rows
end

# =============================================================================
# Custom GenState and initial program generation
#
# The standard GenState only includes types from inputs/outputs in used_types.
# With empty inputs and Bool output, all temps would be Bool — useless for
# bin packing. We need Int32 and Float32 temps explicitly.
#
# The standard create_random_rvalue only returns variables or literals, never
# function calls. For bin packing, initial programs need function calls like
# bp_n_bins(), bp_bin_remaining(i), bp_item_size() to have any chance of
# evolving useful heuristics.
# =============================================================================

"""Create a GenState with explicit Int32/Float32/Bool temp variables."""
function _bp_create_state(rng::AbstractRNG, fset::Arborist.FunctionSet)
    inputs = Dict{Symbol, DataType}()
    outputs = Dict{Symbol, DataType}(:result => Bool)
    used_types = Set{DataType}([Bool, Int32, Float32])

    # Explicit temps: 3 Int32 (loop counter, bin index, spare) + 3 Float32
    temps = Dict{Symbol, DataType}(
        :__temp_1 => Int32,
        :__temp_2 => Int32,
        :__temp_3 => Int32,
        :__temp_4 => Float32,
        :__temp_5 => Float32,
        :__temp_6 => Float32,
    )

    all_vars = merge(inputs, outputs, temps)
    statement_types = [:(=), :call, :for, :while, :if, :block]

    return Arborist.GenState(rng, statement_types, fset, inputs, outputs,
                              temps, used_types, all_vars)
end

"""Create a random rvalue of type T, sometimes using function calls.
Unlike the standard create_random_rvalue which only returns variables/literals,
this version can generate function call expressions like bp_n_bins() or
bp_bin_remaining(__temp_1), which are essential building blocks for bin packing."""
function _bp_random_rvalue(state::Arborist.GenState, T::DataType)
    r = rand(state.rng)
    if r < 0.4
        # Try to generate a function call returning T
        valid_funcs = Arborist._sorted_funcs(
            f for f in state.funcs.funcs if f.return_type == T
        )
        if !isempty(valid_funcs)
            f = rand(state.rng, valid_funcs)
            args = [Arborist.create_random_rvalue(state, at) for at in f.args]
            return Expr(:call, f.name, args...)
        end
    end
    return Arborist.create_random_rvalue(state, T)
end

"""Create a random assignment that can include function calls as rvalues."""
function _bp_random_assignment(state::Arborist.GenState)
    lvalues = Arborist.get_lvalues(state)
    v = rand(state.rng, lvalues)
    r = _bp_random_rvalue(state, v[2])
    # Avoid self-assignment (symbol == symbol)
    if r isa Symbol && v[1] == r
        r = Arborist.get_random_literal(state.rng, v[2])
    end
    return :($(v[1]) = $r)
end

"""Generate an initial program body for bin packing.
Creates a mix of assignments with function calls, control flow, and
standalone function calls to the primitives."""
function _bp_random_initial_body(state::Arborist.GenState)
    stmts = Expr[]
    n_stmts = rand(state.rng, 4:7)
    for _ in 1:n_stmts
        r = rand(state.rng)
        if r < 0.15
            # While loop (important for iterating over bins)
            push!(stmts, Arborist.create_random_while_loop(state; depth=2))
        elseif r < 0.25
            # If statement
            push!(stmts, Arborist.create_random_if_statement(state; depth=2))
        elseif r < 0.35
            # Standalone function call (may trigger side effects like bp_place_in_bin)
            T = rand(state.rng, Arborist._sorted_types(state.used_types))
            push!(stmts, Arborist.create_random_function_call(state, T))
        else
            # Assignment with possible function call rvalue
            push!(stmts, _bp_random_assignment(state))
        end
    end
    return stmts
end

"""Generate a seeded initial program that places items in a specific bin.
These seeds provide useful starting points for evolution:
- Type 1: try bin 1 (approximates first-fit with fallback)
- Type 2: try bin at index n_bins (try last bin)
- Type 3: loop over bins looking for fit, with simple placement
- Type 4: place in bin computed from item size"""
function _bp_seeded_body(state::Arborist.GenState, seed_type::Int)
    if seed_type == 1
        # Always try bin 1 — items overflow to fallback (new bin)
        return Expr[
            :(__temp_1 = Int32(1)),
            :(result = bp_place_in_bin(__temp_1)),
        ]
    elseif seed_type == 2
        # Try the last open bin, fall back to new bin
        return Expr[
            :(__temp_1 = bp_n_bins()),
            :(if __temp_1 > Int32(0)
                result = bp_place_in_bin(__temp_1)
            else
                result = bp_place_in_bin(Int32(1))
            end),
        ]
    elseif seed_type == 3
        # Linear scan: iterate over bins, place in first that fits
        return Expr[
            :(__temp_1 = Int32(1)),
            :(while __temp_1 <= bp_n_bins()
                __temp_4 = bp_bin_remaining(__temp_1)
                if __temp_4 >= bp_item_size()
                    result = bp_place_in_bin(__temp_1)
                end
                __temp_1 = __temp_1 + Int32(1)
            end),
            :(if !result
                __temp_2 = bp_n_bins() + Int32(1)
                result = bp_place_in_bin(__temp_2)
            end),
        ]
    elseif seed_type == 4
        # Best-fit skeleton: track best bin
        return Expr[
            :(__temp_1 = Int32(1)),
            :(__temp_2 = Int32(0)),
            :(__temp_5 = Float32(2.0)),
            :(while __temp_1 <= bp_n_bins()
                __temp_4 = bp_bin_remaining(__temp_1)
                if __temp_4 >= bp_item_size()
                    __temp_6 = __temp_4 - bp_item_size()
                    if __temp_6 < __temp_5
                        __temp_2 = __temp_1
                        __temp_5 = __temp_6
                    end
                end
                __temp_1 = __temp_1 + Int32(1)
            end),
            :(if __temp_2 > Int32(0)
                result = bp_place_in_bin(__temp_2)
            else
                __temp_3 = bp_n_bins() + Int32(1)
                result = bp_place_in_bin(__temp_3)
            end),
        ]
    else
        # Random
        return _bp_random_initial_body(state)
    end
end

# =============================================================================
# Custom solve for BinPackingEvaluator with ExprGenome
# =============================================================================

function Arborist.solve(problem::Arborist.GPProblem{Arborist.ExprGenome, E},
                        algorithm::Arborist.GeneticProgramming;
                        verbose::Bool = false,
                        callback = nothing) where {E<:BinPackingEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    evaluator = problem.evaluator
    pop_size = algorithm.pop_size

    # Create custom GenState with explicit Int32/Float32 temps
    state = _bp_create_state(rng, problem.function_set)

    # Initialize population: mix of seeded templates and random programs.
    # Seeds provide useful starting points; random programs provide diversity.
    genomes = Vector{Arborist.ExprGenome}(undef, pop_size)
    n_seed_types = 4
    seeds_per_type = max(1, pop_size ÷ 10)  # ~10% of population per seed type
    n_seeded = min(seeds_per_type * n_seed_types, pop_size ÷ 2)
    for i in 1:pop_size
        if i <= n_seeded
            seed_type = ((i - 1) % n_seed_types) + 1
            body = _bp_seeded_body(state, seed_type)
        else
            body = _bp_random_initial_body(state)
        end
        genomes[i] = Arborist.ExprGenome(body, state)
    end
    fitnesses = fill(Inf, pop_size)

    bp = algorithm.bloat_penalty

    _bp_parallel_evaluate!(fitnesses, genomes, evaluator, bp, 1:pop_size)

    species_state = Arborist._init_species_state(algorithm.speciation)
    fitness_history = Float64[]
    mean_history = Float64[]
    t0 = time()

    for gen in 1:algorithm.generations
        order = sortperm(fitnesses)
        genomes = genomes[order]
        fitnesses = fitnesses[order]

        push!(fitness_history, fitnesses[1])
        finite_fits = filter(isfinite, fitnesses)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        selection_fitnesses = Arborist._apply_speciation!(genomes, fitnesses,
                                                           algorithm.speciation, species_state, rng)

        # Update LLM operator contexts with current population state.
        Arborist._update_llm_contexts!(algorithm.mutation_ops, gen, algorithm.generations,
                                        fitnesses, genomes)

        elapsed = round(time() - t0, digits=1)
        if verbose
            n_species = species_state isa Vector ? length(species_state) : 0
            species_str = n_species > 0 ? " | species=$n_species" : ""
            # LLM tracking stats
            llm_str = ""
            for op in algorithm.mutation_ops
                if op isa TrackedMutation
                    llm_str = " | llm_calls=$(op.n_calls) llm_ok=$(op.n_slow)"
                    break
                end
            end
            println("Gen $gen/$(algorithm.generations) | best=$(round(fitnesses[1], digits=4)) | " *
                    "mean=$(round(mean_fit, digits=4))$species_str$llm_str | elapsed=$(elapsed)s")
            flush(stdout)
        end

        callback !== nothing && callback(gen, fitnesses[1], genomes[1])

        next_genomes = Vector{Arborist.ExprGenome}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        t_size = algorithm.selection.tournament_size

        idx = algorithm.elitism + 1
        while idx <= pop_size
            r = rand(rng)
            if r < algorithm.crossover_rate && idx + 1 <= pop_size
                p1 = Arborist._tournament_select(selection_fitnesses, t_size, rng)
                p2 = Arborist._tournament_select(selection_fitnesses, t_size, rng)
                op = rand(rng, algorithm.crossover_ops)
                (c1, c2) = Arborist.crossover(op, genomes[p1], genomes[p2], rng)
                next_genomes[idx] = c1
                next_genomes[idx + 1] = c2
                idx += 2
            elseif r < algorithm.crossover_rate + algorithm.mutation_rate
                p_idx = Arborist._tournament_select(selection_fitnesses, t_size, rng)
                op = rand(rng, algorithm.mutation_ops)
                Arborist._set_parent_context!(algorithm.mutation_ops, p_idx, selection_fitnesses)
                child = Arborist.mutate(op, genomes[p_idx], rng)
                next_genomes[idx] = child
                idx += 1
            else
                p_idx = Arborist._tournament_select(selection_fitnesses, t_size, rng)
                next_genomes[idx] = deepcopy(genomes[p_idx])
                idx += 1
            end
        end

        _bp_parallel_evaluate!(next_fitnesses, next_genomes, evaluator, bp,
                               (algorithm.elitism + 1):pop_size)

        genomes = next_genomes
        fitnesses = next_fitnesses
    end

    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    return Arborist.GPResult{Arborist.ExprGenome}(
        genomes[1], fitnesses[1], genomes,
        fitness_history, mean_history,
        algorithm.generations, time() - t0,
        fitnesses[1] < 1.05  # converged if within 5% of optimal
    )
end

# =============================================================================
# Baseline heuristics for comparison
# =============================================================================

"""First Fit: place item in first bin that fits, else open new bin."""
function first_fit(items::Vector{Float32}, capacity::Float32)
    bins = Float32[]
    for item in items
        placed = false
        for j in 1:length(bins)
            if bins[j] >= item
                bins[j] -= item
                placed = true
                break
            end
        end
        if !placed
            push!(bins, capacity - item)
        end
    end
    return length(bins)
end

"""Best Fit: place item in the bin with least remaining space that still fits."""
function best_fit(items::Vector{Float32}, capacity::Float32)
    bins = Float32[]
    for item in items
        best_idx = 0
        best_remaining = capacity + 1.0f0
        for j in 1:length(bins)
            if bins[j] >= item && bins[j] - item < best_remaining
                best_idx = j
                best_remaining = bins[j] - item
            end
        end
        if best_idx > 0
            bins[best_idx] -= item
        else
            push!(bins, capacity - item)
        end
    end
    return length(bins)
end

"""Worst Fit: place item in the bin with most remaining space that still fits."""
function worst_fit(items::Vector{Float32}, capacity::Float32)
    bins = Float32[]
    for item in items
        worst_idx = 0
        worst_remaining = -1.0f0
        for j in 1:length(bins)
            if bins[j] >= item && bins[j] > worst_remaining
                worst_idx = j
                worst_remaining = bins[j]
            end
        end
        if worst_idx > 0
            bins[worst_idx] -= item
        else
            push!(bins, capacity - item)
        end
    end
    return length(bins)
end

"""Run a baseline heuristic over the same episodes as the evaluator."""
function baseline_normalized(heuristic::Function, eval_cfg::BinPackingEvaluator)
    total_ratio = 0.0
    for ep in 1:eval_cfg.n_episodes
        ep_rng = Random.MersenneTwister(eval_cfg.rng_seed + ep)
        items = generate_items(ep_rng, eval_cfg.n_items, eval_cfg.item_dist)
        lb = lower_bound(items, eval_cfg.capacity)
        lb == 0 && continue
        bins_used = heuristic(items, eval_cfg.capacity)
        total_ratio += Float64(bins_used) / Float64(lb)
    end
    return total_ratio / eval_cfg.n_episodes
end

# =============================================================================
# Function set for bin packing
# =============================================================================

function bin_packing_function_set()
    fset = Arborist.FunctionSet(Set{Arborist.FunctionDetails}())

    # Bin packing primitives
    push!(fset.funcs, Arborist.FunctionDetails(:bp_n_bins, DataType[], Int32))
    push!(fset.funcs, Arborist.FunctionDetails(:bp_item_size, DataType[], Float32))
    push!(fset.funcs, Arborist.FunctionDetails(:bp_capacity, DataType[], Float32))
    push!(fset.funcs, Arborist.FunctionDetails(:bp_bin_remaining, [Int32], Float32))
    push!(fset.funcs, Arborist.FunctionDetails(:bp_place_in_bin, [Int32], Bool))

    # Arithmetic
    for T in [Float32, Int32]
        for func in [:+, :-, :*, :/]
            Arborist.add!(fset, func, 2, T, T)
        end
    end

    # Comparisons
    for T in [Float32, Int32]
        for op in [:>, :<, :(==), :!=, :>=, :<=]
            Arborist.add!(fset, op, 2, T, Bool)
        end
    end

    return fset
end

# =============================================================================
# Pretty-print evolved program
# =============================================================================

"""Strip line-number comments (LineNumberNode artifacts) from an Expr string."""
function _strip_line_comments(s::String)
    lines = split(s, "\n")
    filtered = filter(l -> !occursin(r"^\s*#=.*=#\s*$", l), lines)
    return join(filtered, "\n")
end

function pretty_print_genome(g::Arborist.ExprGenome)
    io = IOBuffer()
    println(io, "function evolved_heuristic()")
    sorted_outputs = sort(collect(g.state.outputs), by=first)
    for (v, T) in sorted_outputs
        println(io, "    $v = $(Arborist.default_value(T))")
    end
    sorted_temps = sort(collect(g.state.temps), by=first)
    for (v, T) in sorted_temps
        println(io, "    $v = $(Arborist.default_value(T))")
    end
    println(io)
    for stmt in g.body
        cleaned = _strip_line_comments(string(stmt))
        for line in split(cleaned, "\n")
            isempty(strip(line)) && continue
            println(io, "    $line")
        end
    end
    println(io)
    for (v, _) in sorted_outputs
        println(io, "    return $v")
    end
    println(io, "end")
    return String(take!(io))
end

# =============================================================================
# Main entry point
# =============================================================================

function run_bin_packing(;
        pop_size::Int = 200,
        generations::Int = 500,
        mutation_rate::Float64 = 0.4,
        crossover_rate::Float64 = 0.3,
        elitism::Int = 3,
        tournament_size::Int = 5,  # used via TournamentSelection
        bloat_penalty::Float64 = 0.001,
        speciation::Arborist.AbstractSpeciation = Arborist.NoSpeciation(),
        mutation_ops::Union{Nothing, Vector} = nothing,
        n_episodes::Int = 20,
        n_items::Int = 200,
        capacity::Float32 = 1.0f0,
        item_dist::Symbol = :uniform,
        rng_seed::Int = 42,
        num_temps::Int = 6,
        output_file::String = "bin_packing_results.md",
        verbose::Bool = true)

    _ensure_bp_states()

    println("=" ^ 70)
    println("Arborist.jl — Online Bin Packing Evolution")
    println("  Threads: $(Threads.nthreads())")
    println("=" ^ 70)
    flush(stdout)

    evaluator = BinPackingEvaluator(
        n_episodes=n_episodes, n_items=n_items,
        capacity=capacity, item_dist=item_dist, rng_seed=rng_seed
    )

    println("\nBaseline heuristics ($(n_episodes) episodes × $(n_items) items, $item_dist):")
    ff_score = baseline_normalized(first_fit, evaluator)
    bf_score = baseline_normalized(best_fit, evaluator)
    wf_score = baseline_normalized(worst_fit, evaluator)
    println("  First Fit:  $(round(ff_score, digits=4))")
    println("  Best Fit:   $(round(bf_score, digits=4))")
    println("  Worst Fit:  $(round(wf_score, digits=4))")
    println("  (lower is better, 1.0 = optimal)")
    flush(stdout)

    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=num_temps, seed=rng_seed)

    spec_desc = if speciation isa Arborist.NoSpeciation
        "none"
    elseif speciation isa Arborist.BehavioralSpeciation
        "behavioral(threshold=$(speciation.threshold), sharing=$(speciation.sharing_formula))"
    else
        "threshold=$(speciation.threshold), sharing=$(speciation.sharing_formula)"
    end

    algo_kwargs = Dict{Symbol,Any}(
        :pop_size => pop_size,
        :generations => generations,
        :mutation_rate => mutation_rate,
        :crossover_rate => crossover_rate,
        :elitism => elitism,
        :selection => Arborist.TournamentSelection(tournament_size),
        :bloat_penalty => bloat_penalty,
        :speciation => speciation,
    )
    if mutation_ops !== nothing
        algo_kwargs[:mutation_ops] = mutation_ops
    end
    algorithm = Arborist.GeneticProgramming(; algo_kwargs...)

    println("\nStarting evolution: pop=$pop_size, gens=$generations, speciation=$spec_desc")
    println("-" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=verbose)
    wall_time = time() - t0

    println("-" ^ 70)
    println("\nResults after $(result.generations_run) generations ($(round(wall_time, digits=1))s):")
    println("  Best fitness:     $(round(result.best_fitness, digits=4))")
    println("  vs First Fit:     $(round((ff_score - result.best_fitness) / ff_score * 100, digits=2))%")
    println("  vs Best Fit:      $(round((bf_score - result.best_fitness) / bf_score * 100, digits=2))%")
    println("  Converged:        $(result.converged)")
    flush(stdout)

    # Evaluate on a test set with different seeds
    println("\nTest set evaluation (seed offset +1000):")
    test_eval = BinPackingEvaluator(
        n_episodes=n_episodes, n_items=n_items,
        capacity=capacity, item_dist=item_dist, rng_seed=rng_seed + 1000
    )
    test_ff = baseline_normalized(first_fit, test_eval)
    test_bf = baseline_normalized(best_fit, test_eval)

    fname = gensym("bp_test")
    checked_body = Arborist.add_loop_checks(result.best_genome.body; limit=1000)
    harness = Arborist.create_harness(result.best_genome.state, checked_body, fname)
    f_test = @eval $harness
    test_evolved = Arborist.evaluate(test_eval, f_test)

    println("  First Fit:  $(round(test_ff, digits=4))")
    println("  Best Fit:   $(round(test_bf, digits=4))")
    println("  Evolved:    $(round(test_evolved, digits=4))")
    println("  vs FF:      $(round((test_ff - test_evolved) / test_ff * 100, digits=2))%")
    println("  vs BF:      $(round((test_bf - test_evolved) / test_bf * 100, digits=2))%")
    flush(stdout)

    println("\nBest evolved program:")
    println("-" ^ 70)
    println(pretty_print_genome(result.best_genome))
    println("-" ^ 70)
    flush(stdout)

    # Save results
    results_path = joinpath(@__DIR__, output_file)
    open(results_path, "w") do io
        println(io, "# Bin Packing Evolution Results")
        println(io)
        println(io, "Generated: $(Dates.now())")
        println(io)
        println(io, "## Configuration")
        println(io, "- Population: $pop_size")
        println(io, "- Generations: $(result.generations_run)")
        println(io, "- Items per episode: $n_items")
        println(io, "- Episodes: $n_episodes")
        println(io, "- Distribution: $item_dist")
        println(io, "- Wall time: $(round(wall_time, digits=1))s")
        println(io)
        println(io, "## Training Set Results")
        println(io, "| Heuristic | Normalized bins |")
        println(io, "|-----------|----------------|")
        println(io, "| First Fit | $(round(ff_score, digits=4)) |")
        println(io, "| Best Fit  | $(round(bf_score, digits=4)) |")
        println(io, "| Worst Fit | $(round(wf_score, digits=4)) |")
        println(io, "| **Evolved** | **$(round(result.best_fitness, digits=4))** |")
        println(io)
        println(io, "## Test Set Results (different seeds)")
        println(io, "| Heuristic | Normalized bins |")
        println(io, "|-----------|----------------|")
        println(io, "| First Fit | $(round(test_ff, digits=4)) |")
        println(io, "| Best Fit  | $(round(test_bf, digits=4)) |")
        println(io, "| **Evolved** | **$(round(test_evolved, digits=4))** |")
        println(io)
        println(io, "## Best Evolved Program")
        println(io, "```julia")
        println(io, pretty_print_genome(result.best_genome))
        println(io, "```")
        println(io)
        println(io, "## Fitness History (sampled)")
        n_samples = min(20, length(result.fitness_history))
        step = max(1, length(result.fitness_history) ÷ n_samples)
        println(io, "| Generation | Best Fitness |")
        println(io, "|-----------|-------------|")
        for i in 1:step:length(result.fitness_history)
            println(io, "| $i | $(round(result.fitness_history[i], digits=4)) |")
        end
        if length(result.fitness_history) % step != 1
            println(io, "| $(length(result.fitness_history)) | $(round(result.fitness_history[end], digits=4)) |")
        end
    end
    println("\nResults saved to: $results_path")
    flush(stdout)

    return (
        result = result,
        ff_train = ff_score, bf_train = bf_score, wf_train = wf_score,
        ff_test = test_ff, bf_test = test_bf,
        evolved_test = test_evolved,
        wall_time = wall_time,
        program_text = pretty_print_genome(result.best_genome),
    )
end

"""Create a BehavioralSpeciation configured for bin packing."""
function bp_behavioral_speciation(;
        threshold::Float64=0.15,
        sharing_formula::Symbol=:sqrt,
        min_species_size::Int=2,
        stagnation_limit::Int=15,
        n_probe_items::Int=50,
        n_probe_bins::Int=20)
    probe = BehavioralProbe(n_items=n_probe_items, n_probe_bins=n_probe_bins)
    return Arborist.BehavioralSpeciation(
        fingerprint_fn = g -> compute_bp_fingerprint(g, probe),
        distance_fn = behavioral_distance,
        threshold = threshold,
        min_species_size = min_species_size,
        stagnation_limit = stagnation_limit,
        sharing_formula = sharing_formula
    )
end

# =============================================================================
# Experiment A: LLM Operator via Local Ollama
# =============================================================================

function run_experiment_a(; generations::Int=100, pop_size::Int=200)
    println("=" ^ 70)
    println("EXPERIMENT A: LLM Operator via Local Ollama")
    println("=" ^ 70)
    flush(stdout)

    common_kwargs = Dict{Symbol,Any}(
        :pop_size => pop_size, :generations => generations,
        :mutation_rate => 0.4, :crossover_rate => 0.3,
        :elitism => 3,
        :bloat_penalty => 0.0005,
        :n_episodes => 20, :n_items => 200, :rng_seed => 42,
        :item_dist => :uniform,
    )

    classical_ops = [Arborist.SubtreeMutation(), Arborist.PointMutation(),
                     Arborist.HoistMutation(), Arborist.ExpansionMutation()]

    # --- A1: Classical only (control) ---
    println("\n>>> Variant A1: Classical only (control)")
    flush(stdout)
    r_a1 = run_bin_packing(; common_kwargs...,
        mutation_ops = classical_ops,
        output_file = "bin_packing_results_a1.md",
    )

    # --- A2: Classical + LLM (if Ollama available) ---
    r_a2 = nothing
    tracked_llm = nothing
    ollama_ok = false

    try
        output = IOBuffer()
        Downloads.request(
            "http://localhost:11434/v1/chat/completions";
            method="POST",
            headers=["Content-Type" => "application/json"],
            input=IOBuffer("""{"model":"qwen3-coder:30b","messages":[{"role":"user","content":"Reply with only: OK"}],"max_tokens":5}"""),
            output=output,
            timeout=30
        )
        println("\nOllama check: ", String(take!(output)))
        ollama_ok = true
    catch e
        println("\nERROR: Ollama not reachable: $e")
        println("Skipping LLM variant (A2).")
        println("To test manually:")
        println("  curl http://localhost:11434/v1/chat/completions \\")
        println("    -H 'Content-Type: application/json' \\")
        println("    -d '{\"model\":\"qwen3-coder:30b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":5}'")
    end
    flush(stdout)

    if ollama_ok
        println("\n>>> Variant A2: Classical + Qwen3-Coder LLM (20% weight)")
        flush(stdout)

        llm_op = Arborist.LLMMutationOperator(
            endpoint    = "http://localhost:11434/v1/chat/completions",
            model       = "qwen3-coder:30b",
            api_key_env = "",
            system_prompt = BP_LLM_SYSTEM_PROMPT,
            temperature = 0.7,
            max_tokens  = 256,
            timeout_seconds = 60.0,
            fallback_op = Arborist.SubtreeMutation()
        )
        tracked_llm = TrackedMutation(llm_op)

        llm_ops = [tracked_llm, Arborist.SubtreeMutation(), Arborist.PointMutation(),
                   Arborist.HoistMutation(), Arborist.ExpansionMutation()]

        r_a2 = run_bin_packing(; common_kwargs...,
            mutation_ops = llm_ops,
            output_file = "bin_packing_results_a2.md",
        )
    end

    # --- Write combined results ---
    results_path = joinpath(@__DIR__, "bin_packing_results_llm.md")
    open(results_path, "w") do io
        println(io, "# Experiment A: LLM Operator Impact (Uniform Distribution)")
        println(io, "\nGenerated: $(Dates.now())")
        println(io, "\n## Configuration")
        println(io, "- Population: $pop_size, Generations: $generations")
        println(io, "- Mutation rate: 0.4, Crossover rate: 0.3")
        println(io, "- Bloat penalty: 0.0005, Tournament size: 3, Elitism: 3")
        println(io, "- Episodes: 20, Items: 200, Distribution: uniform, Seed: 42")

        println(io, "\n## Results")
        println(io, "\n| Config | Best (train) | Best (test) | vs FF (test) | vs BF (test) | LLM fallback rate | Time |")
        println(io, "|---|---|---|---|---|---|---|")

        # A1 row
        vs_ff_a1 = round((r_a1.ff_test - r_a1.evolved_test) / r_a1.ff_test * 100, digits=2)
        vs_bf_a1 = round((r_a1.bf_test - r_a1.evolved_test) / r_a1.bf_test * 100, digits=2)
        println(io, "| Classical only (A1) | $(round(r_a1.result.best_fitness, digits=4)) | " *
                "$(round(r_a1.evolved_test, digits=4)) | $(vs_ff_a1)% | $(vs_bf_a1)% | N/A | $(round(r_a1.wall_time, digits=1))s |")

        if r_a2 !== nothing && tracked_llm !== nothing
            vs_ff_a2 = round((r_a2.ff_test - r_a2.evolved_test) / r_a2.ff_test * 100, digits=2)
            vs_bf_a2 = round((r_a2.bf_test - r_a2.evolved_test) / r_a2.bf_test * 100, digits=2)
            fallback_rate = tracked_llm.n_calls > 0 ?
                round((tracked_llm.n_calls - tracked_llm.n_slow) / tracked_llm.n_calls * 100, digits=1) : 0.0
            mean_latency = tracked_llm.n_calls > 0 ?
                round(tracked_llm.total_latency / tracked_llm.n_calls, digits=2) : 0.0
            println(io, "| Classical + Qwen3-Coder (A2) | $(round(r_a2.result.best_fitness, digits=4)) | " *
                    "$(round(r_a2.evolved_test, digits=4)) | $(vs_ff_a2)% | $(vs_bf_a2)% | " *
                    "$(fallback_rate)% | $(round(r_a2.wall_time, digits=1))s |")
        else
            println(io, "| Classical + Qwen3-Coder (A2) | — | — | — | — | — | Ollama unavailable |")
        end

        println(io, "\n## Baselines (test set)")
        println(io, "- First Fit: $(round(r_a1.ff_test, digits=4))")
        println(io, "- Best Fit: $(round(r_a1.bf_test, digits=4))")

        if tracked_llm !== nothing
            println(io, "\n## LLM Operator Metrics")
            println(io, "- Total LLM calls: $(tracked_llm.n_calls)")
            println(io, "- Successful parses (>0.5s): $(tracked_llm.n_slow)")
            fallback_n = tracked_llm.n_calls - tracked_llm.n_slow
            fallback_pct = tracked_llm.n_calls > 0 ?
                round(fallback_n / tracked_llm.n_calls * 100, digits=1) : 0.0
            println(io, "- Fallback count: $fallback_n ($fallback_pct%)")
            mean_lat = tracked_llm.n_calls > 0 ?
                round(tracked_llm.total_latency / tracked_llm.n_calls, digits=2) : 0.0
            println(io, "- Mean call latency: $(mean_lat)s")
            println(io, "- Total LLM time: $(round(tracked_llm.total_latency, digits=1))s")
        end

        println(io, "\n## Best Evolved Programs")
        println(io, "\n### A1: Classical Only")
        println(io, "```julia")
        println(io, r_a1.program_text)
        println(io, "```")

        if r_a2 !== nothing
            println(io, "\n### A2: Classical + LLM")
            println(io, "```julia")
            println(io, r_a2.program_text)
            println(io, "```")
        end
    end
    println("\nExperiment A results saved to: $results_path")
    flush(stdout)
end

# =============================================================================
# Experiment B: Bimodal Item Distribution
# =============================================================================

function run_experiment_b(; generations::Int=100, pop_size::Int=200)
    println("=" ^ 70)
    println("EXPERIMENT B: Bimodal Item Distribution")
    println("=" ^ 70)
    flush(stdout)

    common_kwargs = Dict{Symbol,Any}(
        :pop_size => pop_size, :generations => generations,
        :mutation_rate => 0.4, :crossover_rate => 0.3,
        :elitism => 3,
        :bloat_penalty => 0.0005,
        :n_episodes => 20, :n_items => 200, :rng_seed => 42,
    )

    # --- Bimodal baselines ---
    bimodal_eval = BinPackingEvaluator(n_episodes=20, n_items=200,
        capacity=1.0f0, item_dist=:bimodal, rng_seed=42)
    ff_bimodal = baseline_normalized(first_fit, bimodal_eval)
    bf_bimodal = baseline_normalized(best_fit, bimodal_eval)
    println("\nBimodal baselines:")
    println("  First Fit: $(round(ff_bimodal, digits=4))")
    println("  Best Fit:  $(round(bf_bimodal, digits=4))")
    flush(stdout)

    # --- B1: Uniform distribution baseline ---
    println("\n>>> Variant B1: Uniform distribution (reference)")
    flush(stdout)
    r_b1 = run_bin_packing(; common_kwargs...,
        item_dist = :uniform,
        output_file = "bin_packing_results_b1.md",
    )

    # --- B2: Bimodal, no speciation ---
    println("\n>>> Variant B2: Bimodal, no speciation")
    flush(stdout)
    r_b2 = run_bin_packing(; common_kwargs...,
        item_dist = :bimodal,
        output_file = "bin_packing_results_b2.md",
    )

    # --- B3: Bimodal, behavioral speciation ---
    println("\n>>> Variant B3: Bimodal, behavioral speciation")
    flush(stdout)
    r_b3 = run_bin_packing(; common_kwargs...,
        item_dist = :bimodal,
        speciation = bp_behavioral_speciation(threshold=0.15, sharing_formula=:sqrt),
        output_file = "bin_packing_results_b3.md",
    )

    # --- Write combined results ---
    results_path = joinpath(@__DIR__, "bin_packing_results_bimodal.md")
    open(results_path, "w") do io
        println(io, "# Experiment B: Bimodal Item Distribution")
        println(io, "\nGenerated: $(Dates.now())")
        println(io, "\n## Configuration")
        println(io, "- Population: $pop_size, Generations: $generations")
        println(io, "- Mutation rate: 0.4, Crossover rate: 0.3")
        println(io, "- Bloat penalty: 0.0005, Tournament size: 3, Elitism: 3")
        println(io, "- Episodes: 20, Items: 200, Seed: 42")

        println(io, "\n## Bimodal Baselines")
        println(io, "- First Fit: $(round(ff_bimodal, digits=4))")
        println(io, "- Best Fit: $(round(bf_bimodal, digits=4))")
        println(io, "- Gap (FF-BF): $(round(ff_bimodal - bf_bimodal, digits=4))")

        println(io, "\n## Results")
        println(io, "\n| Config | Distribution | Best (train) | Best (test) | vs FF (test) | vs BF (test) | Time |")
        println(io, "|---|---|---|---|---|---|---|")

        for (name, r, dist) in [("No speciation (B1)", r_b1, "uniform"),
                                 ("No speciation (B2)", r_b2, "bimodal"),
                                 ("Behavioral spec (B3)", r_b3, "bimodal")]
            vs_ff = round((r.ff_test - r.evolved_test) / r.ff_test * 100, digits=2)
            vs_bf = round((r.bf_test - r.evolved_test) / r.bf_test * 100, digits=2)
            println(io, "| $name | $dist | $(round(r.result.best_fitness, digits=4)) | " *
                    "$(round(r.evolved_test, digits=4)) | $(vs_ff)% | $(vs_bf)% | $(round(r.wall_time, digits=1))s |")
        end

        println(io, "\n## Best Evolved Programs")
        for (name, r) in [("B1: Uniform", r_b1), ("B2: Bimodal", r_b2), ("B3: Bimodal + Behavioral Speciation", r_b3)]
            println(io, "\n### $name")
            println(io, "```julia")
            println(io, r.program_text)
            println(io, "```")
        end
    end
    println("\nExperiment B results saved to: $results_path")
    flush(stdout)
end

# =============================================================================
# Group 1: Extended LLM run (300 generations)
# =============================================================================

function run_extended_classical(; generations::Int=300, pop_size::Int=200)
    println("=" ^ 70)
    println("G1A: Classical GP, $generations generations (extended)")
    println("=" ^ 70)
    flush(stdout)

    r = run_bin_packing(; _common_kwargs(seed=42, generations=generations, pop_size=pop_size)...,
        mutation_ops=_classical_ops(),
        output_file="bin_packing_results_g1a.md",
    )

    # Write fitness history to data file
    data_dir = _ensure_data_dir()
    open(joinpath(data_dir, "extended_classical_history.tsv"), "w") do io
        println(io, "generation\tbest_fitness")
        for (gen, fit) in enumerate(r.result.fitness_history)
            println(io, "$gen\t$(round(fit, digits=6))")
        end
    end

    results_path = joinpath(@__DIR__, "bin_packing_results_extended_classical.md")
    open(results_path, "w") do io
        println(io, "# G1A: Extended Classical GP ($generations generations)\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "## Result")
        println(io, "- Train fitness: $(round(r.result.best_fitness, digits=4))")
        println(io, "- Test fitness: $(round(r.evolved_test, digits=4))")
        vs_bf = round((r.bf_test - r.evolved_test) / r.bf_test * 100, digits=2)
        println(io, "- vs Best Fit (test): $(vs_bf)%")
        println(io, "- Wall time: $(round(r.wall_time, digits=1))s\n")
        println(io, "## Fitness History (every 10 gens)")
        println(io, "| Gen | Best |")
        println(io, "|-----|------|")
        for gen in 1:10:length(r.result.fitness_history)
            println(io, "| $gen | $(round(r.result.fitness_history[gen], digits=4)) |")
        end
        if length(r.result.fitness_history) % 10 != 1
            println(io, "| $(length(r.result.fitness_history)) | $(round(r.result.fitness_history[end], digits=4)) |")
        end
        println(io, "\n## Best Program\n```julia")
        println(io, r.program_text)
        println(io, "```")
    end
    println("G1A results saved to: $results_path")
    flush(stdout)
    return r
end

function run_extended_llm(; generations::Int=300, pop_size::Int=200)
    println("=" ^ 70)
    println("G1B: Classical + LLM GP, $generations generations (extended)")
    println("=" ^ 70)
    flush(stdout)

    if !_check_ollama()
        open(joinpath(@__DIR__, "bin_packing_results_extended_llm.md"), "w") do io
            println(io, "# G1B: SKIPPED — Ollama unavailable")
        end
        return nothing
    end

    tracked = _make_tracked_llm()
    llm_ops = [tracked; _classical_ops()]
    r = run_bin_packing(; _common_kwargs(seed=42, generations=generations, pop_size=pop_size)...,
        mutation_ops=llm_ops,
        output_file="bin_packing_results_g1b.md",
    )

    # Write fitness history
    data_dir = _ensure_data_dir()
    open(joinpath(data_dir, "extended_llm_history.tsv"), "w") do io
        println(io, "generation\tbest_fitness")
        for (gen, fit) in enumerate(r.result.fitness_history)
            println(io, "$gen\t$(round(fit, digits=6))")
        end
    end

    fallback_rate = tracked.n_calls > 0 ?
        round((tracked.n_calls - tracked.n_slow) / tracked.n_calls * 100, digits=1) : 0.0
    mean_lat = tracked.n_calls > 0 ?
        round(tracked.total_latency / tracked.n_calls, digits=2) : 0.0

    results_path = joinpath(@__DIR__, "bin_packing_results_extended_llm.md")
    open(results_path, "w") do io
        println(io, "# G1B: Extended Classical + LLM GP ($generations generations)\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "## Result")
        println(io, "- Train fitness: $(round(r.result.best_fitness, digits=4))")
        println(io, "- Test fitness: $(round(r.evolved_test, digits=4))")
        vs_bf = round((r.bf_test - r.evolved_test) / r.bf_test * 100, digits=2)
        println(io, "- vs Best Fit (test): $(vs_bf)%")
        println(io, "- Wall time: $(round(r.wall_time, digits=1))s\n")
        println(io, "## LLM Metrics")
        println(io, "- Total calls: $(tracked.n_calls)")
        println(io, "- Successful: $(tracked.n_slow) ($(round(100.0 - fallback_rate, digits=1))%)")
        println(io, "- Fallback rate: $(fallback_rate)%")
        println(io, "- Mean latency: $(mean_lat)s")
        println(io, "- Total LLM time: $(round(tracked.total_latency, digits=1))s\n")
        println(io, "## Fitness History (every 10 gens)")
        println(io, "| Gen | Best |")
        println(io, "|-----|------|")
        for gen in 1:10:length(r.result.fitness_history)
            println(io, "| $gen | $(round(r.result.fitness_history[gen], digits=4)) |")
        end
        if length(r.result.fitness_history) % 10 != 1
            println(io, "| $(length(r.result.fitness_history)) | $(round(r.result.fitness_history[end], digits=4)) |")
        end
        println(io, "\n## Best Program\n```julia")
        println(io, r.program_text)
        println(io, "```")
    end
    println("G1B results saved to: $results_path")
    flush(stdout)
    return r
end

# =============================================================================
# Group 2: Multi-seed comparison (5 seeds × 4 configs)
# =============================================================================

function _run_multiseed(label::String;
                        make_ops_fn, speciation_fn=() -> Arborist.NoSpeciation(),
                        generations::Int=100, pop_size::Int=200)
    seeds = OVERNIGHT_SEEDS
    rows = NamedTuple[]

    for (i, seed) in enumerate(seeds)
        println("\n--- $label: Seed $seed ($i/$(length(seeds))) ---")
        flush(stdout)

        ops = make_ops_fn()
        tracked = nothing
        for op in ops
            if op isa TrackedMutation
                tracked = op
                break
            end
        end

        r = run_bin_packing(; _common_kwargs(seed=seed, generations=generations, pop_size=pop_size)...,
            mutation_ops=ops,
            speciation=speciation_fn(),
            output_file="bin_packing_results_$(label)_seed$(seed).md",
        )

        push!(rows, (
            seed=seed,
            train=r.result.best_fitness,
            test=r.evolved_test,
            wall_time=r.wall_time,
            ff_test=r.ff_test,
            bf_test=r.bf_test,
            llm_calls=tracked === nothing ? 0 : tracked.n_calls,
            llm_ok=tracked === nothing ? 0 : tracked.n_slow,
            llm_latency=tracked === nothing ? 0.0 : tracked.total_latency,
            program=r.program_text,
        ))
    end

    _write_seed_data("$(label).tsv", rows)

    # Summary markdown
    test_fits = [r.test for r in rows]
    bf_tests = [r.bf_test for r in rows]
    beats_bf = sum(t < b for (t, b) in zip(test_fits, bf_tests))
    mean_test = Statistics.mean(test_fits)
    std_test = length(test_fits) > 1 ? Statistics.std(test_fits) : 0.0

    results_path = joinpath(@__DIR__, "bin_packing_results_$(label).md")
    open(results_path, "w") do io
        println(io, "# Multi-seed Results: $label\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "## Summary")
        println(io, "- Mean test fitness: $(round(mean_test, digits=4)) ± $(round(std_test, digits=4))")
        println(io, "- Min: $(round(minimum(test_fits), digits=4)), Max: $(round(maximum(test_fits), digits=4))")
        mean_bf = Statistics.mean(bf_tests)
        vs_bf = round((mean_bf - mean_test) / mean_bf * 100, digits=2)
        println(io, "- vs BF (mean): $(vs_bf)%")
        println(io, "- Seeds beating BF: $beats_bf/$(length(seeds))")
        total_time = sum(r.wall_time for r in rows)
        println(io, "- Total wall time: $(round(total_time, digits=1))s")

        if any(r.llm_calls > 0 for r in rows)
            total_calls = sum(r.llm_calls for r in rows)
            total_ok = sum(r.llm_ok for r in rows)
            total_lt = sum(r.llm_latency for r in rows)
            fb = total_calls > 0 ? round((total_calls - total_ok) / total_calls * 100, digits=1) : 0.0
            println(io, "\n## LLM Metrics (aggregate)")
            println(io, "- Total calls: $total_calls, Successful: $total_ok, Fallback: $(fb)%")
            println(io, "- Total LLM time: $(round(total_lt, digits=1))s")
        end

        println(io, "\n## Per-seed Results")
        println(io, "| Seed | Train | Test | vs BF | Time |")
        println(io, "|------|-------|------|-------|------|")
        for r in rows
            vs = round((r.bf_test - r.test) / r.bf_test * 100, digits=2)
            println(io, "| $(r.seed) | $(round(r.train, digits=4)) | $(round(r.test, digits=4)) | $(vs)% | $(round(r.wall_time, digits=1))s |")
        end

        println(io, "\n## Best Program (best seed)")
        best_idx = argmin(test_fits)
        println(io, "Seed $(rows[best_idx].seed), test=$(round(rows[best_idx].test, digits=4))")
        println(io, "```julia")
        println(io, rows[best_idx].program)
        println(io, "```")
    end
    println("\n$label results saved to: $results_path")
    flush(stdout)
    return rows
end

function run_multiseed_classical()
    println("=" ^ 70)
    println("G2A: Multi-seed Classical GP ($(length(OVERNIGHT_SEEDS)) seeds × 100 gen)")
    println("=" ^ 70)
    flush(stdout)
    _run_multiseed("multiseed_classical", make_ops_fn=_classical_ops)
end

function run_multiseed_behavioral()
    println("=" ^ 70)
    println("G2B: Multi-seed Behavioral Speciation ($(length(OVERNIGHT_SEEDS)) seeds × 100 gen)")
    println("=" ^ 70)
    flush(stdout)
    _run_multiseed("multiseed_behavioral",
        make_ops_fn=_classical_ops,
        speciation_fn=() -> bp_behavioral_speciation(threshold=0.15, sharing_formula=:sqrt))
end

function run_multiseed_llm()
    println("=" ^ 70)
    println("G2C: Multi-seed Classical + LLM ($(length(OVERNIGHT_SEEDS)) seeds × 100 gen)")
    println("=" ^ 70)
    flush(stdout)

    if !_check_ollama()
        open(joinpath(@__DIR__, "bin_packing_results_multiseed_llm.md"), "w") do io
            println(io, "# G2C: SKIPPED — Ollama unavailable")
        end
        return nothing
    end

    _run_multiseed("multiseed_llm",
        make_ops_fn=() -> [_make_tracked_llm(); _classical_ops()])
end

function run_template_baseline()
    println("=" ^ 70)
    println("G2D: Template Baseline (no evolution)")
    println("=" ^ 70)
    flush(stdout)

    seeds = OVERNIGHT_SEEDS
    fset = bin_packing_function_set()
    rng = Random.MersenneTwister(42)
    state = _bp_create_state(rng, fset)

    # Best-fit seed template (type 4 — clean best-fit scanner)
    template_body = _bp_seeded_body(state, 4)
    genome = Arborist.ExprGenome(template_body, state)

    _ensure_bp_states()
    f = _bp_compile(genome)
    if f === nothing
        println("ERROR: Template compilation failed")
        flush(stdout)
        return nothing
    end

    rows = NamedTuple[]
    for seed in seeds
        test_eval = BinPackingEvaluator(n_episodes=20, n_items=200,
            capacity=1.0f0, item_dist=:uniform, rng_seed=seed + 1000)
        test_ff = baseline_normalized(first_fit, test_eval)
        test_bf = baseline_normalized(best_fit, test_eval)
        test_template = Arborist.evaluate(test_eval, f)
        push!(rows, (
            seed=seed, train=test_template, test=test_template,
            wall_time=0.0, ff_test=test_ff, bf_test=test_bf,
            llm_calls=0, llm_ok=0, llm_latency=0.0,
            program=pretty_print_genome(genome),
        ))
        vs_bf = round((test_bf - test_template) / test_bf * 100, digits=2)
        println("  Seed $seed: template=$(round(test_template, digits=4)) bf=$(round(test_bf, digits=4)) vs_bf=$(vs_bf)%")
        flush(stdout)
    end

    _write_seed_data("multiseed_template.tsv", rows)

    test_fits = [r.test for r in rows]
    bf_tests = [r.bf_test for r in rows]
    beats_bf = sum(t < b for (t, b) in zip(test_fits, bf_tests))
    mean_test = Statistics.mean(test_fits)
    std_test = length(test_fits) > 1 ? Statistics.std(test_fits) : 0.0

    results_path = joinpath(@__DIR__, "bin_packing_results_multiseed_template.md")
    open(results_path, "w") do io
        println(io, "# Multi-seed Results: Template Baseline (no evolution)\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "## Summary")
        println(io, "- Mean test fitness: $(round(mean_test, digits=4)) ± $(round(std_test, digits=4))")
        println(io, "- Seeds beating BF: $beats_bf/$(length(seeds))\n")
        println(io, "## Per-seed Results")
        println(io, "| Seed | Template | FF | BF | vs BF |")
        println(io, "|------|----------|-----|-----|-------|")
        for r in rows
            vs = round((r.bf_test - r.test) / r.bf_test * 100, digits=2)
            println(io, "| $(r.seed) | $(round(r.test, digits=4)) | $(round(r.ff_test, digits=4)) | $(round(r.bf_test, digits=4)) | $(vs)% |")
        end
        println(io, "\n## Template Program\n```julia")
        println(io, rows[1].program)
        println(io, "```")
    end
    println("Template baseline results saved to: $results_path")
    flush(stdout)
    return rows
end

# =============================================================================
# Group 3: Time-normalized comparison
# =============================================================================

function run_timenorm_classical(; pop_size::Int=200)
    target_gens = 800
    println("=" ^ 70)
    println("G3A: Time-normalized Classical GP ($target_gens generations)")
    println("  Target wall time: ~1800s (matching LLM variant A2)")
    println("=" ^ 70)
    flush(stdout)

    r = run_bin_packing(; _common_kwargs(seed=42, generations=target_gens, pop_size=pop_size)...,
        mutation_ops=_classical_ops(),
        output_file="bin_packing_results_g3a.md",
    )

    data_dir = _ensure_data_dir()
    open(joinpath(data_dir, "timenorm_classical.tsv"), "w") do io
        println(io, "generations\ttrain\ttest\twall_time\tff_test\tbf_test")
        println(io, "$target_gens\t$(round(r.result.best_fitness, digits=6))\t$(round(r.evolved_test, digits=6))\t$(round(r.wall_time, digits=1))\t$(round(r.ff_test, digits=6))\t$(round(r.bf_test, digits=6))")
    end

    results_path = joinpath(@__DIR__, "bin_packing_results_timenorm.md")
    vs_bf = round((r.bf_test - r.evolved_test) / r.bf_test * 100, digits=2)
    open(results_path, "w") do io
        println(io, "# Group 3: Time-Normalized Comparison\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "## Comparison (same wall-clock budget)")
        println(io, "| Config | Generations | Test fitness | vs BF | Wall time |")
        println(io, "|--------|------------|-------------|-------|-----------|")
        println(io, "| Classical (A1, ref) | 100 | 1.0782 | 0.05% | 218s |")
        println(io, "| Classical + LLM (A2, ref) | 100 | 1.0679 | 1.00% | 1767s |")
        println(io, "| Classical time-matched (G3A) | $target_gens | $(round(r.evolved_test, digits=4)) | $(vs_bf)% | $(round(r.wall_time, digits=1))s |")
        println(io, "\n## Best Program (G3A)\n```julia")
        println(io, r.program_text)
        println(io, "```")
    end
    println("G3A results saved to: $results_path")
    flush(stdout)
    return r
end

# =============================================================================
# Combine all overnight results
# =============================================================================

function run_combine_results()
    println("=" ^ 70)
    println("Combining overnight results")
    println("=" ^ 70)
    flush(stdout)

    results_path = joinpath(@__DIR__, "bin_packing_overnight_results.md")
    open(results_path, "w") do io
        println(io, "# Arborist.jl — Overnight Bin Packing Experiment Results\n")
        println(io, "Generated: $(Dates.now())")

        # --- Multi-seed summary ---
        println(io, "\n## 1. Multi-seed Comparison (5 seeds: $(join(OVERNIGHT_SEEDS, ", ")))\n")
        println(io, "| Config | Mean test | Std | Min | Max | vs BF (mean) | Beats BF |")
        println(io, "|--------|-----------|-----|-----|-----|-------------|----------|")

        configs = [
            ("Best-fit template", "multiseed_template.tsv"),
            ("Classical GP", "multiseed_classical.tsv"),
            ("Behavioral speciation", "multiseed_behavioral.tsv"),
            ("Classical + LLM", "multiseed_llm.tsv"),
        ]
        per_seed = Dict{String, Vector}()
        for (name, file) in configs
            rows = _read_seed_data(file)
            if rows === nothing || isempty(rows)
                println(io, "| $name | — | — | — | — | — | — |")
                continue
            end
            per_seed[name] = rows
            tests = [r.test for r in rows]
            bfs = [r.bf_test for r in rows]
            m = round(Statistics.mean(tests), digits=4)
            s = round(Statistics.std(tests), digits=4)
            vs = round((Statistics.mean(bfs) - Statistics.mean(tests)) / Statistics.mean(bfs) * 100, digits=2)
            beats = sum(t < b for (t, b) in zip(tests, bfs))
            println(io, "| $name | $m | $s | $(round(minimum(tests),digits=4)) | $(round(maximum(tests),digits=4)) | $(vs)% | $beats/$(length(tests)) |")
        end

        println(io, "\n### Per-seed Detail\n")
        println(io, "| Seed | Template | Classical | Behavioral | LLM |")
        println(io, "|------|----------|-----------|------------|-----|")
        for seed in OVERNIGHT_SEEDS
            parts = [string(seed)]
            for (name, _) in configs
                rows = get(per_seed, name, nothing)
                if rows === nothing
                    push!(parts, "—")
                else
                    idx = findfirst(r -> r.seed == seed, rows)
                    push!(parts, idx === nothing ? "—" : string(round(rows[idx].test, digits=4)))
                end
            end
            println(io, "| ", join(parts, " | "), " |")
        end

        # --- Extended run ---
        println(io, "\n## 2. Extended Run (300 generations, seed 42)\n")
        data_dir = _ensure_data_dir()
        for (label, file) in [("Classical (G1A)", "extended_classical_history.tsv"),
                               ("Classical + LLM (G1B)", "extended_llm_history.tsv")]
            path = joinpath(data_dir, file)
            if isfile(path)
                lines = readlines(path)
                data = Tuple{Int,Float64}[]
                for l in lines
                    startswith(l, "gen") && continue
                    ps = split(l, "\t")
                    length(ps) >= 2 || continue
                    push!(data, (parse(Int, ps[1]), parse(Float64, ps[2])))
                end
                if !isempty(data)
                    println(io, "### $label")
                    println(io, "| Gen | Best |")
                    println(io, "|-----|------|")
                    for (gen, fit) in data
                        if gen == 1 || gen % 10 == 0 || gen == length(data)
                            println(io, "| $gen | $(round(fit, digits=4)) |")
                        end
                    end
                    println(io, "")
                end
            else
                println(io, "### $label — data not found\n")
            end
        end

        # --- Time-normalized ---
        println(io, "\n## 3. Time-Normalized Comparison\n")
        tn_path = joinpath(@__DIR__, "bin_packing_results_timenorm.md")
        if isfile(tn_path)
            for line in readlines(tn_path)
                startswith(line, "# ") && continue  # skip title
                startswith(line, "Generated:") && continue
                println(io, line)
            end
        else
            println(io, "Time-normalized results not yet available.")
        end

        println(io, "\n## 4. Key Findings\n")
        println(io, "*To be completed after reviewing all results.*")
    end

    println("Combined results saved to: $results_path")
    flush(stdout)
end

# =============================================================================
# Prompt enrichment ablation experiment
# =============================================================================

const ABLATION_VARIANTS = Dict{String, Vector{Arborist.AbstractPromptSection}}(
    "baseline"     => Arborist.AbstractPromptSection[],
    "fitness_only" => Arborist.AbstractPromptSection[Arborist.FitnessSection()],
    "elites_3"     => Arborist.AbstractPromptSection[Arborist.ElitesSection(3)],
    "full"         => Arborist.AbstractPromptSection[
        Arborist.FitnessSection(), Arborist.ElitesSection(3), Arborist.GenerationSection()],
)

function _make_ablation_llm(sections::Vector{Arborist.AbstractPromptSection})
    llm_op = Arborist.LLMMutationOperator(
        endpoint    = "http://localhost:11434/v1/chat/completions",
        model       = "qwen3-coder:30b",
        api_key_env = "",
        system_prompt = BP_LLM_SYSTEM_PROMPT,
        temperature = 0.7,
        max_tokens  = 256,
        timeout_seconds = 60.0,
        fallback_op = Arborist.SubtreeMutation(),
        sections    = sections,
    )
    return TrackedMutation(llm_op)
end

"""
Run prompt enrichment ablation: 4 variants × 5 seeds.
Pass `--variant=<name>` to run a single variant, or omit to run all.
"""
function run_ablation_enrichment(; variant::Union{String,Nothing}=nothing)
    println("=" ^ 70)
    println("Prompt Enrichment Ablation Experiment")
    println("=" ^ 70)
    flush(stdout)

    if !_check_ollama()
        println("ERROR: Ollama not reachable. Aborting ablation.")
        return
    end

    variants = if variant !== nothing
        if !haskey(ABLATION_VARIANTS, variant)
            error("Unknown variant: $variant. Available: $(join(keys(ABLATION_VARIANTS), ", "))")
        end
        [variant]
    else
        sort(collect(keys(ABLATION_VARIANTS)))
    end

    all_rows = NamedTuple[]
    data_dir = _ensure_data_dir()

    for vname in variants
        sections = ABLATION_VARIANTS[vname]
        println("\n" * "=" ^ 70)
        println("Variant: $vname ($(length(sections)) sections)")
        println("=" ^ 70)
        flush(stdout)

        for (i, seed) in enumerate(OVERNIGHT_SEEDS)
            println("\n--- $vname: Seed $seed ($i/$(length(OVERNIGHT_SEEDS))) ---")
            flush(stdout)

            tracked = _make_ablation_llm(sections)
            ops = [tracked, Arborist.SubtreeMutation(), Arborist.PointMutation(),
                   Arborist.HoistMutation(), Arborist.ExpansionMutation()]

            r = run_bin_packing(; _common_kwargs(seed=seed, generations=100, pop_size=200)...,
                mutation_ops=ops,
                output_file="bin_packing_results_ablation_$(vname)_seed$(seed).md",
            )

            push!(all_rows, (
                variant=vname,
                seed=seed,
                train=r.result.best_fitness,
                test=r.evolved_test,
                wall_time=r.wall_time,
                ff_test=r.ff_test,
                bf_test=r.bf_test,
                llm_calls=tracked.n_calls,
                llm_ok=tracked.n_slow,
                llm_latency=tracked.total_latency,
            ))
        end
    end

    # Write combined TSV
    tsv_path = joinpath(data_dir, "ablation_enrichment.tsv")
    open(tsv_path, "a") do io  # append so multi-variant runs accumulate
        for r in all_rows
            println(io, join([r.variant, r.seed, round(r.train, digits=6),
                              round(r.test, digits=6), round(r.wall_time, digits=1),
                              round(r.ff_test, digits=6), round(r.bf_test, digits=6),
                              r.llm_calls, r.llm_ok,
                              round(r.llm_latency, digits=1)], "\t"))
        end
    end
    println("\nData appended to: $tsv_path")

    # Print summary table
    println("\n" * "=" ^ 70)
    println("Ablation Results Summary")
    println("=" ^ 70)
    println("| Variant | Mean test | Std | Min | Max | BF beat | Seeds |")
    println("|---------|-----------|-----|-----|-----|---------|-------|")
    for vname in sort(collect(Set(r.variant for r in all_rows)))
        vrows = filter(r -> r.variant == vname, all_rows)
        tests = [r.test for r in vrows]
        bfs = [r.bf_test for r in vrows]
        n_beat = count(t < b for (t, b) in zip(tests, bfs))
        mu = sum(tests) / length(tests)
        sd = length(tests) > 1 ?
            sqrt(sum((t - mu)^2 for t in tests) / (length(tests) - 1)) : 0.0
        println("| $vname | $(round(mu, digits=4)) | $(round(sd, digits=4)) | " *
                "$(round(minimum(tests), digits=4)) | $(round(maximum(tests), digits=4)) | " *
                "$n_beat/$(length(tests)) | $(length(tests)) |")
    end
    flush(stdout)
end


# =============================================================================
# Main entry point
# =============================================================================

function main()
    experiment = nothing
    kwargs = Dict{Symbol, Any}()

    for arg in ARGS
        m = match(r"^--(\w+)=(.+)$", arg)
        if m !== nothing
            key = Symbol(m.captures[1])
            val_str = m.captures[2]
            if key == :experiment
                experiment = val_str
            elseif key in (:pop_size, :generations, :n_episodes, :n_items, :rng_seed,
                           :elitism, :tournament_size, :num_temps)
                kwargs[key] = parse(Int, val_str)
            elseif key in (:mutation_rate, :crossover_rate, :bloat_penalty)
                kwargs[key] = parse(Float64, val_str)
            elseif key == :capacity
                kwargs[key] = parse(Float32, val_str)
            elseif key == :item_dist
                kwargs[key] = Symbol(val_str)
            elseif key == :variant
                kwargs[key] = val_str
            elseif key == :verbose
                kwargs[key] = parse(Bool, val_str)
            elseif key == :speciation
                if val_str == "none"
                    kwargs[key] = Arborist.NoSpeciation()
                elseif val_str == "threshold"
                    kwargs[key] = Arborist.ThresholdSpeciation()
                elseif val_str == "behavioral"
                    kwargs[key] = bp_behavioral_speciation()
                end
            end
        end
    end

    if experiment === nothing
        run_bin_packing(; kwargs...)
        return
    end

    exp_kwargs = Dict{Symbol,Any}()
    haskey(kwargs, :generations) && (exp_kwargs[:generations] = kwargs[:generations])
    haskey(kwargs, :pop_size) && (exp_kwargs[:pop_size] = kwargs[:pop_size])

    # Fault tolerance: write error to results file on failure
    try
        if experiment == "llm"
            run_experiment_a(; exp_kwargs...)
        elseif experiment == "bimodal"
            run_experiment_b(; exp_kwargs...)
        elseif experiment == "extended_classical"
            run_extended_classical(; exp_kwargs...)
        elseif experiment == "extended_llm"
            run_extended_llm(; exp_kwargs...)
        elseif experiment == "multiseed_classical"
            run_multiseed_classical()
        elseif experiment == "multiseed_behavioral"
            run_multiseed_behavioral()
        elseif experiment == "multiseed_llm"
            run_multiseed_llm()
        elseif experiment == "template_baseline"
            run_template_baseline()
        elseif experiment == "timenorm_classical"
            run_timenorm_classical(; filter(p -> p.first == :pop_size, exp_kwargs)...)
        elseif experiment == "combine_results"
            run_combine_results()
        elseif experiment == "ablation_enrichment"
            run_ablation_enrichment(;
                variant=get(kwargs, :variant, nothing))
        else
            error("Unknown experiment: $experiment")
        end
    catch e
        output_path = joinpath(@__DIR__, "bin_packing_results_$(experiment)_FAILED.md")
        open(output_path, "w") do io
            println(io, "# EXPERIMENT FAILED: $experiment\n")
            println(io, "Error: $e\n")
            println(io, "Stacktrace:")
            Base.show_backtrace(io, catch_backtrace())
        end
        rethrow(e)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
