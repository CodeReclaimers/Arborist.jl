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

        elapsed = round(time() - t0, digits=1)
        if verbose
            n_species = species_state isa Vector ? length(species_state) : 0
            species_str = n_species > 0 ? " | species=$n_species" : ""
            println("Gen $gen/$(algorithm.generations) | best=$(round(fitnesses[1], digits=4)) | " *
                    "mean=$(round(mean_fit, digits=4))$species_str | elapsed=$(elapsed)s")
            flush(stdout)
        end

        callback !== nothing && callback(gen, fitnesses[1], genomes[1])

        next_genomes = Vector{Arborist.ExprGenome}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        t_size = algorithm.selection isa Arborist.TournamentSelection ?
                 algorithm.selection.tournament_size : algorithm.tournament_size

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
        tournament_size::Int = 5,
        bloat_penalty::Float64 = 0.001,
        speciation::Arborist.AbstractSpeciation = Arborist.ThresholdSpeciation(
            threshold=10.0, min_species_size=2, stagnation_limit=15),
        n_episodes::Int = 20,
        n_items::Int = 200,
        capacity::Float32 = 1.0f0,
        item_dist::Symbol = :uniform,
        rng_seed::Int = 42,
        num_temps::Int = 6,
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

    spec_desc = speciation isa Arborist.NoSpeciation ? "none" :
        "threshold=$(speciation.threshold), stagnation=$(speciation.stagnation_limit)"

    algorithm = Arborist.GeneticProgramming(
        pop_size = pop_size,
        generations = generations,
        mutation_rate = mutation_rate,
        crossover_rate = crossover_rate,
        elitism = elitism,
        tournament_size = tournament_size,
        bloat_penalty = bloat_penalty,
        speciation = speciation,
    )

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
    results_path = joinpath(@__DIR__, "bin_packing_results.md")
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

    return result
end

function main()
    kwargs = Dict{Symbol, Any}()
    for arg in ARGS
        m = match(r"^--(\w+)=(.+)$", arg)
        if m !== nothing
            key = Symbol(m.captures[1])
            val_str = m.captures[2]
            if key in (:pop_size, :generations, :n_episodes, :n_items, :rng_seed,
                        :elitism, :tournament_size, :num_temps)
                kwargs[key] = parse(Int, val_str)
            elseif key in (:mutation_rate, :crossover_rate, :bloat_penalty)
                kwargs[key] = parse(Float64, val_str)
            elseif key == :capacity
                kwargs[key] = parse(Float32, val_str)
            elseif key == :item_dist
                kwargs[key] = Symbol(val_str)
            elseif key == :verbose
                kwargs[key] = parse(Bool, val_str)
            end
        end
    end

    run_bin_packing(; kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
