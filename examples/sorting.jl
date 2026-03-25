#!/usr/bin/env julia
# examples/sorting.jl — Evolving a sorting algorithm via GP.
#
# Demonstrates evolving a comparison-based sorting algorithm using the
# Arborist.jl ExprGenome with a side-effectful array simulator, following
# the AntGenome/BinPacking pattern.
#
# Usage:
#   julia --project examples/sorting.jl
#   julia --project examples/sorting.jl --generations=500 --pop_size=300

using Arborist
using Dates
using Random
using Statistics

# =============================================================================
# Array sorting simulator (module-level mutable state)
# =============================================================================

mutable struct SortState
    arr::Vector{Int32}     # the array being sorted
    n::Int32               # array length
    comparisons::Int32     # number of comparisons made (for analysis)
    swaps::Int32           # number of swaps made (for analysis)
    op_count::Int32        # total operations (for bloat analysis)
end

const _sort_state = Ref{SortState}(
    SortState(Int32[], Int32(0), Int32(0), Int32(0), Int32(0))
)

function reset_sort_state!(arr::Vector{Int32})
    s = _sort_state[]
    s.arr = copy(arr)
    s.n = Int32(length(arr))
    s.comparisons = Int32(0)
    s.swaps = Int32(0)
    s.op_count = Int32(0)
    return nothing
end

# =============================================================================
# Primitives callable from @eval'd evolved programs
# =============================================================================

"""Get element at index i (1-indexed, clamped to [1, n])."""
function sort_get(i::Int32)::Int32
    s = _sort_state[]
    s.op_count += Int32(1)
    idx = clamp(Int(i), 1, Int(s.n))
    return s.arr[idx]
end

"""Array length."""
function sort_n()::Int32
    s = _sort_state[]
    s.op_count += Int32(1)
    return s.n
end

"""Compare arr[i] < arr[j] (clamped indices)."""
function sort_less(i::Int32, j::Int32)::Bool
    s = _sort_state[]
    s.op_count += Int32(1)
    s.comparisons += Int32(1)
    ci = clamp(Int(i), 1, Int(s.n))
    cj = clamp(Int(j), 1, Int(s.n))
    return s.arr[ci] < s.arr[cj]
end

"""Swap elements at indices i and j. Returns true if swap occurred."""
function sort_swap!(i::Int32, j::Int32)::Bool
    s = _sort_state[]
    s.op_count += Int32(1)
    ci = clamp(Int(i), 1, Int(s.n))
    cj = clamp(Int(j), 1, Int(s.n))
    ci == cj && return false
    s.arr[ci], s.arr[cj] = s.arr[cj], s.arr[ci]
    s.swaps += Int32(1)
    return true
end

# =============================================================================
# Fitness helpers
# =============================================================================

"""Normalized inversion count: fraction of pairs that are out of order.
0.0 = sorted, 1.0 = reverse sorted, ~0.5 = random."""
function inversion_count(arr::Vector{Int32})::Float64
    n = length(arr)
    n <= 1 && return 0.0
    count = 0
    for i in 1:n-1
        for j in i+1:n
            arr[i] > arr[j] && (count += 1)
        end
    end
    return count / (n * (n - 1) / 2)
end

# =============================================================================
# SortingEvaluator
# =============================================================================

struct SortingEvaluator <: Arborist.AbstractEvaluator
    n_episodes::Int        # number of test arrays per evaluation
    array_length::Int      # length of each test array
    value_range::Int       # values in [-value_range, value_range]
    rng_seed::Int          # for reproducible test arrays
    partial_credit::Bool   # whether to give partial credit via inversion count
    loop_limit::Int        # loop iteration limit for add_loop_checks
    comparison_alpha::Float64  # weight for comparison count penalty (0 = no penalty)
end

function SortingEvaluator(;
        n_episodes::Int=30,
        array_length::Int=8,
        value_range::Int=100,
        rng_seed::Int=42,
        partial_credit::Bool=true,
        loop_limit::Int=640,
        comparison_alpha::Float64=0.0)
    SortingEvaluator(n_episodes, array_length, value_range, rng_seed,
                     partial_credit, loop_limit, comparison_alpha)
end

Arborist.input_signature(::SortingEvaluator) = Dict{Symbol,DataType}()
Arborist.output_signature(::SortingEvaluator) = Dict(:done => Bool)

function Arborist.evaluate(e::SortingEvaluator, f::Function)
    total_score = 0.0
    total_comp_penalty = 0.0
    n = e.array_length
    nlogn = n * log2(n)
    for ep in 1:e.n_episodes
        ep_rng = Random.MersenneTwister(e.rng_seed + ep)
        arr = rand(ep_rng, Int32(-e.value_range):Int32(e.value_range), n)
        reset_sort_state!(arr)
        try
            Base.invokelatest(f)
        catch
            # program threw (e.g. LoopLimitExceeded) — score whatever state we have
        end
        sorted = issorted(_sort_state[].arr)
        if e.partial_credit
            total_score += inversion_count(_sort_state[].arr)
        else
            total_score += sorted ? 0.0 : 1.0
        end
        # Only reward comparison efficiency on correctly sorted episodes.
        # This prevents evolution from trading correctness for fewer comparisons.
        if e.comparison_alpha > 0.0 && sorted
            total_comp_penalty += Float64(_sort_state[].comparisons) / nlogn
        end
    end
    fitness = total_score / e.n_episodes
    if e.comparison_alpha > 0.0
        fitness += e.comparison_alpha * (total_comp_penalty / e.n_episodes)
    end
    return fitness
end

# =============================================================================
# CurriculumSortingEvaluator — wraps SortingEvaluator with curriculum learning
# =============================================================================

mutable struct CurriculumSortingEvaluator <: Arborist.AbstractEvaluator
    current_length::Int    # starts small, increases to target_length
    target_length::Int     # final array length
    n_episodes::Int        # episodes per evaluation
    value_range::Int
    rng_seed::Int
    partial_credit::Bool
    upgrade_threshold::Float64  # fitness level to trigger curriculum advance
    comparison_alpha_base::Float64  # base multiplier for comparison penalty
                                    # adaptive α = base * max(0, current_length - 6)
    max_seed_type::Int     # max seed template type to use (1-4, 0=random only)
end

function CurriculumSortingEvaluator(;
        current_length::Int=3,
        target_length::Int=8,
        n_episodes::Int=30,
        value_range::Int=100,
        rng_seed::Int=42,
        partial_credit::Bool=true,
        upgrade_threshold::Float64=0.05,
        comparison_alpha_base::Float64=0.05,
        max_seed_type::Int=4)
    CurriculumSortingEvaluator(current_length, target_length, n_episodes,
                               value_range, rng_seed, partial_credit,
                               upgrade_threshold, comparison_alpha_base,
                               max_seed_type)
end

Arborist.input_signature(::CurriculumSortingEvaluator) = Dict{Symbol,DataType}()
Arborist.output_signature(::CurriculumSortingEvaluator) = Dict(:done => Bool)

"""Build a SortingEvaluator for the current curriculum length.
Adaptive α: zero for lengths ≤ 6 (correctness only), then increases
linearly so the comparison penalty only kicks in once arrays are large
enough for the O(n²) vs O(n log n) difference to be meaningful."""
function _make_evaluator(c::CurriculumSortingEvaluator)
    loop_limit = 10 * c.current_length^2
    α = c.comparison_alpha_base * max(0, c.current_length - 6)
    SortingEvaluator(
        n_episodes=c.n_episodes,
        array_length=c.current_length,
        value_range=c.value_range,
        rng_seed=c.rng_seed,
        partial_credit=c.partial_credit,
        loop_limit=loop_limit,
        comparison_alpha=α
    )
end

function Arborist.evaluate(e::CurriculumSortingEvaluator, f::Function)
    inner = _make_evaluator(e)
    return Arborist.evaluate(inner, f)
end

# =============================================================================
# ExprGenome integration
# =============================================================================

"""Compile an ExprGenome to a callable function for sorting."""
function _sort_compile(g::Arborist.ExprGenome, loop_limit::Int)
    fname = gensym("sort_evolved")
    try
        checked_body = Arborist.add_loop_checks(g.body; limit=loop_limit)
        harness = Arborist.create_harness(g.state, checked_body, fname)
        return @eval $harness
    catch
        return nothing
    end
end

"""Compile and evaluate an ExprGenome for sorting (curriculum evaluator)."""
function Arborist.evaluate_genome(g::Arborist.ExprGenome, e::CurriculumSortingEvaluator)
    loop_limit = 10 * e.current_length^2
    f = _sort_compile(g, loop_limit)
    f === nothing && return Inf
    try
        return Arborist.evaluate(e, f)
    catch
        return Inf
    end
end

"""Compile and evaluate an ExprGenome for sorting (basic evaluator)."""
function Arborist.evaluate_genome(g::Arborist.ExprGenome, e::SortingEvaluator)
    f = _sort_compile(g, e.loop_limit)
    f === nothing && return Inf
    try
        return Arborist.evaluate(e, f)
    catch
        return Inf
    end
end

# =============================================================================
# Function set for sorting
# =============================================================================

function sorting_function_set()
    fset = Arborist.FunctionSet(Set{Arborist.FunctionDetails}())

    # Sorting primitives
    push!(fset.funcs, Arborist.FunctionDetails(:sort_get, [Int32], Int32))
    push!(fset.funcs, Arborist.FunctionDetails(:sort_n, DataType[], Int32))
    push!(fset.funcs, Arborist.FunctionDetails(:sort_less, [Int32, Int32], Bool))
    push!(fset.funcs, Arborist.FunctionDetails(:sort_swap!, [Int32, Int32], Bool))

    # Arithmetic on Int32
    for func in [:+, :-]
        Arborist.add!(fset, func, 2, Int32, Int32)
    end

    # Comparisons on Int32
    for op in [:>, :<, :(==), :!=, :>=, :<=]
        Arborist.add!(fset, op, 2, Int32, Bool)
    end

    return fset
end

# =============================================================================
# Custom GenState for sorting
# =============================================================================

"""Create a GenState with explicit Int32/Bool temp variables for sorting."""
function _sort_create_state(rng::AbstractRNG, fset::Arborist.FunctionSet)
    inputs = Dict{Symbol, DataType}()
    outputs = Dict{Symbol, DataType}(:done => Bool)
    used_types = Set{DataType}([Bool, Int32])

    # 8 Int32 temps for loop counters and indices
    temps = Dict{Symbol, DataType}(
        :__temp_1 => Int32,
        :__temp_2 => Int32,
        :__temp_3 => Int32,
        :__temp_4 => Int32,
        :__temp_5 => Int32,
        :__temp_6 => Int32,
        :__temp_7 => Int32,
        :__temp_8 => Int32,
    )

    all_vars = merge(inputs, outputs, temps)
    statement_types = [:(=), :call, :for, :while, :if, :block]

    return Arborist.GenState(rng, statement_types, fset, inputs, outputs,
                              temps, used_types, all_vars)
end

"""Create a random rvalue of type T, sometimes using sorting function calls."""
function _sort_random_rvalue(state::Arborist.GenState, T::DataType)
    r = rand(state.rng)
    if r < 0.4
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
function _sort_random_assignment(state::Arborist.GenState)
    lvalues = Arborist.get_lvalues(state)
    v = rand(state.rng, lvalues)
    r = _sort_random_rvalue(state, v[2])
    if r isa Symbol && v[1] == r
        r = Arborist.get_random_literal(state.rng, v[2])
    end
    return :($(v[1]) = $r)
end

"""Generate an initial program body for sorting."""
function _sort_random_initial_body(state::Arborist.GenState)
    stmts = Expr[]
    n_stmts = rand(state.rng, 4:8)
    for _ in 1:n_stmts
        r = rand(state.rng)
        if r < 0.2
            push!(stmts, Arborist.create_random_while_loop(state; depth=2))
        elseif r < 0.35
            push!(stmts, Arborist.create_random_if_statement(state; depth=2))
        elseif r < 0.45
            T = rand(state.rng, Arborist._sorted_types(state.used_types))
            push!(stmts, Arborist.create_random_function_call(state, T))
        else
            push!(stmts, _sort_random_assignment(state))
        end
    end
    return stmts
end

"""Generate seeded initial programs for sorting.
These provide useful starting points for evolution:
- Type 1: single pass comparing adjacent elements (bubble sort kernel)
- Type 2: nested loop with swap (bubble sort skeleton)
- Type 3: single pass from end (reverse direction)
- Type 4: random swap attempt (random seed)"""
function _sort_seeded_body(state::Arborist.GenState, seed_type::Int)
    if seed_type == 1
        # Single pass comparing adjacent elements
        return Expr[
            :(__temp_1 = Int32(1)),
            :(while __temp_1 < sort_n()
                if sort_less(__temp_1 + Int32(1), __temp_1)
                    done = sort_swap!(__temp_1, __temp_1 + Int32(1))
                end
                __temp_1 = __temp_1 + Int32(1)
            end),
        ]
    elseif seed_type == 2
        # Nested loop — bubble sort skeleton
        return Expr[
            :(__temp_1 = Int32(1)),
            :(while __temp_1 < sort_n()
                __temp_2 = Int32(1)
                while __temp_2 < sort_n()
                    if sort_less(__temp_2 + Int32(1), __temp_2)
                        done = sort_swap!(__temp_2, __temp_2 + Int32(1))
                    end
                    __temp_2 = __temp_2 + Int32(1)
                end
                __temp_1 = __temp_1 + Int32(1)
            end),
        ]
    elseif seed_type == 3
        # Single pass from end (reverse direction)
        return Expr[
            :(__temp_1 = sort_n()),
            :(while __temp_1 > Int32(1)
                if sort_less(__temp_1, __temp_1 - Int32(1))
                    done = sort_swap!(__temp_1, __temp_1 - Int32(1))
                end
                __temp_1 = __temp_1 - Int32(1)
            end),
        ]
    elseif seed_type == 4
        # Outer loop with inner scan — selection sort skeleton
        return Expr[
            :(__temp_1 = Int32(1)),
            :(while __temp_1 < sort_n()
                __temp_3 = __temp_1
                __temp_2 = __temp_1 + Int32(1)
                while __temp_2 <= sort_n()
                    if sort_less(__temp_2, __temp_3)
                        __temp_3 = __temp_2
                    end
                    __temp_2 = __temp_2 + Int32(1)
                end
                done = sort_swap!(__temp_1, __temp_3)
                __temp_1 = __temp_1 + Int32(1)
            end),
        ]
    else
        return _sort_random_initial_body(state)
    end
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
    println(io, "function evolved_sorter()")
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
# Verification
# =============================================================================

"""Run the evolved genome on a single array and return the result."""
function run_genome_on_array(g::Arborist.ExprGenome, arr::Vector{Int32};
                             loop_limit::Int=640)
    f = _sort_compile(g, loop_limit)
    f === nothing && return copy(arr)
    reset_sort_state!(arr)
    try
        Base.invokelatest(f)
    catch
    end
    return copy(_sort_state[].arr)
end

"""Verification result for a single array length."""
struct VerifyResult
    accuracy::Float64        # fraction correctly sorted
    mean_comparisons::Float64
    mean_swaps::Float64
end

"""Verify the evolved sorter on unseen test arrays.
Returns a Dict mapping array length to VerifyResult with accuracy,
mean comparisons, and mean swaps."""
function verify_sorter(g::Arborist.ExprGenome;
                       n_tests::Int=1000,
                       max_length::Int=24,
                       rng=Random.MersenneTwister(999))
    # Pre-compile once (use max_length for loop limit)
    loop_limit_max = 10 * max_length^2
    f = _sort_compile(g, loop_limit_max)

    results = Dict{Int, VerifyResult}()
    for len in 3:max_length
        correct = 0
        total_comps = 0
        total_swaps = 0
        for _ in 1:n_tests
            arr = rand(rng, Int32(-100):Int32(100), len)
            reset_sort_state!(arr)
            if f !== nothing
                try
                    Base.invokelatest(f)
                catch
                end
            end
            s = _sort_state[]
            issorted(s.arr) && (correct += 1)
            total_comps += Int(s.comparisons)
            total_swaps += Int(s.swaps)
        end
        results[len] = VerifyResult(
            correct / n_tests,
            total_comps / n_tests,
            total_swaps / n_tests
        )
    end
    return results
end

# =============================================================================
# Custom solve method with curriculum learning
# =============================================================================

function Arborist.solve(problem::Arborist.GPProblem{Arborist.ExprGenome, E},
                        algorithm::Arborist.GeneticProgramming;
                        verbose::Bool = false,
                        callback = nothing) where {E<:CurriculumSortingEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    evaluator = problem.evaluator
    pop_size = algorithm.pop_size

    # Create custom GenState with explicit Int32 temps
    state = _sort_create_state(rng, problem.function_set)

    # Initialize population: mix of seeded templates and random programs
    genomes = Vector{Arborist.ExprGenome}(undef, pop_size)
    n_seed_types = evaluator.max_seed_type
    if n_seed_types > 0
        seeds_per_type = max(1, pop_size ÷ 10)
        n_seeded = min(seeds_per_type * n_seed_types, pop_size ÷ 2)
    else
        n_seeded = 0
    end
    for i in 1:pop_size
        if i <= n_seeded
            seed_type = ((i - 1) % n_seed_types) + 1
            body = _sort_seeded_body(state, seed_type)
        else
            body = _sort_random_initial_body(state)
        end
        genomes[i] = Arborist.ExprGenome(body, state)
    end
    fitnesses = fill(Inf, pop_size)

    bp = algorithm.bloat_penalty

    # Evaluate initial population
    for i in 1:pop_size
        fitnesses[i] = Arborist.evaluate_genome(genomes[i], evaluator)
        if bp > 0.0 && isfinite(fitnesses[i])
            fitnesses[i] += bp * Arborist.complexity(genomes[i])
        end
    end

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

        # Check curriculum advancement using correctness-only fitness.
        # We check the top individuals (not just genomes[1]) because with
        # the gated comparison penalty, a correct sorter that pays the
        # comparison cost can rank below an incorrect program that pays none.
        should_advance = false
        if evaluator.current_length < evaluator.target_length
            correctness_eval = SortingEvaluator(
                n_episodes=evaluator.n_episodes,
                array_length=evaluator.current_length,
                value_range=evaluator.value_range,
                rng_seed=evaluator.rng_seed,
                partial_credit=evaluator.partial_credit,
                loop_limit=10 * evaluator.current_length^2,
                comparison_alpha=0.0
            )
            n_check = min(algorithm.elitism * 4, pop_size)
            for i in 1:n_check
                c = Arborist.evaluate_genome(genomes[i], correctness_eval)
                if c < evaluator.upgrade_threshold
                    should_advance = true
                    break
                end
            end
        end
        if should_advance
            old_length = evaluator.current_length
            evaluator.current_length += 1
            new_α = evaluator.comparison_alpha_base * max(0, evaluator.current_length - 6)
            α_note = new_α > 0 ? " (α=$(round(new_α, digits=3)))" : ""
            println("Curriculum advance: length $old_length -> $(evaluator.current_length) at gen $gen$α_note")
            flush(stdout)
            # Re-evaluate entire population at new length
            for i in 1:pop_size
                fitnesses[i] = Arborist.evaluate_genome(genomes[i], evaluator)
                if bp > 0.0 && isfinite(fitnesses[i])
                    fitnesses[i] += bp * Arborist.complexity(genomes[i])
                end
            end
            # Re-sort after re-evaluation
            order = sortperm(fitnesses)
            genomes = genomes[order]
            fitnesses = fitnesses[order]
        end

        selection_fitnesses = Arborist._apply_speciation!(genomes, fitnesses,
                                                           algorithm.speciation, species_state, rng)

        elapsed = round(time() - t0, digits=1)
        α = evaluator.comparison_alpha_base * max(0, evaluator.current_length - 6)
        if verbose
            α_str = α > 0 ? " | α=$(round(α, digits=3))" : ""
            println("Gen $gen/$(algorithm.generations) | " *
                    "best=$(round(fitnesses[1], digits=4)) | " *
                    "mean=$(round(mean_fit, digits=4)) | " *
                    "curriculum_len=$(evaluator.current_length)$α_str | " *
                    "elapsed=$(elapsed)s")
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

        # Evaluate new individuals (skip elites)
        for i in (algorithm.elitism + 1):pop_size
            next_fitnesses[i] = Arborist.evaluate_genome(next_genomes[i], evaluator)
            if bp > 0.0 && isfinite(next_fitnesses[i])
                next_fitnesses[i] += bp * Arborist.complexity(next_genomes[i])
            end
        end

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
        fitnesses[1] < 0.01  # converged if nearly perfect sort
    )
end

# =============================================================================
# Main entry point
# =============================================================================

function run_sorting(;
        pop_size::Int = 300,
        generations::Int = 1000,
        mutation_rate::Float64 = 0.4,
        crossover_rate::Float64 = 0.3,
        elitism::Int = 5,
        tournament_size::Int = 5,
        max_depth::Int = 12,
        bloat_penalty::Float64 = 0.0005,
        seed::Int = 42,
        start_length::Int = 3,
        target_length::Int = 20,
        n_episodes::Int = 30,
        upgrade_threshold::Float64 = 0.05,
        comparison_alpha_base::Float64 = 0.05,
        max_seed_type::Int = 4,
        verbose::Bool = true)

    println("=" ^ 70)
    println("Arborist.jl — Evolving a Sorting Algorithm")
    println("=" ^ 70)
    println("Parameters:")
    println("  pop_size=$pop_size, generations=$generations")
    println("  mutation_rate=$mutation_rate, crossover_rate=$crossover_rate")
    println("  elitism=$elitism, tournament_size=$tournament_size")
    println("  max_depth=$max_depth, bloat_penalty=$bloat_penalty")
    println("  seed=$seed")
    println("  curriculum: start_length=$start_length -> target_length=$target_length")
    println("  n_episodes=$n_episodes, upgrade_threshold=$upgrade_threshold")
    println("  comparison_alpha_base=$comparison_alpha_base")
    println("  (adaptive α = $comparison_alpha_base * max(0, length - 6))")
    seed_desc = max_seed_type == 0 ? "none (random only)" :
                max_seed_type == 1 ? "type 1 only (single-pass adjacent swap)" :
                max_seed_type == 2 ? "types 1-2 (single-pass + bubble sort skeleton)" :
                max_seed_type == 3 ? "types 1-3 (single-pass + bubble + reverse)" :
                "types 1-4 (single-pass + bubble + reverse + selection sort)"
    println("  seed templates: $seed_desc")
    println("  started: $(Dates.now())")
    println("=" ^ 70)
    flush(stdout)

    evaluator = CurriculumSortingEvaluator(
        current_length=start_length,
        target_length=target_length,
        n_episodes=n_episodes,
        upgrade_threshold=upgrade_threshold,
        comparison_alpha_base=comparison_alpha_base,
        max_seed_type=max_seed_type,
    )

    fset = sorting_function_set()

    algorithm = Arborist.GeneticProgramming(
        pop_size        = pop_size,
        generations     = generations,
        mutation_rate   = mutation_rate,
        crossover_rate  = crossover_rate,
        elitism         = elitism,
        tournament_size = tournament_size,
        max_depth       = max_depth,
        bloat_penalty   = bloat_penalty,
        mutation_ops    = [Arborist.SubtreeMutation(), Arborist.PointMutation(),
                           Arborist.HoistMutation(), Arborist.ExpansionMutation()],
    )

    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=8, seed=seed)

    result = Arborist.solve(problem, algorithm; verbose=verbose)

    println()
    println("=" ^ 70)
    println("RESULTS")
    println("=" ^ 70)
    println("Best fitness: $(round(result.best_fitness, digits=6))")
    println("Generations run: $(result.generations_run)")
    println("Wall time: $(round(result.wall_time, digits=1))s")
    println("Converged: $(result.converged)")
    println("Final curriculum length: $(evaluator.current_length)")
    flush(stdout)

    println()
    println("Best evolved program:")
    println("-" ^ 40)
    program_text = pretty_print_genome(result.best_genome)
    println(program_text)
    flush(stdout)

    # Verification
    verify_max = max(target_length + 4, 24)
    println()
    println("Verification results (1000 random arrays per length, unseen seeds):")
    verification = verify_sorter(result.best_genome; n_tests=1000, max_length=verify_max)
    println("  Length | Correct |  Comps  |  Swaps  | Comps/n² | Comps/(n·lg n)")
    println("  ------+---------+---------+---------+----------+--------------")
    for len in 3:verify_max
        v = verification[len]
        pct = round(v.accuracy * 100, digits=1)
        comps = round(v.mean_comparisons, digits=1)
        swaps = round(v.mean_swaps, digits=1)
        comp_n2 = round(v.mean_comparisons / len^2, digits=3)
        nlogn = len * log2(len)
        comp_nlogn = round(v.mean_comparisons / nlogn, digits=3)
        println("  $(lpad(len, 5)) | $(lpad(pct, 6))% | $(lpad(comps, 7)) | $(lpad(swaps, 7)) | $(lpad(comp_n2, 8)) | $(lpad(comp_nlogn, 12))")
    end
    flush(stdout)

    # Determine tier
    len8_pct = verification[8].accuracy
    tier = if len8_pct >= 1.0
        "Tier 2 (target): 100% correct on length-8"
    elseif len8_pct >= 0.9
        "Tier 1 (minimum): ≥90% correct on length-8"
    else
        "Below Tier 1: $(round(len8_pct * 100, digits=1))% correct on length-8"
    end

    # Check generalization
    gen_max = min(verify_max, target_length + 4)
    generalizes = gen_max > target_length &&
                  all(verification[len].accuracy >= 0.95 for len in (target_length+1):gen_max)
    gen_str = generalizes ?
        "Yes — program generalizes beyond training distribution" :
        "No — performance degrades on longer arrays (possible overfitting to length)"

    # Characterize comparison complexity
    # Compare to n*(n-1)/2 (selection sort) and n*log2(n) (optimal)
    comp_analysis = ""
    if verification[verify_max].accuracy >= 0.95
        comps_at_max = verification[verify_max].mean_comparisons
        n = verify_max
        sel_sort_comps = n * (n - 1) / 2
        nlogn_comps = n * log2(n)
        ratio_sel = comps_at_max / sel_sort_comps
        ratio_nlogn = comps_at_max / nlogn_comps
        comp_analysis = if ratio_sel < 0.6
            "Sub-quadratic: $(round(comps_at_max, digits=0)) comps at n=$n " *
            "($(round(ratio_nlogn, digits=1))× n·lg(n), $(round(ratio_sel, digits=2))× selection sort)"
        elseif ratio_sel < 0.9
            "Reduced quadratic: $(round(comps_at_max, digits=0)) comps at n=$n " *
            "($(round(ratio_sel, digits=2))× selection sort)"
        else
            "Quadratic: $(round(comps_at_max, digits=0)) comps at n=$n " *
            "(≈ selection sort's $(round(sel_sort_comps, digits=0)))"
        end
    end

    println()
    println("Achievement: $tier")
    println("Generalization (length $(target_length+1)-$gen_max): $gen_str")
    if !isempty(comp_analysis)
        println("Comparison complexity: $comp_analysis")
    end
    flush(stdout)

    # Save results
    results_path = joinpath(@__DIR__, "sorting_results.md")
    open(results_path, "w") do io
        println(io, "# Sorting Algorithm Evolution Results")
        println(io)
        println(io, "**Date:** $(Dates.now())")
        println(io, "**Seed:** $seed")
        println(io)
        println(io, "## Parameters")
        println(io, "- pop_size=$pop_size, generations=$generations")
        println(io, "- mutation_rate=$mutation_rate, crossover_rate=$crossover_rate")
        println(io, "- elitism=$elitism, tournament_size=$tournament_size")
        println(io, "- bloat_penalty=$bloat_penalty")
        println(io, "- curriculum: $start_length → $target_length")
        println(io, "- upgrade_threshold=$upgrade_threshold")
        println(io, "- comparison_alpha_base=$comparison_alpha_base")
        println(io, "- (adaptive α = $comparison_alpha_base × max(0, length − 6))")
        println(io)
        println(io, "## Results")
        println(io, "- **Best fitness:** $(round(result.best_fitness, digits=6))")
        println(io, "- **Wall time:** $(round(result.wall_time, digits=1))s")
        println(io, "- **Converged:** $(result.converged)")
        println(io, "- **Final curriculum length:** $(evaluator.current_length)")
        println(io, "- **Achievement:** $tier")
        println(io, "- **Generalization:** $gen_str")
        if !isempty(comp_analysis)
            println(io, "- **Comparison complexity:** $comp_analysis")
        end
        println(io)
        println(io, "## Evolved Program")
        println(io, "```julia")
        print(io, program_text)
        println(io, "```")
        println(io)
        println(io, "## Fitness History (curriculum advances)")
        for (gen, fit) in enumerate(result.fitness_history)
            if gen == 1 || gen == length(result.fitness_history) ||
               (gen > 1 && result.fitness_history[gen] != result.fitness_history[gen-1] &&
                abs(result.fitness_history[gen] - result.fitness_history[max(1,gen-1)]) > 0.01)
                println(io, "- Gen $gen: $(round(fit, digits=6))")
            end
        end
        println(io)
        println(io, "## Verification")
        println(io, "```")
        println(io, "Verification results (1000 random arrays per length, unseen seeds):")
        println(io, "  Length | Correct |  Comps  |  Swaps  | Comps/n² | Comps/(n·lg n)")
        println(io, "  ------+---------+---------+---------+----------+--------------")
        for len in 3:verify_max
            v = verification[len]
            pct = round(v.accuracy * 100, digits=1)
            comps = round(v.mean_comparisons, digits=1)
            swaps = round(v.mean_swaps, digits=1)
            comp_n2 = round(v.mean_comparisons / len^2, digits=3)
            nlogn = len * log2(len)
            comp_nlogn = round(v.mean_comparisons / nlogn, digits=3)
            println(io, "  $(lpad(len, 5)) | $(lpad(pct, 6))% | $(lpad(comps, 7)) | $(lpad(swaps, 7)) | $(lpad(comp_n2, 8)) | $(lpad(comp_nlogn, 12))")
        end
        println(io, "```")
        println(io)
        println(io, "## Notes")
        println(io, "This example evolves a sorting algorithm from scratch using genetic")
        println(io, "programming with curriculum learning and an adaptive comparison count")
        println(io, "penalty. The evolved program accesses the array only through scalar")
        println(io, "primitives (sort_get, sort_n, sort_less, sort_swap!) — no built-in")
        println(io, "sort function is available.")
        println(io)
        println(io, "The comparison penalty α = $(comparison_alpha_base) × max(0, n − 6) is zero for")
        println(io, "the early curriculum stages (lengths 3–6, correctness only) and")
        println(io, "increases linearly as arrays get longer, creating pressure to reduce")
        println(io, "comparison count. At length 24, the O(n²) vs O(n log n) difference")
        println(io, "is ~5×, which provides meaningful fitness signal.")
        println(io)
        println(io, "Sorting is a classic GP benchmark (Koza 1992). Unlike Koza's sorting")
        println(io, "networks which used a fixed comparator representation, this example")
        println(io, "evolves imperative programs with loops, conditionals, and variable")
        println(io, "assignments — making the search space much larger but the evolved")
        println(io, "programs more interpretable as conventional sorting algorithms.")
    end

    println()
    println("Results saved to: $results_path")
    flush(stdout)

    return result
end

function main()
    # Parse command-line arguments
    kwargs = Dict{Symbol, Any}()
    for arg in ARGS
        m = match(r"^--(\w+)=(.+)$", arg)
        if m !== nothing
            key = Symbol(m.captures[1])
            val_str = m.captures[2]
            if key in (:pop_size, :generations, :elitism, :tournament_size,
                       :max_depth, :seed, :start_length, :target_length,
                       :n_episodes, :max_seed_type)
                kwargs[key] = parse(Int, val_str)
            elseif key in (:mutation_rate, :crossover_rate, :bloat_penalty,
                           :upgrade_threshold, :comparison_alpha_base)
                kwargs[key] = parse(Float64, val_str)
            elseif key == :verbose
                kwargs[key] = val_str == "true"
            end
        end
    end

    run_sorting(; kwargs...)
end

# Run if invoked as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
