#!/usr/bin/env julia
# NSGA-II bin packing ablation experiments.
#
# Six ablations of the baseline 3-objective unseeded experiment (run_nsga2_unseeded.jl):
#   1. no_llm         — Pure GP, no LLM mutation operator
#   2. no_behavioral   — Random init instead of behavioral initialization (10k pool)
#   3. two_objective   — fitness + failed_placements only (drop success_rate)
#   4a. pop100         — Population 100 (half baseline)
#   4b. pop50          — Population 50 (quarter baseline)
#   5. llm_only        — LLM mutation only, no classical operators
#   6. neutral_prompt  — LLM with neutral system prompt (no Best Fit hint)
#
# All ablations use 200 generations, seed 42, same evaluation parameters.
#
# Usage: julia --project -t auto examples/run_nsga2_ablations.jl --ablation=no_llm
#        julia --project -t auto examples/run_nsga2_ablations.jl --ablation=all

using Arborist
using Random
using Dates

include(joinpath(@__DIR__, "bin_packing.jl"))

# =============================================================================
# Neutral system prompt — describes the problem without hinting at Best Fit
# =============================================================================

const BP_NEUTRAL_SYSTEM_PROMPT = """
You are a genetic programming mutation operator for an online bin
packing heuristic written in Julia.

## Problem
Items arrive one at a time. Each item has a size between 0 and 1.
Each bin has capacity 1.0. The program is called once per item and
must place it in a bin. The goal is to pack the maximum number of
objects in the minimum number of bins.

## API (primitives available to the program)
  bp_n_bins()::Int32
    Returns the number of currently open bins. Bins are 1-indexed:
    valid bin indices are 1, 2, ..., bp_n_bins(). Placing an item
    in an index > bp_n_bins() opens a new bin.

  bp_bin_remaining(i::Int32)::Float32
    Returns the remaining capacity of bin i (1-indexed). Calling
    with i < 1 or i > bp_n_bins() returns 0.0.

  bp_item_size()::Float32
    Returns the size of the current item (between 0.0 and 1.0).

  bp_capacity()::Float32
    Returns the bin capacity (always 1.0).

  bp_place_in_bin(i::Int32)::Bool
    Places the current item in bin i. Returns true if successful
    (item fits), false otherwise. Only i in 1..bp_n_bins()+1 is
    valid: existing bins 1..bp_n_bins(), or bp_n_bins()+1 to open
    exactly one new bin. Indices outside this range return false.
    Each item can only be placed once; subsequent calls after a
    successful placement return false.
    IMPORTANT: failed placement calls incur a fitness penalty.

## Variables
All variables are pre-declared with fixed types. Use only these:
  __temp_1, __temp_2, __temp_3 :: Int32   (loop counters, bin indices)
  __temp_4, __temp_5, __temp_6 :: Float32 (scores, remaining capacity)
  result :: Bool                          (output, set by bp_place_in_bin)

All Int32 variables are initialized to 0. All Float32 variables are
initialized to 0.0. Bin scanning should start at Int32(1), not 0.

## Rules
- Return ONLY valid Julia assignment statements and control flow.
- Use only the variables and primitives listed above.
- Do not import anything or define functions.
- Use Int32 literals for integer values: Int32(1), Int32(0), etc.
- Use Float32 literals for float values: 1.0f0, 0.0f0, etc.

Respond with only the Julia statements, nothing else.
"""

# =============================================================================
# Two-objective evaluator: fitness + failed_placements (no success_rate)
# =============================================================================

struct BPTwoObjectiveEvaluator <: Arborist.AbstractMultiObjectiveEvaluator
    inner::BinPackingEvaluator
end

function Arborist.evaluate_multi(e::BPTwoObjectiveEvaluator, genome::Arborist.ExprGenome)
    f = _bp_compile(genome)
    if f === nothing
        return [Inf, Inf]
    end
    try
        _ensure_bp_states()
        fitness, aux = _bp_evaluate_with_aux(e.inner, f)
        return [fitness, Float64(aux.failed_placements)]
    catch
        return [Inf, Inf]
    end
end

Arborist.objective_names(::BPTwoObjectiveEvaluator) = ["fitness", "failed_placements"]
Arborist.input_signature(e::BPTwoObjectiveEvaluator) = Arborist.input_signature(e.inner)
Arborist.output_signature(e::BPTwoObjectiveEvaluator) = Arborist.output_signature(e.inner)

# Behavioral init override for the two-objective evaluator
function Arborist._nsga2_init_population(
    problem::Arborist.GPProblem{Arborist.ExprGenome, BPTwoObjectiveEvaluator},
    algorithm::Arborist.NSGAII,
    rng::AbstractRNG)

    inner_eval = problem.evaluator.inner
    fset = problem.function_set
    state = _bp_create_state(rng, fset)

    _ensure_bp_states()
    probe = BehavioralProbe(n_items=30, n_probe_bins=10)
    fp_fn = g -> compute_bp_fingerprint(g, probe)

    return Arborist.behavioral_initialize(
        state, inner_eval, fp_fn, behavioral_distance, algorithm.pop_size;
        pool_size=10_000,
        bin_threshold=0.15,
        body_generator=s -> _bp_random_initial_body(s),
        parallel=true,
        verbose=true
    )
end

# =============================================================================
# Random-init evaluator wrapper (bypasses behavioral init override)
# =============================================================================

struct BPRandomInitEvaluator <: Arborist.AbstractMultiObjectiveEvaluator
    inner::BinPackingEvaluator
end

function Arborist.evaluate_multi(e::BPRandomInitEvaluator, genome::Arborist.ExprGenome)
    f = _bp_compile(genome)
    if f === nothing
        return [Inf, 0.0, Inf]
    end
    try
        _ensure_bp_states()
        fitness, aux = _bp_evaluate_with_aux(e.inner, f)
        return [fitness, -aux.success_rate, Float64(aux.failed_placements)]
    catch
        return [Inf, 0.0, Inf]
    end
end

Arborist.objective_names(::BPRandomInitEvaluator) = ["fitness", "neg_success_rate", "failed_placements"]
Arborist.input_signature(e::BPRandomInitEvaluator) = Arborist.input_signature(e.inner)
Arborist.output_signature(e::BPRandomInitEvaluator) = Arborist.output_signature(e.inner)

# Falls through to the default _nsga2_init_population in nsga2.jl
# (3 random assignments per genome, no behavioral diversity)

# =============================================================================
# Shared helpers
# =============================================================================

const ABLATION_NAMES = ["no_llm", "no_behavioral", "two_objective",
                        "pop100", "pop50", "llm_only", "neutral_prompt"]

function make_inner_eval()
    BinPackingEvaluator(n_episodes=20, n_items=200, capacity=1.0f0,
                        item_dist=:uniform, rng_seed=42)
end

function make_llm_op(debug_io; system_prompt=BP_LLM_SYSTEM_PROMPT)
    op = Arborist.LLMMutationOperator(
        endpoint    = "http://localhost:11434/v1/chat/completions",
        model       = "qwen3-coder:30b",
        api_key_env = "",
        system_prompt = system_prompt,
        temperature = 0.7,
        max_tokens  = 256,
        timeout_seconds = 60.0,
        fallback_op = Arborist.SubtreeMutation(),
        sections    = Arborist.AbstractPromptSection[Arborist.ElitesSection(3)],
    )
    op.debug_log = debug_io
    return op
end

function classical_mutation_ops()
    Arborist.AbstractMutationOperator[
        Arborist.SubtreeMutation(),
        Arborist.SubtreeMutation(),  # double-weight to compensate for missing LLM slot
        Arborist.PointMutation(),
        Arborist.HoistMutation(),
        Arborist.ExpansionMutation(),
    ]
end

function baseline_mutation_ops(llm_op)
    Arborist.AbstractMutationOperator[
        TrackedMutation(llm_op),
        Arborist.SubtreeMutation(),
        Arborist.PointMutation(),
        Arborist.HoistMutation(),
        Arborist.ExpansionMutation(),
    ]
end

function print_results(result, wall; llm_op=nothing)
    n_obj = length(result.pareto_fitnesses[1])

    println("\n" * "=" ^ 70)
    println("Results after $(round(wall, digits=1))s")
    println("=" ^ 70)
    println("Pareto front size: $(length(result.pareto_front))")

    sorted_idx = sortperm(result.pareto_fitnesses, by=f -> f[1])

    if n_obj == 3
        println("\nTop 10 by primary fitness:")
        println("  # | fitness | success_rate | failed_place | has_while | has_if")
        println("  --|---------|--------------|--------------|-----------|-------")
        for (rank, i) in enumerate(sorted_idx[1:min(10, length(sorted_idx))])
            f = result.pareto_fitnesses[i]
            src = Arborist.serialize(result.pareto_front[i])
            hw = occursin("while", src) ? "yes" : "no"
            hi = occursin("if", src) ? "yes" : "no"
            println("  $(rank) | $(round(f[1], digits=4)) | $(round(-f[2], digits=3)) | $(round(f[3], digits=0)) | $hw | $hi")
        end

        fp_idx = sortperm(result.pareto_fitnesses, by=f -> f[3])
        println("\nTop 5 by fewest failed placements:")
        for (rank, i) in enumerate(fp_idx[1:min(5, length(fp_idx))])
            f = result.pareto_fitnesses[i]
            src = Arborist.serialize(result.pareto_front[i])
            hw = occursin("while", src) ? "yes" : "no"
            hi = occursin("if", src) ? "yes" : "no"
            println("  $(rank) | fitness=$(round(f[1], digits=4)) sr=$(round(-f[2], digits=3)) failed=$(round(f[3], digits=0)) | while=$hw if=$hi")
        end
    elseif n_obj == 2
        println("\nTop 10 by primary fitness:")
        println("  # | fitness | failed_place | has_while | has_if")
        println("  --|---------|--------------|-----------|-------")
        for (rank, i) in enumerate(sorted_idx[1:min(10, length(sorted_idx))])
            f = result.pareto_fitnesses[i]
            src = Arborist.serialize(result.pareto_front[i])
            hw = occursin("while", src) ? "yes" : "no"
            hi = occursin("if", src) ? "yes" : "no"
            println("  $(rank) | $(round(f[1], digits=4)) | $(round(f[2], digits=0)) | $hw | $hi")
        end

        fp_idx = sortperm(result.pareto_fitnesses, by=f -> f[2])
        println("\nTop 5 by fewest failed placements:")
        for (rank, i) in enumerate(fp_idx[1:min(5, length(fp_idx))])
            f = result.pareto_fitnesses[i]
            src = Arborist.serialize(result.pareto_front[i])
            hw = occursin("while", src) ? "yes" : "no"
            hi = occursin("if", src) ? "yes" : "no"
            println("  $(rank) | fitness=$(round(f[1], digits=4)) failed=$(round(f[2], digits=0)) | while=$hw if=$hi")
        end
    end

    # Print best program
    best_i = sorted_idx[1]
    println("\nBest program (by primary fitness):")
    best = result.pareto_front[best_i]
    checked_body = Arborist.add_loop_checks(best.body; limit=1000)
    harness = Arborist.create_harness(best.state, checked_body, gensym("bp_best"))
    println(harness)

    # Print fewest-failed-placements program if different
    fp_col = n_obj == 3 ? 3 : 2
    fp_idx = sortperm(result.pareto_fitnesses, by=f -> f[fp_col])
    sr_best_i = fp_idx[1]
    if sr_best_i != best_i
        println("\nFewest-failed-placements program:")
        sr_best = result.pareto_front[sr_best_i]
        checked_body = Arborist.add_loop_checks(sr_best.body; limit=1000)
        harness = Arborist.create_harness(sr_best.state, checked_body, gensym("bp_fewest_failed"))
        println(harness)
    end

    # LLM stats if applicable
    if llm_op !== nothing
        s = llm_op.stats
        println("\nLLM Stats:")
        println("  Calls: $(s.total_calls), Successes: $(s.llm_successes), Failures: $(s.llm_failures)")
    end
    flush(stdout)
end

# =============================================================================
# Ablation runners
# =============================================================================

function run_no_llm()
    _ensure_bp_states()

    evaluator = BPMultiObjectiveEvaluator(make_inner_eval())
    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)
    algorithm = Arborist.NSGAII(
        pop_size=200, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=classical_mutation_ops(),
    )

    println("=" ^ 70)
    println("Ablation: NO_LLM — Pure GP + NSGA-II (3 objectives)")
    println("  Objectives: fitness, success_rate, failed_placements")
    println("  Population: 200, Generations: 200")
    println("  Init: behavioral (10k pool)")
    println("  Mutation: SubtreeMutation(x2), PointMutation, HoistMutation, ExpansionMutation")
    println("=" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=true)
    wall = time() - t0

    print_results(result, wall)
end

function run_no_behavioral()
    _ensure_bp_states()

    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "nsga2_ablation_no_behavioral.log")
    debug_io = open(debug_path, "w")
    llm_op = make_llm_op(debug_io)

    # BPRandomInitEvaluator has no custom _nsga2_init_population override,
    # so it falls through to the default (3 random assignments per genome).
    evaluator = BPRandomInitEvaluator(make_inner_eval())
    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)
    algorithm = Arborist.NSGAII(
        pop_size=200, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=baseline_mutation_ops(llm_op),
    )

    println("=" ^ 70)
    println("Ablation: NO_BEHAVIORAL — Random init + NSGA-II + LLM")
    println("  Objectives: fitness, success_rate, failed_placements")
    println("  Population: 200, Generations: 200")
    println("  Init: random (3 assignments per genome)")
    println("  LLM: qwen3-coder:30b + ElitesSection(3)")
    println("  Debug log: $debug_path")
    println("=" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=true)
    wall = time() - t0
    close(debug_io)

    print_results(result, wall; llm_op=llm_op)
end

function run_two_objective()
    _ensure_bp_states()

    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "nsga2_ablation_two_objective.log")
    debug_io = open(debug_path, "w")
    llm_op = make_llm_op(debug_io)

    evaluator = BPTwoObjectiveEvaluator(make_inner_eval())
    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)
    algorithm = Arborist.NSGAII(
        pop_size=200, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=baseline_mutation_ops(llm_op),
    )

    println("=" ^ 70)
    println("Ablation: TWO_OBJECTIVE — fitness + failed_placements (no success_rate)")
    println("  Objectives: fitness, failed_placements")
    println("  Population: 200, Generations: 200")
    println("  Init: behavioral (10k pool)")
    println("  LLM: qwen3-coder:30b + ElitesSection(3)")
    println("  Debug log: $debug_path")
    println("=" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=true)
    wall = time() - t0
    close(debug_io)

    print_results(result, wall; llm_op=llm_op)
end

function run_reduced_pop(pop_size::Int)
    _ensure_bp_states()

    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "nsga2_ablation_pop$(pop_size).log")
    debug_io = open(debug_path, "w")
    llm_op = make_llm_op(debug_io)

    evaluator = BPMultiObjectiveEvaluator(make_inner_eval())
    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)
    algorithm = Arborist.NSGAII(
        pop_size=pop_size, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=baseline_mutation_ops(llm_op),
    )

    println("=" ^ 70)
    println("Ablation: POP$(pop_size) — Reduced population ($(pop_size))")
    println("  Objectives: fitness, success_rate, failed_placements")
    println("  Population: $(pop_size), Generations: 200")
    println("  Init: behavioral (10k pool)")
    println("  LLM: qwen3-coder:30b + ElitesSection(3)")
    println("  Debug log: $debug_path")
    println("=" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=true)
    wall = time() - t0
    close(debug_io)

    print_results(result, wall; llm_op=llm_op)
end

function run_llm_only()
    _ensure_bp_states()

    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "nsga2_ablation_llm_only.log")
    debug_io = open(debug_path, "w")
    llm_op = make_llm_op(debug_io)

    evaluator = BPMultiObjectiveEvaluator(make_inner_eval())
    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)
    algorithm = Arborist.NSGAII(
        pop_size=200, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=Arborist.AbstractMutationOperator[TrackedMutation(llm_op)],
    )

    println("=" ^ 70)
    println("Ablation: LLM_ONLY — LLM mutation only, no classical operators")
    println("  Objectives: fitness, success_rate, failed_placements")
    println("  Population: 200, Generations: 200")
    println("  Init: behavioral (10k pool)")
    println("  Mutation: TrackedMutation(LLM) only")
    println("  LLM: qwen3-coder:30b + ElitesSection(3)")
    println("  Debug log: $debug_path")
    println("=" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=true)
    wall = time() - t0
    close(debug_io)

    print_results(result, wall; llm_op=llm_op)
end

function run_neutral_prompt()
    _ensure_bp_states()

    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "nsga2_ablation_neutral_prompt.log")
    debug_io = open(debug_path, "w")
    llm_op = make_llm_op(debug_io; system_prompt=BP_NEUTRAL_SYSTEM_PROMPT)

    evaluator = BPMultiObjectiveEvaluator(make_inner_eval())
    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)
    algorithm = Arborist.NSGAII(
        pop_size=200, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=baseline_mutation_ops(llm_op),
    )

    println("=" ^ 70)
    println("Ablation: NEUTRAL_PROMPT — LLM with neutral system prompt")
    println("  Objectives: fitness, success_rate, failed_placements")
    println("  Population: 200, Generations: 200")
    println("  Init: behavioral (10k pool)")
    println("  LLM: qwen3-coder:30b + ElitesSection(3) + NEUTRAL prompt")
    println("  Debug log: $debug_path")
    println("=" ^ 70)
    flush(stdout)

    t0 = time()
    result = Arborist.solve(problem, algorithm; verbose=true)
    wall = time() - t0
    close(debug_io)

    print_results(result, wall; llm_op=llm_op)
end

# =============================================================================
# CLI dispatch
# =============================================================================

function parse_ablation()
    for arg in ARGS
        m = match(r"^--ablation=(.+)$", arg)
        if m !== nothing
            return m[1]
        end
    end
    println("Usage: julia --project -t auto examples/run_nsga2_ablations.jl --ablation=<name>")
    println("  Available ablations: $(join(ABLATION_NAMES, ", ")), all")
    exit(1)
end

function run_ablation(name::AbstractString)
    if name == "no_llm"
        run_no_llm()
    elseif name == "no_behavioral"
        run_no_behavioral()
    elseif name == "two_objective"
        run_two_objective()
    elseif name == "pop100"
        run_reduced_pop(100)
    elseif name == "pop50"
        run_reduced_pop(50)
    elseif name == "llm_only"
        run_llm_only()
    elseif name == "neutral_prompt"
        run_neutral_prompt()
    else
        println("Unknown ablation: $name")
        println("Available: $(join(ABLATION_NAMES, ", ")), all")
        exit(1)
    end
end

function main()
    ablation = parse_ablation()

    if ablation == "all"
        for name in ABLATION_NAMES
            println("\n\n" * "#" ^ 70)
            println("# Running ablation: $name")
            println("#" ^ 70 * "\n")
            flush(stdout)
            run_ablation(name)
        end
    else
        run_ablation(ablation)
    end
end

main()
