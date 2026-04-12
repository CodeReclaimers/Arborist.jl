#!/usr/bin/env julia
# NSGA-II bin packing with auxiliary trace objectives + qwen3-coder.
#
# Three objectives: [primary_fitness, -coverage, -success_rate]
# Behavioral initialization (10k pool).
# LLM mutation with ElitesSection(3) enrichment + debug logging.
#
# Usage: julia --project -t auto examples/run_nsga2_unseeded.jl

using Arborist
using Random
using Dates

include(joinpath(@__DIR__, "bin_packing.jl"))

function main()
    _ensure_bp_states()

    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "nsga2_unseeded.log")
    debug_io = open(debug_path, "w")

    llm_op = Arborist.LLMMutationOperator(
        endpoint    = "http://localhost:11434/v1/chat/completions",
        model       = "qwen3-coder:30b",
        api_key_env = "",
        system_prompt = BP_LLM_SYSTEM_PROMPT,
        temperature = 0.7,
        max_tokens  = 256,
        timeout_seconds = 60.0,
        fallback_op = Arborist.SubtreeMutation(),
        sections    = Arborist.AbstractPromptSection[Arborist.ElitesSection(3)],
    )
    llm_op.debug_log = debug_io

    inner_eval = BinPackingEvaluator(n_episodes=20, n_items=200, capacity=1.0f0,
                                      item_dist=:uniform, rng_seed=42)
    evaluator = BPMultiObjectiveEvaluator(inner_eval)

    fset = bin_packing_function_set()
    problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                  function_set=fset, num_temps=6, seed=42)

    algorithm = Arborist.NSGAII(
        pop_size=200, generations=200,
        mutation_rate=0.4, crossover_rate=0.3,
        mutation_ops=[TrackedMutation(llm_op), Arborist.SubtreeMutation(),
                      Arborist.PointMutation(), Arborist.HoistMutation(),
                      Arborist.ExpansionMutation()],
    )

    println("=" ^ 70)
    println("NSGA-II Unseeded Bin Packing — 3 objectives + qwen3-coder")
    println("  Objectives: fitness, coverage, success_rate")
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

    println("\n" * "=" ^ 70)
    println("Results after $(round(wall, digits=1))s")
    println("=" ^ 70)
    println("Pareto front size: $(length(result.pareto_front))")

    # Show the Pareto front sorted by primary fitness
    sorted_idx = sortperm(result.pareto_fitnesses, by=f -> f[1])
    println("\nTop 10 by primary fitness:")
    println("  # | fitness | coverage | success_rate | has_while | has_if")
    println("  --|---------|----------|--------------|-----------|-------")
    for (rank, i) in enumerate(sorted_idx[1:min(10, length(sorted_idx))])
        f = result.pareto_fitnesses[i]
        g = result.pareto_front[i]
        src = Arborist.serialize(g)
        hw = occursin("while", src) ? "yes" : "no"
        hi = occursin("if", src) ? "yes" : "no"
        println("  $(rank) | $(round(f[1], digits=4)) | $(round(-f[2], digits=3)) | $(round(-f[3], digits=3)) | $hw | $hi")
    end

    # Show highest-coverage programs
    cov_idx = sortperm(result.pareto_fitnesses, by=f -> f[2])  # most negative = highest coverage
    println("\nTop 5 by coverage:")
    for (rank, i) in enumerate(cov_idx[1:min(5, length(cov_idx))])
        f = result.pareto_fitnesses[i]
        g = result.pareto_front[i]
        src = Arborist.serialize(g)
        hw = occursin("while", src) ? "yes" : "no"
        println("  $(rank) | fitness=$(round(f[1], digits=4)) cov=$(round(-f[2], digits=3)) sr=$(round(-f[3], digits=3)) | while=$hw")
    end

    # Print best program by primary fitness
    best_i = sorted_idx[1]
    println("\nBest program (by primary fitness):")
    best = result.pareto_front[best_i]
    checked_body = Arborist.add_loop_checks(best.body; limit=1000)
    harness = Arborist.create_harness(best.state, checked_body)
    println(harness)

    # Print highest-coverage program
    cov_best_i = cov_idx[1]
    if cov_best_i != best_i
        println("\nHighest-coverage program:")
        cov_best = result.pareto_front[cov_best_i]
        checked_body = Arborist.add_loop_checks(cov_best.body; limit=1000)
        harness = Arborist.create_harness(cov_best.state, checked_body)
        println(harness)
    end

    # LLM stats
    s = llm_op.stats
    println("\nLLM Stats:")
    println("  Calls: $(s.total_calls), Successes: $(s.llm_successes), Failures: $(s.llm_failures)")
    flush(stdout)
end

main()
