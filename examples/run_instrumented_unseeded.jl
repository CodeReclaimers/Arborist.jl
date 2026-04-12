#!/usr/bin/env julia
# Instrumented unseeded bin-packing run with qwen3-coder:30b.
# Uses behavioral_initialize (10k pool) and debug logging.
#
# Usage: julia --project -t auto examples/run_instrumented_unseeded.jl

using Arborist
using Random
using Dates

include(joinpath(@__DIR__, "bin_packing.jl"))

function main()
    _ensure_bp_states()

    # Debug log for raw LLM I/O
    mkpath(joinpath(@__DIR__, "logs", "debug"))
    debug_path = joinpath(@__DIR__, "logs", "debug", "instrumented_unseeded.log")
    debug_io = open(debug_path, "w")
    println("Debug log: $debug_path")

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

    tracked = TrackedMutation(llm_op)
    ops = [tracked, Arborist.SubtreeMutation(), Arborist.PointMutation(),
           Arborist.HoistMutation(), Arborist.ExpansionMutation()]

    println("=" ^ 70)
    println("Instrumented Unseeded Bin Packing — qwen3-coder:30b + behavioral init")
    println("  Population: 200, Generations: 200")
    println("  Init: behavioral (10k pool)")
    println("  LLM sections: ElitesSection(3)")
    println("  Debug log: $debug_path")
    println("=" ^ 70)
    flush(stdout)

    r = run_bin_packing(;
        _common_kwargs(seed=42, generations=200, pop_size=200)...,
        mutation_ops=ops,
        init_mode=:behavioral,
        output_file="bin_packing_results_instrumented_unseeded.md",
    )

    close(debug_io)

    s = llm_op.stats
    println("\n" * "=" ^ 70)
    println("Final Stats")
    println("=" ^ 70)
    println("  LLM calls:     $(s.total_calls)")
    println("  LLM successes: $(s.llm_successes)")
    println("  LLM failures:  $(s.llm_failures)")
    println("  Best fitness:  $(round(r.result.best_fitness, digits=4))")
    println("  Test fitness:  $(round(r.evolved_test, digits=4))")

    best_text = r.program_text
    println("\n  Has while: $(occursin("while", best_text))")
    println("  Has if:    $(occursin("if", best_text))")
    println("\nBest program:")
    println(best_text)
    flush(stdout)
end

main()
