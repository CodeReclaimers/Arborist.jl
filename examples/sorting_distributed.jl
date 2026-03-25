#!/usr/bin/env julia
# examples/sorting_distributed.jl — Distributed island model on sorting
#
# Demonstrates that multiple worker processes can evolve sorting programs
# in parallel, and compares wall-clock time across three modes:
#   1. Sequential (in-process) islands
#   2. Synchronous distributed islands
#   3. Asynchronous distributed islands
#
# Usage:
#   julia --project examples/sorting_distributed.jl
#   julia --project examples/sorting_distributed.jl --n_islands=4

using Distributed

# Parse n_islands from CLI before adding workers
n_islands = 2
for arg in ARGS
    m = match(r"^--n_islands=(\d+)$", arg)
    m !== nothing && (global n_islands = parse(Int, m.captures[1]))
end

# Add workers and load sorting code on them
println("Adding $n_islands worker processes...")
flush(stdout)
added = addprocs(n_islands; exeflags="--project=$(Base.active_project())")
@everywhere include(joinpath(@__DIR__, "sorting.jl"))

using Arborist

const GENERATIONS = 50
const POP_SIZE = 50

function run_comparison(; n_islands::Int=2, seed::Int=42, verbose::Bool=true)
    println("=" ^ 70)
    println("Distributed Island Model — Sorting Benchmark")
    println("=" ^ 70)
    println("Islands: $n_islands, Pop/island: $POP_SIZE, Generations: $GENERATIONS")
    println("Workers: $(Distributed.nworkers())")
    flush(stdout)

    # Use SortingEvaluator directly (not curriculum) — simpler for demo,
    # no mutable curriculum state to coordinate across workers.
    evaluator = SortingEvaluator(
        n_episodes=20, array_length=6, loop_limit=360
    )
    fset = sorting_function_set()

    island_alg = GeneticProgramming(
        pop_size=POP_SIZE, generations=GENERATIONS,
        mutation_rate=0.4, crossover_rate=0.3,
        elitism=2, selection=TournamentSelection(4), max_depth=10,
        bloat_penalty=0.0005, parallel=false
    )

    problem = GPProblem(evaluator, ExprGenome;
                        function_set=fset, num_temps=8, seed=seed)

    # --- Sequential island model ---
    println("\n--- Sequential islands (in-process) ---")
    flush(stdout)
    seq_alg = IslandModel(
        n_islands=n_islands, island_algorithm=island_alg,
        migration_interval=10, migration_size=2,
        distributed=false
    )
    seq_result = solve(problem, seq_alg; verbose=verbose)
    println("Sequential: best=$(round(seq_result.best_fitness, digits=6)), " *
            "wall_time=$(round(seq_result.wall_time, digits=2))s")
    flush(stdout)

    # --- Sync distributed ---
    println("\n--- Sync distributed islands ---")
    flush(stdout)
    sync_alg = IslandModel(
        n_islands=n_islands, island_algorithm=island_alg,
        migration_interval=10, migration_size=2,
        distributed=true, async=false
    )
    sync_result = solve(problem, sync_alg; verbose=verbose)
    println("Sync distributed: best=$(round(sync_result.best_fitness, digits=6)), " *
            "wall_time=$(round(sync_result.wall_time, digits=2))s")
    flush(stdout)

    # --- Async distributed ---
    println("\n--- Async distributed islands ---")
    flush(stdout)
    async_alg = IslandModel(
        n_islands=n_islands, island_algorithm=island_alg,
        migration_interval=10, migration_size=2,
        distributed=true, async=true
    )
    async_result = solve(problem, async_alg; verbose=verbose)
    println("Async distributed: best=$(round(async_result.best_fitness, digits=6)), " *
            "wall_time=$(round(async_result.wall_time, digits=2))s")
    flush(stdout)

    # --- Summary ---
    println("\n" * "=" ^ 70)
    println("COMPARISON SUMMARY")
    println("=" ^ 70)
    println("Mode           | Best Fitness | Wall Time")
    println("---------------+-------------+----------")
    for (name, r) in [("Sequential", seq_result),
                       ("Sync Dist.", sync_result),
                       ("Async Dist.", async_result)]
        println("$(rpad(name, 15))| $(lpad(round(r.best_fitness, digits=6), 11)) | $(round(r.wall_time, digits=2))s")
    end
    println()
    seq_t = seq_result.wall_time
    sync_t = sync_result.wall_time
    async_t = async_result.wall_time
    println("Sync speedup vs sequential:  $(round(seq_t / sync_t, digits=2))x")
    println("Async speedup vs sequential: $(round(seq_t / async_t, digits=2))x")
    flush(stdout)
end

run_comparison(n_islands=n_islands)

# Cleanup
rmprocs(added)
