#!/usr/bin/env julia
# NSGA-II bin packing tuning metric harness — OFF-LIMITS for autoloop subagent.
#
# This file is the measurement harness. It compiles the candidate NSGAII
# algorithm from examples/nsga2_tuning_config.jl (which IS the editable
# surface), runs three training seeds, evaluates each run's best Pareto
# member on the held-out test seed (train_seed + 1000), and prints the mean
# test fitness as the single scalar on the last non-empty line of stdout.
#
# Usage:
#   BP_POOL_SIZE=2000 julia --project=. -t 32 examples/nsga2_tuning_metric.jl
#   BP_POOL_SIZE=10000 julia --project=. -t 32 examples/nsga2_tuning_metric.jl --holdout
#
# Flags:
#   --holdout    Use 10k-pool behavioral init + 200 gen (full reproduction).
#                Without this flag, assumes BP_POOL_SIZE is already set by
#                the caller for a screen metric.
#   --seeds=a,b,c   Override the default training seed set (42,123,1337).
#   --generations=N Override the generation budget (default 200).

using Arborist
using Random
using Statistics

include(joinpath(@__DIR__, "bin_packing.jl"))
include(joinpath(@__DIR__, "nsga2_tuning_config.jl"))

function parse_args()
    seeds = [42, 123, 1337]
    generations = 200
    holdout = false
    for a in ARGS
        if a == "--holdout"
            holdout = true
        elseif startswith(a, "--seeds=")
            seeds = parse.(Int, split(a[9:end], ","))
        elseif startswith(a, "--generations=")
            generations = parse(Int, a[15:end])
        end
    end
    return (seeds=seeds, generations=generations, holdout=holdout)
end

"""Return test-set fitness for the best Pareto-front member of `result`."""
function _evaluate_best_on_test(result, train_seed::Int)
    # The "best" in a 2-objective Pareto front is the lowest-primary-fitness
    # member; this is the convention the bin_packing.jl runners use.
    best_idx = argmin(first.(result.pareto_fitnesses))
    best_genome = result.pareto_front[best_idx]

    test_eval = BinPackingEvaluator(
        n_episodes=20, n_items=200, capacity=1.0f0,
        item_dist=:uniform, rng_seed=train_seed + 1000
    )

    fname = gensym("bp_tuning_test")
    checked_body = Arborist.add_loop_checks(best_genome.body; limit=1000)
    harness = Arborist.create_harness(best_genome.state, checked_body, fname)
    f_test = @eval $harness
    return Arborist.evaluate(test_eval, f_test)
end

function main()
    cfg = parse_args()
    _ensure_bp_states()

    # Holdout mode forces the full 10k behavioral-init pool; screen mode
    # trusts the caller's BP_POOL_SIZE.
    if cfg.holdout
        ENV["BP_POOL_SIZE"] = "10000"
    end

    pool_size = get(ENV, "BP_POOL_SIZE", "10000")
    println("# NSGA-II tuning metric")
    println("#   seeds: $(cfg.seeds)")
    println("#   generations: $(cfg.generations)")
    println("#   pool_size: $pool_size")
    println("#   threads: $(Threads.nthreads())")
    println("#   mode: $(cfg.holdout ? "holdout" : "screen")")
    flush(stdout)

    test_fitnesses = Float64[]
    for seed in cfg.seeds
        inner_eval = BinPackingEvaluator(n_episodes=20, n_items=200,
                                          capacity=1.0f0, item_dist=:uniform,
                                          rng_seed=seed)
        evaluator = BPMultiObjectiveEvaluator(inner_eval)
        fset = bin_packing_function_set()

        algorithm, num_temps = build_tuning_algorithm(generations=cfg.generations)
        problem = Arborist.GPProblem(evaluator, Arborist.ExprGenome;
                                     function_set=fset, num_temps=num_temps,
                                     seed=seed)

        t0 = time()
        result = Arborist.solve(problem, algorithm; verbose=false)
        wall = time() - t0

        test_fit = _evaluate_best_on_test(result, seed)
        push!(test_fitnesses, test_fit)

        train_fit = minimum(first.(result.pareto_fitnesses))
        println("# seed=$seed  train=$(round(train_fit, digits=4))  test=$(round(test_fit, digits=4))  wall=$(round(wall, digits=1))s")
        flush(stdout)
    end

    mean_test = mean(test_fitnesses)
    std_test = length(test_fitnesses) > 1 ? std(test_fitnesses) : 0.0
    println("# mean_test=$(round(mean_test, digits=4))  std_test=$(round(std_test, digits=4))")
    # The autoloop metric: single scalar on the last non-empty line.
    println(mean_test)
    flush(stdout)
end

main()
