#!/usr/bin/env julia
# NSGA-II bin packing tuning config — EDITABLE SURFACE for autoloop.
#
# This file is the single editable surface for overnight NSGA-II hyperparameter
# tuning. It returns an NSGAII algorithm + the operator list + num_temps. The
# metric harness (examples/nsga2_tuning_metric.jl) calls build_tuning_algorithm
# once per seed and measures mean test fitness across 3 seeds.
#
# Allowed edits: any NSGAII kwarg (pop_size [must be even, ≥4], mutation_rate,
# crossover_rate, mutation_ops, crossover_ops), num_temps. Note: NSGA-II does
# not expose `elitism`, `selection`, `bloat_penalty`, or `speciation` — those
# are GeneticProgramming knobs only. Selection pressure in NSGA-II comes from
# non-dominated sorting + crowding distance (mu+lambda survivor selection).
#
# Off-limits edits (see autoloop/DIRECTION.md): the evaluator, the test
# harness, the baseline heuristics, n_episodes/n_items/capacity, BP_POOL_SIZE.

using Arborist

"""
    build_tuning_algorithm(; generations::Int) -> (algorithm, num_temps)

Return an NSGAII algorithm configured with the current tuning candidate. The
metric harness invokes this once per seed. `generations` is passed in by the
harness (screen vs holdout may use different budgets).
"""
function build_tuning_algorithm(; generations::Int)
    algorithm = Arborist.NSGAII(
        pop_size = 200,
        generations = generations,
        mutation_rate = 0.6,
        crossover_rate = 0.2,
        mutation_ops = Arborist.AbstractMutationOperator[
            Arborist.SubtreeMutation(),
            Arborist.SubtreeMutation(),
            Arborist.PointMutation(),
            Arborist.PointMutation(),
            Arborist.ExpansionMutation(),
            Arborist.ExpansionMutation(),
        ],
    )
    num_temps = 6
    return (algorithm, num_temps)
end
