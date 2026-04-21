# neat_mutation.jl — NEAT mutation operators for GraphGenome.
#
# These operators expose the NEAT mutation primitives through the framework's
# `AbstractMutationOperator` dispatch so callers can build custom mutation mixes
# (e.g. weight-only, structural-only) for ablation studies. The bare
# `_mutate_*!` helpers in genome/graph_genome.jl remain the implementation;
# each operator here is a thin dispatch wrapper.
#
# `NEATDefaultMutation` reproduces the canonical Stanley & Miikkulainen (2002)
# branching (0.80 weight perturb / 0.10 weight replace / 0.05 add connection /
# 0.03 add node / 0.02 toggle) for drop-in use.

"""
    WeightPerturbMutation <: AbstractMutationOperator

Per-connection Gaussian weight perturbation. Each enabled connection has a
`perturb_prob` chance of being perturbed by `Normal(0, perturb_sigma)`.
Defaults match NEAT: `perturb_prob=0.9`, `perturb_sigma=0.3`.
"""
struct WeightPerturbMutation <: AbstractMutationOperator
    perturb_prob::Float64
    perturb_sigma::Float64
end

WeightPerturbMutation(; perturb_prob::Float64=0.9, perturb_sigma::Float64=0.3) =
    WeightPerturbMutation(perturb_prob, perturb_sigma)

function mutate(op::WeightPerturbMutation, g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    _mutate_weights!(new_g, rng; perturb_prob=op.perturb_prob, perturb_sigma=op.perturb_sigma)
    return new_g
end

"""
    WeightReplaceMutation <: AbstractMutationOperator

Full redraw of one random connection weight from `Normal(0, replace_sigma)`.
Default matches NEAT: `replace_sigma=2.0`.
"""
struct WeightReplaceMutation <: AbstractMutationOperator
    replace_sigma::Float64
end

WeightReplaceMutation(; replace_sigma::Float64=2.0) = WeightReplaceMutation(replace_sigma)

function mutate(op::WeightReplaceMutation, g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    _mutate_weight_replace!(new_g, rng; replace_sigma=op.replace_sigma)
    return new_g
end

"""
    AddConnectionMutation <: AbstractMutationOperator

Structural mutation: add a new enabled connection between two existing nodes.
Tries `max_attempts` random (from, to) pairs before giving up. Input/bias
nodes are never destinations; output nodes are never sources.
"""
struct AddConnectionMutation <: AbstractMutationOperator
    max_attempts::Int
end

AddConnectionMutation(; max_attempts::Int=20) = AddConnectionMutation(max_attempts)

function mutate(op::AddConnectionMutation, g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    _mutate_add_connection!(new_g, rng; max_attempts=op.max_attempts)
    return new_g
end

"""
    AddNodeMutation <: AbstractMutationOperator

Structural mutation: disable a random enabled connection and insert a new
hidden node on the split. The new node picks an activation uniformly from
`hidden_activations`. The incoming edge to the new node has weight 1.0; the
outgoing edge inherits the original connection's weight.
"""
struct AddNodeMutation <: AbstractMutationOperator
    hidden_activations::Vector{Symbol}
end

AddNodeMutation(; hidden_activations::Vector{Symbol}=Symbol[:sigmoid, :tanh, :relu]) =
    AddNodeMutation(hidden_activations)

function mutate(op::AddNodeMutation, g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    _mutate_add_node!(new_g, rng; hidden_activations=op.hidden_activations)
    return new_g
end

"""
    ToggleConnectionMutation <: AbstractMutationOperator

Flip the `enabled` flag on a random connection.
"""
struct ToggleConnectionMutation <: AbstractMutationOperator end

function mutate(::ToggleConnectionMutation, g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    _mutate_toggle_connection!(new_g, rng)
    return new_g
end

"""
    NEATCrossover <: AbstractCrossoverOperator

NEAT-style innovation-aligned crossover for `GraphGenome` (Stanley &
Miikkulainen, 2002). Matching genes are inherited randomly from either
parent; disjoint/excess genes come from the fitter parent. Produces two
children — the second swaps parent order so that the less-fit parent's
disjoint/excess genes are also preserved in the population.
"""
struct NEATCrossover <: AbstractCrossoverOperator end

function crossover(::NEATCrossover, g1::GraphGenome, g2::GraphGenome, rng::AbstractRNG)
    # Use cached fitness to pick the fitter parent (lower = better).
    if g1.fitness <= g2.fitness
        child1 = _neat_crossover(g1, g2, rng)
        child2 = _neat_crossover(g2, g1, rng)
    else
        child1 = _neat_crossover(g2, g1, rng)
        child2 = _neat_crossover(g1, g2, rng)
    end
    return (child1, child2)
end

"""
    NEATDefaultMutation <: AbstractMutationOperator

Composite operator that reproduces the canonical NEAT mutation branching
(Stanley & Miikkulainen, 2002). A single `mutate` call picks one branch by
the cumulative probabilities derived from the positive weights below.

Defaults: 0.80 weight perturb / 0.10 weight replace / 0.05 add connection /
0.03 add node / 0.02 toggle — matching the original bare `mutate(g, rng)`
implementation.

Each sub-operator's parameters are exposed via keyword arguments for tuning
(e.g. `perturb_sigma`, `max_attempts`); see the per-operator docstrings.
"""
struct NEATDefaultMutation <: AbstractMutationOperator
    weight_perturb::WeightPerturbMutation
    weight_replace::WeightReplaceMutation
    add_connection::AddConnectionMutation
    add_node::AddNodeMutation
    toggle::ToggleConnectionMutation
    # Cumulative thresholds over [0, 1), matching the original if-chain.
    thr_perturb::Float64
    thr_replace::Float64
    thr_add_conn::Float64
    thr_add_node::Float64
end

function NEATDefaultMutation(;
    weight_perturb_rate::Float64 = 0.80,
    weight_replace_rate::Float64 = 0.10,
    add_connection_rate::Float64 = 0.05,
    add_node_rate::Float64 = 0.03,
    toggle_rate::Float64 = 0.02,
    perturb_prob::Float64 = 0.9,
    perturb_sigma::Float64 = 0.3,
    replace_sigma::Float64 = 2.0,
    add_connection_max_attempts::Int = 20,
    hidden_activations::Vector{Symbol} = Symbol[:sigmoid, :tanh, :relu],
)
    total = weight_perturb_rate + weight_replace_rate + add_connection_rate +
            add_node_rate + toggle_rate
    if !(total ≈ 1.0)
        throw(ArgumentError(
            "NEATDefaultMutation rates must sum to 1.0 (got $total): " *
            "weight_perturb=$weight_perturb_rate, weight_replace=$weight_replace_rate, " *
            "add_connection=$add_connection_rate, add_node=$add_node_rate, toggle=$toggle_rate"))
    end
    thr_perturb  = weight_perturb_rate
    thr_replace  = thr_perturb + weight_replace_rate
    thr_add_conn = thr_replace + add_connection_rate
    thr_add_node = thr_add_conn + add_node_rate
    # toggle takes everything above thr_add_node (up to 1.0)
    NEATDefaultMutation(
        WeightPerturbMutation(; perturb_prob=perturb_prob, perturb_sigma=perturb_sigma),
        WeightReplaceMutation(; replace_sigma=replace_sigma),
        AddConnectionMutation(; max_attempts=add_connection_max_attempts),
        AddNodeMutation(; hidden_activations=hidden_activations),
        ToggleConnectionMutation(),
        thr_perturb, thr_replace, thr_add_conn, thr_add_node,
    )
end

function mutate(op::NEATDefaultMutation, g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    r = rand(rng)
    if r < op.thr_perturb
        _mutate_weights!(new_g, rng;
            perturb_prob=op.weight_perturb.perturb_prob,
            perturb_sigma=op.weight_perturb.perturb_sigma)
    elseif r < op.thr_replace
        _mutate_weight_replace!(new_g, rng; replace_sigma=op.weight_replace.replace_sigma)
    elseif r < op.thr_add_conn
        _mutate_add_connection!(new_g, rng; max_attempts=op.add_connection.max_attempts)
    elseif r < op.thr_add_node
        _mutate_add_node!(new_g, rng; hidden_activations=op.add_node.hidden_activations)
    else
        _mutate_toggle_connection!(new_g, rng)
    end
    return new_g
end

"""
    neat_defaults() -> NamedTuple

Return `(mutation_ops, crossover_ops)` populated with `NEATDefaultMutation()`
and `NEATCrossover()` for drop-in use with `GeneticProgramming`, `NSGAII`, or
`IslandModel` when evolving `GraphGenome`:

```julia
ops = neat_defaults()
algorithm = GeneticProgramming(; pop_size=150, generations=100,
                                mutation_ops=ops.mutation_ops,
                                crossover_ops=ops.crossover_ops)
```
"""
neat_defaults() = (
    mutation_ops  = AbstractMutationOperator[NEATDefaultMutation()],
    crossover_ops = AbstractCrossoverOperator[NEATCrossover()],
)
