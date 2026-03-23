"""
DynExprExt — Package extension providing TreeGenome backed by
DynamicExpressions.jl Node{T} for fast vectorized evaluation.

TreeGenome is appropriate for pure function approximation (symbolic regression,
system identification). For programs requiring control flow, mutable state, or
side effects, use ExprGenome instead.
"""
module DynExprExt

using GenProg
using GenProg: AbstractGenome, AbstractEvaluator, AbstractMutationOperator,
              AbstractCrossoverOperator, GPProblem, GeneticProgramming,
              GPResult, TournamentSelection, NoSpeciation,
              SubtreeMutation, PointMutation, SubtreeCrossover,
              _tournament_select, _apply_speciation!, _init_species_state
import GenProg: mutate, crossover, distance, complexity, serialize, deserialize,
               solve, evaluate, evaluate_genome, input_signature, output_signature
using DynamicExpressions
using Random

# =============================================================================
# TreeGenome struct
# =============================================================================

"""
    TreeGenome{T} <: AbstractGenome

A genome backed by a DynamicExpressions.jl expression tree.
Supports fast vectorized evaluation over datasets without `@eval`.
Appropriate for pure function approximation problems.

Type parameter `T` is the numeric type of the expression (`Float32`
is recommended for most GP applications).

# Fields
- `tree::Node{T}`: the expression tree
- `operators::OperatorEnum`: operator configuration for evaluation
- `n_features::Int`: number of input features
"""
struct TreeGenome{T} <: AbstractGenome
    tree::Node{T}
    operators::OperatorEnum
    n_features::Int
end

# =============================================================================
# TreeFitnessEvaluator
# =============================================================================

"""
    TreeFitnessEvaluator{T} <: AbstractEvaluator

Fitness evaluator for `TreeGenome`. Evaluates the expression tree directly
over a data matrix without `@eval`. Dramatically faster than
`TableFitnessEvaluator` for large datasets.

# Fields
- `X::Matrix{T}`: input data, `n_features × n_samples`
- `y::Vector{T}`: target output, length `n_samples`
- `operators::OperatorEnum`: operator configuration
"""
struct TreeFitnessEvaluator{T} <: AbstractEvaluator
    X::Matrix{T}
    y::Vector{T}
    operators::OperatorEnum
end

input_signature(e::TreeFitnessEvaluator) = Dict(Symbol("x$i") => eltype(e.X) for i in 1:size(e.X, 1))
output_signature(e::TreeFitnessEvaluator{T}) where T = Dict(:y => T)

"""
    evaluate(e::TreeFitnessEvaluator{T}, g::TreeGenome{T}) -> Float64

Evaluate a TreeGenome against the data matrix. Returns mean squared error.
Returns `Inf` if evaluation throws or produces NaN/Inf values.
"""
function evaluate(e::TreeFitnessEvaluator{T}, g::TreeGenome{T}) where T
    try
        predictions = g.tree(e.X, e.operators)
        n = length(e.y)
        mse = zero(Float64)
        for i in 1:n
            d = Float64(predictions[i]) - Float64(e.y[i])
            mse += d * d
        end
        mse /= n
        return isfinite(mse) ? mse : Inf
    catch
        return Inf
    end
end

# Also support evaluate_genome dispatch for _evaluate_with_penalty
function evaluate_genome(g::TreeGenome{T}, e::TreeFitnessEvaluator{T}) where T
    evaluate(e, g)
end

# =============================================================================
# Version-safe operator accessors
# =============================================================================

# DynamicExpressions v2 stores operators in an `ops` field as (unary_tuple, binary_tuple).
# DynamicExpressions v1 stores them as separate type parameters.
# These accessors handle both.
function _get_unary_ops(operators::OperatorEnum)
    hasproperty(operators, :ops) ? operators.ops[1] : ()
end
function _get_binary_ops(operators::OperatorEnum)
    hasproperty(operators, :ops) ? operators.ops[2] : ()
end

# =============================================================================
# Random tree generation (ramped half-and-half)
# =============================================================================

"""Generate a random expression tree using either full or grow method."""
function _random_tree(rng::AbstractRNG, operators::OperatorEnum, n_features::Int,
                      ::Type{T}, depth::Int, method::Symbol) where T
    n_unary = length(_get_unary_ops(operators))
    n_binary = length(_get_binary_ops(operators))
    n_ops = n_unary + n_binary

    # Terminal condition
    if depth <= 0 || (method == :grow && n_ops > 0 && rand(rng) < 0.3 && depth < 4)
        return _random_terminal(rng, n_features, T)
    end

    if n_ops == 0
        return _random_terminal(rng, n_features, T)
    end

    # Choose operator
    op_choice = rand(rng, 1:n_ops)
    if op_choice <= n_binary
        # Binary operator
        l = _random_tree(rng, operators, n_features, T, depth - 1, method)
        r = _random_tree(rng, operators, n_features, T, depth - 1, method)
        return Node{T}(; op=UInt8(op_choice), l=l, r=r)
    else
        # Unary operator
        ui = op_choice - n_binary
        child = _random_tree(rng, operators, n_features, T, depth - 1, method)
        return Node{T}(; op=UInt8(ui), l=child)
    end
end

function _random_terminal(rng::AbstractRNG, n_features::Int, ::Type{T}) where T
    if n_features > 0 && rand(rng, Bool)
        return Node{T}(; feature=UInt16(rand(rng, 1:n_features)))
    else
        return Node{T}(; val=T(randn(rng)))
    end
end

# =============================================================================
# AbstractGenome interface for TreeGenome
# =============================================================================

function GenProg.mutate(g::TreeGenome{T}, rng::AbstractRNG) where T
    r = rand(rng, 1:3)
    if r == 1
        return _point_mutate(g, rng)
    elseif r == 2
        return _constant_perturb(g, rng)
    else
        return _hoist_mutate(g, rng)
    end
end

function GenProg.crossover(g1::TreeGenome{T}, g2::TreeGenome{T}, rng::AbstractRNG) where T
    _subtree_crossover(g1, g2, rng)
end

function GenProg.distance(g1::TreeGenome{T}, g2::TreeGenome{T}) where T
    # Node count difference. Known limitation: less semantically meaningful
    # than ExprGenome's symmetric-difference metric.
    Float64(abs(count_nodes(g1.tree) - count_nodes(g2.tree)))
end

function GenProg.complexity(g::TreeGenome{T}) where T
    Float64(count_nodes(g.tree))
end

function GenProg.serialize(g::TreeGenome{T}) where T
    string_tree(g.tree, g.operators)
end

function GenProg.deserialize(::Type{TreeGenome{T}}, s::String,
                             operators::OperatorEnum, n_features::Int) where T
    # Parsing expression trees from strings is complex.
    # Return nothing — the LLM operator would need a custom parser per format.
    return nothing
end

# --- Operator dispatches for TreeGenome ---

function GenProg.mutate(::SubtreeMutation, g::TreeGenome{T}, rng::AbstractRNG) where T
    mutate(g, rng)
end

function GenProg.mutate(::PointMutation, g::TreeGenome{T}, rng::AbstractRNG) where T
    mutate(g, rng)
end

function GenProg.crossover(::SubtreeCrossover, g1::TreeGenome{T}, g2::TreeGenome{T},
                           rng::AbstractRNG) where T
    crossover(g1, g2, rng)
end

# =============================================================================
# Mutation implementations
# =============================================================================

"""Point mutation: replace a random node with a new random subtree of depth <= 2."""
function _point_mutate(g::TreeGenome{T}, rng::AbstractRNG) where T
    new_tree = copy_node(g.tree)
    nodes = collect(new_tree)
    isempty(nodes) && return g

    target_idx = rand(rng, 1:length(nodes))
    replacement = _random_tree(rng, g.operators, g.n_features, T, 2, :grow)

    if target_idx == 1
        # Replace root
        return TreeGenome{T}(replacement, g.operators, g.n_features)
    end

    # Find parent of target and replace
    _replace_nth_node!(new_tree, target_idx, replacement)
    return TreeGenome{T}(new_tree, g.operators, g.n_features)
end

"""Constant perturbation: add Gaussian noise to a random constant."""
function _constant_perturb(g::TreeGenome{T}, rng::AbstractRNG) where T
    new_tree = copy_node(g.tree)
    constants, refs = get_scalar_constants(new_tree)
    if isempty(constants)
        # No constants to perturb, fall back to point mutation
        return _point_mutate(g, rng)
    end
    idx = rand(rng, 1:length(constants))
    constants[idx] += T(0.1) * T(randn(rng))
    set_scalar_constants!(new_tree, constants, refs)
    return TreeGenome{T}(new_tree, g.operators, g.n_features)
end

"""Hoist mutation: replace a subtree with one of its children."""
function _hoist_mutate(g::TreeGenome{T}, rng::AbstractRNG) where T
    new_tree = copy_node(g.tree)
    nodes = collect(new_tree)

    # Find non-leaf nodes
    non_leaf_indices = [i for (i, n) in enumerate(nodes) if n.degree > 0]
    if isempty(non_leaf_indices)
        return _point_mutate(g, rng)
    end

    target_idx = rand(rng, non_leaf_indices)
    target = nodes[target_idx]

    # Pick a child
    if target.degree == 2
        child = rand(rng, Bool) ? copy_node(target.l) : copy_node(target.r)
    else
        child = copy_node(target.l)
    end

    if target_idx == 1
        return TreeGenome{T}(child, g.operators, g.n_features)
    end

    _replace_nth_node!(new_tree, target_idx, child)
    return TreeGenome{T}(new_tree, g.operators, g.n_features)
end

"""Subtree crossover between two TreeGenomes."""
function _subtree_crossover(g1::TreeGenome{T}, g2::TreeGenome{T}, rng::AbstractRNG) where T
    t1 = copy_node(g1.tree)
    t2 = copy_node(g2.tree)

    nodes1 = collect(t1)
    nodes2 = collect(t2)

    idx1 = rand(rng, 1:length(nodes1))
    idx2 = rand(rng, 1:length(nodes2))

    sub1 = copy_node(nodes1[idx1])
    sub2 = copy_node(nodes2[idx2])

    # Replace in tree 1
    if idx1 == 1
        t1 = sub2
    else
        _replace_nth_node!(t1, idx1, sub2)
    end

    # Replace in tree 2
    if idx2 == 1
        t2 = sub1
    else
        _replace_nth_node!(t2, idx2, sub1)
    end

    return (TreeGenome{T}(t1, g1.operators, g1.n_features),
            TreeGenome{T}(t2, g1.operators, g1.n_features))
end

"""Replace the nth node (in pre-order traversal) with replacement."""
function _replace_nth_node!(tree::Node{T}, target_idx::Int, replacement::Node{T}) where T
    counter = Ref(0)
    _replace_nth_recursive!(tree, target_idx, replacement, counter)
end

function _replace_nth_recursive!(node::Node{T}, target_idx::Int,
                                  replacement::Node{T}, counter::Ref{Int}) where T
    counter[] += 1
    if node.degree >= 1
        counter_before_l = counter[]
        # Check if left child is the target
        if counter_before_l + 1 == target_idx
            # Count nodes in left subtree to advance counter
            left_count = count_nodes(node.l)
            counter[] += left_count
            node.l = replacement
            return true
        end
        if _replace_nth_recursive!(node.l, target_idx, replacement, counter)
            return true
        end
    end
    if node.degree == 2
        counter_before_r = counter[]
        if counter_before_r + 1 == target_idx
            right_count = count_nodes(node.r)
            counter[] += right_count
            node.r = replacement
            return true
        end
        if _replace_nth_recursive!(node.r, target_idx, replacement, counter)
            return true
        end
    end
    return false
end

# =============================================================================
# Solve method for TreeGenome
# =============================================================================

"""
    solve(problem::GPProblem{TreeGenome{T}}, algorithm::GeneticProgramming; ...) -> GPResult

Run genetic programming evolution with TreeGenome. Uses DynamicExpressions.jl
for fast vectorized evaluation without `@eval`.
"""
function GenProg.solve(problem::GPProblem{TreeGenome{T}, E},
                       algorithm::GeneticProgramming;
                       verbose::Bool = false,
                       callback = nothing) where {T, E<:TreeFitnessEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    evaluator = problem.evaluator
    ops = evaluator.operators
    n_feat = size(evaluator.X, 1)
    pop_size = algorithm.pop_size
    bp = algorithm.bloat_penalty

    # Initialize population with ramped half-and-half.
    genomes = Vector{TreeGenome{T}}(undef, pop_size)
    for i in 1:pop_size
        method = i <= pop_size ÷ 2 ? :full : :grow
        depth = 2 + (i % 3)  # depths 2, 3, 4
        tree = _random_tree(rng, ops, n_feat, T, depth, method)
        genomes[i] = TreeGenome{T}(tree, ops, n_feat)
    end

    # Evaluate initial population.
    fitnesses = fill(Inf, pop_size)
    for i in 1:pop_size
        raw = evaluate(evaluator, genomes[i])
        fitnesses[i] = (bp > 0.0 && isfinite(raw)) ? raw + bp * complexity(genomes[i]) : raw
    end

    # Initialize speciation state.
    species_state = _init_species_state(algorithm.speciation)

    fitness_history = Float64[]
    mean_history = Float64[]
    t0 = time()

    for gen in 1:algorithm.generations
        # Sort by fitness.
        order = sortperm(fitnesses)
        genomes = genomes[order]
        fitnesses = fitnesses[order]

        push!(fitness_history, fitnesses[1])
        finite_fits = filter(isfinite, fitnesses)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        if verbose
            println("Generation $gen: best=$(round(fitnesses[1], digits=6)), mean=$(round(mean_fit, digits=6))")
        end

        callback !== nothing && callback(gen, fitnesses[1], genomes[1])

        selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                  algorithm.speciation, species_state, rng)

        next_genomes = Vector{TreeGenome{T}}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        # Elitism.
        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = TreeGenome{T}(copy_node(genomes[i].tree), ops, n_feat)
            next_fitnesses[i] = fitnesses[i]
        end

        t_size = algorithm.selection isa TournamentSelection ?
                 algorithm.selection.tournament_size : algorithm.tournament_size

        idx = algorithm.elitism + 1
        while idx <= pop_size
            r = rand(rng)
            if r < algorithm.crossover_rate && idx + 1 <= pop_size
                p1 = _tournament_select(selection_fitnesses, t_size, rng)
                p2 = _tournament_select(selection_fitnesses, t_size, rng)
                op = rand(rng, algorithm.crossover_ops)
                (c1, c2) = crossover(op, genomes[p1], genomes[p2], rng)
                next_genomes[idx] = c1
                next_genomes[idx + 1] = c2
                idx += 2
            elseif r < algorithm.crossover_rate + algorithm.mutation_rate
                p_idx = _tournament_select(selection_fitnesses, t_size, rng)
                op = rand(rng, algorithm.mutation_ops)
                child = mutate(op, genomes[p_idx], rng)
                next_genomes[idx] = child
                idx += 1
            else
                p_idx = _tournament_select(selection_fitnesses, t_size, rng)
                next_genomes[idx] = TreeGenome{T}(copy_node(genomes[p_idx].tree), ops, n_feat)
                idx += 1
            end
        end

        # Evaluate new individuals.
        for i in (algorithm.elitism + 1):pop_size
            raw = evaluate(evaluator, next_genomes[i])
            next_fitnesses[i] = (bp > 0.0 && isfinite(raw)) ? raw + bp * complexity(next_genomes[i]) : raw
        end

        genomes = next_genomes
        fitnesses = next_fitnesses
    end

    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    wall_time = time() - t0

    return GPResult{TreeGenome{T}}(
        genomes[1], fitnesses[1], genomes,
        fitness_history, mean_history,
        algorithm.generations, wall_time,
        fitnesses[1] < 1.0
    )
end

end # module DynExprExt
