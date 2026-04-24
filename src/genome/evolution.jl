# evolution.jl — evolutionary machinery from existing project.
#
# Modifications from original:
# - Removed `include("codegen.jl")` (included separately by the module)
# - Removed `abstract type FitnessEvaluator end` (replaced by AbstractEvaluator
#   from abstractions.jl; Population/etc. now reference AbstractEvaluator directly)
# - Removed TableFitnessEvaluator and related methods (migrated to evaluators.jl)
# - All rand() calls now use explicit rng from GenState or passed parameters

###################################################################################
# Subtree crossover.
###################################################################################

"""
    replace_subtree!(tree::Expr, target::Expr, replacement::Expr) -> Bool

Replace the first occurrence of `target` (by object identity) in `tree`
with `replacement`. Returns true if a replacement was made, false otherwise.
"""
function replace_subtree!(tree::Expr, target::Expr, replacement::Expr)
    for i in 1:length(tree.args)
        if tree.args[i] isa Expr && tree.args[i] === target
            tree.args[i] = replacement
            return true
        elseif tree.args[i] isa Expr
            if replace_subtree!(tree.args[i], target, replacement)
                return true
            end
        end
    end
    return false
end

"""
    crossover(s::GenState, parent_a::Expr, parent_b::Expr) -> Tuple{Expr, Expr}

Perform subtree crossover between two parent expression trees.

Strategy:
- Flatten both trees with `unravel()`.
- Find pairs of sub-expressions with matching types (via `get_rvalue_type`).
- Pick a random compatible pair, deepcopy both parents, and swap the subtrees.
- If no compatible pair exists, return deepcopy of both parents unchanged.
"""
function crossover(s::GenState, parent_a::Expr, parent_b::Expr)
    nodes_a = unravel(parent_a)
    nodes_b = unravel(parent_b)

    # Build list of (index_in_a, index_in_b) pairs with matching rvalue types.
    compatible_pairs = Tuple{Int, Int}[]
    # Cache types for nodes_a to avoid redundant computation.
    types_a = Vector{Union{DataType, Nothing}}(undef, length(nodes_a))
    for (i, na) in enumerate(nodes_a)
        types_a[i] = try
            get_rvalue_type(s, na)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
    end

    for (j, nb) in enumerate(nodes_b)
        type_b = try
            get_rvalue_type(s, nb)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        type_b === nothing && continue
        for (i, type_a) in enumerate(types_a)
            type_a === nothing && continue
            if type_a == type_b
                push!(compatible_pairs, (i, j))
            end
        end
    end

    if isempty(compatible_pairs)
        return (deepcopy(parent_a), deepcopy(parent_b))
    end

    (idx_a, idx_b) = rand(s.rng, compatible_pairs)

    offspring_a = deepcopy(parent_a)
    offspring_b = deepcopy(parent_b)

    # Get the corresponding nodes in the copies via the same traversal order.
    copy_nodes_a = unravel(offspring_a)
    copy_nodes_b = unravel(offspring_b)

    # Extract subtrees to swap (deepcopy since each will be inserted into the other tree).
    subtree_from_a = deepcopy(copy_nodes_a[idx_a])
    subtree_from_b = deepcopy(copy_nodes_b[idx_b])

    # Replace in offspring_a: swap node at idx_a with subtree from b.
    if copy_nodes_a[idx_a] === offspring_a
        offspring_a = subtree_from_b
    else
        replace_subtree!(offspring_a, copy_nodes_a[idx_a], subtree_from_b)
    end

    # Replace in offspring_b: swap node at idx_b with subtree from a.
    if copy_nodes_b[idx_b] === offspring_b
        offspring_b = subtree_from_a
    else
        replace_subtree!(offspring_b, copy_nodes_b[idx_b], subtree_from_a)
    end

    return (offspring_a, offspring_b)
end
