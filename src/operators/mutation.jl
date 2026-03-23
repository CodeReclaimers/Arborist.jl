"""
    SubtreeMutation <: AbstractMutationOperator

Mutation operator that replaces a randomly selected statement in the genome's
body with a new randomly generated statement.
"""
struct SubtreeMutation <: AbstractMutationOperator end

"""
    PointMutation <: AbstractMutationOperator

Mutation operator that selects a random sub-expression in the genome's body
and applies a local modification (e.g., changing a variable, literal, or operator).
"""
struct PointMutation <: AbstractMutationOperator end

"""
    HoistMutation <: AbstractMutationOperator

Mutation operator that selects a random non-leaf subtree and replaces it with
one of its own child subtrees, reducing program complexity by one level of
nesting. Falls back to `SubtreeMutation` if no non-leaf subtree exists.

This is a standard bloat-reduction operator.
"""
struct HoistMutation <: AbstractMutationOperator end

"""
    ExpansionMutation <: AbstractMutationOperator

Mutation operator that selects a random leaf node (Symbol or Number in rvalue
position) and wraps it in a randomly chosen function call from the function set.
Returns the genome unchanged if no suitable wrapping function exists.
"""
struct ExpansionMutation <: AbstractMutationOperator end

# --- SubtreeMutation ---

"""
    mutate(op::SubtreeMutation, g::ExprGenome, rng::AbstractRNG) -> ExprGenome

Replace a random statement in the genome body with a new randomly generated statement.
"""
function mutate(op::SubtreeMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if !isempty(new_body)
        idx = rand(rng, 1:length(new_body))
        new_body[idx] = create_random_statement(g.state)
    end
    return ExprGenome(new_body, g.state)
end

# --- PointMutation ---

"""
    mutate(op::PointMutation, g::ExprGenome, rng::AbstractRNG) -> ExprGenome

Select a random sub-expression in the genome body and apply a local modification
using the existing `mutate!` function from codegen.jl.
"""
function mutate(op::PointMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if !isempty(new_body)
        all_nodes = Expr[]
        for stmt in new_body
            append!(all_nodes, unravel(stmt))
        end
        if !isempty(all_nodes)
            target = rand(rng, all_nodes)
            try
                mutate!(g.state, target)
            catch
                # If mutation fails (e.g., no compatible lvalues), keep as-is.
            end
        end
    end
    return ExprGenome(new_body, g.state)
end

# --- HoistMutation ---

"""
    mutate(op::HoistMutation, g::ExprGenome, rng::AbstractRNG) -> ExprGenome

Select a random non-leaf subtree (an Expr with at least one Expr child) and
replace it with one of its Expr children. Falls back to SubtreeMutation if
no non-leaf subtree exists.
"""
function mutate(op::HoistMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if isempty(new_body)
        return ExprGenome(new_body, g.state)
    end

    # Collect all Expr nodes across the body.
    all_nodes = Expr[]
    for stmt in new_body
        append!(all_nodes, unravel(stmt))
    end

    # Filter to non-leaf subtrees (Expr with at least one Expr arg).
    non_leaf = [node for node in all_nodes if any(a -> a isa Expr, node.args)]

    if isempty(non_leaf)
        # Fall back to SubtreeMutation.
        return mutate(SubtreeMutation(), g, rng)
    end

    target = rand(rng, non_leaf)

    # Pick a random Expr child of the target.
    expr_children = [a for a in target.args if a isa Expr]
    child = deepcopy(rand(rng, expr_children))

    # Replace target in the body.
    # Check if target is a top-level statement.
    for i in 1:length(new_body)
        if new_body[i] === target
            new_body[i] = child
            return ExprGenome(new_body, g.state)
        end
    end

    # Otherwise, search and replace in nested expressions.
    for stmt in new_body
        if replace_subtree!(stmt, target, child)
            return ExprGenome(new_body, g.state)
        end
    end

    # Fallback: return unchanged.
    return ExprGenome(new_body, g.state)
end

# --- ExpansionMutation ---

"""
    mutate(op::ExpansionMutation, g::ExprGenome, rng::AbstractRNG) -> ExprGenome

Select a random leaf node in rvalue position and wrap it in a function call.
Returns the genome unchanged if no suitable wrapping function exists.
"""
function mutate(op::ExpansionMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if isempty(new_body)
        return ExprGenome(new_body, g.state)
    end

    # Collect all (parent_expr, arg_index) pairs where arg is a leaf in rvalue position.
    leaf_positions = Tuple{Expr, Int}[]
    for stmt in new_body
        _collect_leaf_positions!(leaf_positions, stmt, g.state)
    end

    if isempty(leaf_positions)
        return ExprGenome(new_body, g.state)
    end

    (parent, idx) = rand(rng, leaf_positions)
    leaf_value = parent.args[idx]

    # Try to wrap the leaf in a function call.
    wrapped = try
        wrap_rvalue(g.state, leaf_value)
    catch
        nothing
    end

    if wrapped !== nothing && wrapped !== leaf_value
        parent.args[idx] = wrapped
    end

    return ExprGenome(new_body, g.state)
end

"""
    _collect_leaf_positions!(positions, expr, state)

Recursively collect (parent_expr, arg_index) pairs for leaf nodes in rvalue
positions. Skips lvalues (first arg of assignments) and function names (first
arg of calls).
"""
function _collect_leaf_positions!(positions::Vector{Tuple{Expr, Int}}, expr::Expr, s::GenState)
    for i in 1:length(expr.args)
        a = expr.args[i]
        if a isa Expr
            _collect_leaf_positions!(positions, a, s)
        elseif (a isa Symbol || a isa Number)
            # Skip lvalues and function names.
            if (expr.head == :(=) && i == 1) || (expr.head == :call && i == 1)
                continue
            end
            # Check if we can determine the rvalue type.
            try
                get_rvalue_type(s, a)
                push!(positions, (expr, i))
            catch
                # Not a known rvalue (e.g., iterator variable), skip.
            end
        end
    end
end
