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
