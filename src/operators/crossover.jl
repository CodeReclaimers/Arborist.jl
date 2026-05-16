"""
    SubtreeCrossover(; max_depth=nothing, max_size=nothing)

Crossover operator that performs subtree crossover between two genomes.
Selects compatible subtrees (matching types) and swaps them between parents.

Optional `max_depth` / `max_size` cap the offspring. When an offspring
violates either cap, the corresponding parent is returned in its place —
Koza-style reject-and-retry bloat control. Caps default to `nothing`
(no limit) for behavior-preserving upgrade.
"""
struct SubtreeCrossover <: AbstractCrossoverOperator
    max_depth::Union{Int, Nothing}
    max_size::Union{Int, Nothing}
    SubtreeCrossover(; max_depth::Union{Int, Nothing}=nothing,
                       max_size::Union{Int, Nothing}=nothing) = new(max_depth, max_size)
end

"""
    crossover(op::SubtreeCrossover, g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG) -> Tuple{ExprGenome, ExprGenome}

Produce two offspring by performing subtree crossover on the body expressions
of two parent genomes. Falls back to copies of the parents if no compatible
subtree pair is found.
"""
function crossover(op::SubtreeCrossover, g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG)
    block_a = Expr(:block, deepcopy(g1.body)...)
    block_b = Expr(:block, deepcopy(g2.body)...)

    (offspring_a, offspring_b) = crossover(g1.state, block_a, block_b, rng)

    expr_a = offspring_a.head == :block ? collect(offspring_a.args) : [offspring_a]
    expr_b = offspring_b.head == :block ? collect(offspring_b.args) : [offspring_b]

    filter_exprs(v) = Expr[e for e in v if e isa Expr]

    body_a = filter_exprs(expr_a)
    body_b = filter_exprs(expr_b)

    if isempty(body_a)
        body_a = deepcopy(g1.body)
    end
    if isempty(body_b)
        body_b = deepcopy(g2.body)
    end

    child_a = _respect_caps(op, g1, ExprGenome(body_a, g1.state))
    child_b = _respect_caps(op, g2, ExprGenome(body_b, g1.state))
    return (child_a, child_b)
end
