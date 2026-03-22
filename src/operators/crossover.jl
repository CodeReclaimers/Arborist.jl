"""
    SubtreeCrossover <: AbstractCrossoverOperator

Crossover operator that performs subtree crossover between two genomes.
Selects compatible subtrees (matching types) and swaps them between parents.
"""
struct SubtreeCrossover <: AbstractCrossoverOperator end

"""
    crossover(op::SubtreeCrossover, g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG) -> Tuple{ExprGenome, ExprGenome}

Produce two offspring by performing subtree crossover on the body expressions
of two parent genomes. Falls back to copies of the parents if no compatible
subtree pair is found.
"""
function crossover(op::SubtreeCrossover, g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG)
    block_a = Expr(:block, deepcopy(g1.body)...)
    block_b = Expr(:block, deepcopy(g2.body)...)

    (offspring_a, offspring_b) = crossover(g1.state, block_a, block_b)

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

    return (ExprGenome(body_a, g1.state), ExprGenome(body_b, g1.state))
end
