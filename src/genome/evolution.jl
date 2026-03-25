# evolution.jl — evolutionary machinery from existing project.
#
# Modifications from original:
# - Removed `include("codegen.jl")` (included separately by the module)
# - Removed `abstract type FitnessEvaluator end` (replaced by AbstractEvaluator
#   via the `const FitnessEvaluator = AbstractEvaluator` alias in the module)
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


###################################################################################
# Population and evolution loop.
###################################################################################

"""
    Individual

A single program in the evolving population.

Fields:
- `expr`: body statements (not yet wrapped in a harness)
- `fitness`: lower is better; Inf means unevaluated or non-viable
- `age`: number of generations this individual has survived
"""
mutable struct Individual
    expr::Vector{Expr}
    fitness::Float64
    age::Int
end

"""
    Population

The evolving population of programs, along with shared state for
evaluation and code generation.
"""
mutable struct Population
    individuals::Vector{Individual}
    evaluator::FitnessEvaluator
    state::GenState
    generation::Int
end

"""
    Population(rng, evaluator, pop_size, initial_body_size, num_temps; fset)

Create a new population of random individuals.

- `rng`: random number generator to use for all stochastic operations
- `evaluator`: the fitness evaluator defining input/output signatures
- `pop_size`: number of individuals
- `initial_body_size`: number of random statements per individual
- `num_temps`: number of temporary variables in the GenState
"""
function Population(rng::AbstractRNG, evaluator::FitnessEvaluator, pop_size::Int,
                    initial_body_size::Int, num_temps::Int;
                    fset::Union{FunctionSet, Nothing}=nothing)
    inputs = input_signature(evaluator)
    outputs = output_signature(evaluator)

    if fset === nothing
        # Build a default function set with basic arithmetic.
        fset_actual = FunctionSet(Set{FunctionDetails}())
        for T in [Float32, Int32]
            for func in [:+, :-, :*, :/, :^]
                add!(fset_actual, func, 2, T, T)
            end
        end
        for func in [:cos, :sin, :tanh, :exp, :sign]
            add!(fset_actual, func, 1, Float32, Float32)
        end
        # Comparison operators for Bool-typed conditions.
        for T in [Float32, Int32]
            for op in [:>, :<, :(==), :!=, :>=, :<=]
                add!(fset_actual, op, 2, T, Bool)
            end
        end
    else
        fset_actual = fset
    end

    s = GenState(rng, fset_actual, inputs, outputs, num_temps)

    individuals = Vector{Individual}(undef, pop_size)
    for i in 1:pop_size
        body = [create_random_assignment(s) for _ in 1:initial_body_size]
        individuals[i] = Individual(body, Inf, 0)
    end

    return Population(individuals, evaluator, s, 0)
end

# Backward-compatible constructor using default RNG.
function Population(evaluator::FitnessEvaluator, pop_size::Int,
                    initial_body_size::Int, num_temps::Int;
                    fset::Union{FunctionSet, Nothing}=nothing)
    Population(Random.default_rng(), evaluator, pop_size, initial_body_size, num_temps; fset=fset)
end

"""
    evaluate_individual!(pop::Population, ind::Individual)

Wrap an individual's expression vector in a harness, @eval it, and
evaluate fitness. Assigns Inf on any failure.
"""
function evaluate_individual!(pop::Population, ind::Individual)
    fname = gensym("evolved")
    try
        checked_body = add_loop_checks(ind.expr)
        harness = create_harness(pop.state, checked_body, fname)
        f = @eval $harness
        ind.fitness = evaluate(pop.evaluator, f)
    catch e
        e isa InterruptException && rethrow()
        ind.fitness = Inf
    end
    return ind.fitness
end

"""
    tournament_select(pop::Population, tournament_size::Int) -> Individual

Select an individual via tournament selection (pick `tournament_size`
random individuals, return the one with best fitness).
"""
function tournament_select(pop::Population, tournament_size::Int)
    rng = pop.state.rng
    best = rand(rng, pop.individuals)
    for _ in 2:tournament_size
        challenger = rand(rng, pop.individuals)
        if challenger.fitness < best.fitness
            best = challenger
        end
    end
    return best
end

"""
    mutate_individual(s::GenState, ind::Individual) -> Individual

Create a mutated copy of an individual. Picks a random statement
in the body and applies mutate! to a deepcopy.
"""
function mutate_individual(s::GenState, ind::Individual)
    new_expr = deepcopy(ind.expr)
    if !isempty(new_expr)
        # Collect all mutable sub-expressions across the body.
        all_nodes = Expr[]
        for stmt in new_expr
            append!(all_nodes, unravel(stmt))
        end
        if !isempty(all_nodes)
            target = rand(s.rng, all_nodes)
            try
                mutate!(s, target)
            catch e
                e isa InterruptException && rethrow()
                # If mutation fails (e.g., no compatible lvalues), keep as-is.
            end
        end
    end
    return Individual(new_expr, Inf, 0)
end

"""
    crossover_individuals(s::GenState, a::Individual, b::Individual) -> Tuple{Individual, Individual}

Produce two offspring via subtree crossover on the body expressions.
"""
function crossover_individuals(s::GenState, a::Individual, b::Individual)
    block_a = Expr(:block, deepcopy(a.expr)...)
    block_b = Expr(:block, deepcopy(b.expr)...)

    (offspring_a, offspring_b) = crossover(s, block_a, block_b)

    # Extract statements from the resulting blocks.
    expr_a = offspring_a.head == :block ? collect(offspring_a.args) : [offspring_a]
    expr_b = offspring_b.head == :block ? collect(offspring_b.args) : [offspring_b]

    # Ensure all elements are Expr (filter out LineNumberNode etc.).
    filter_exprs(v) = [e for e in v if e isa Expr]

    new_a = filter_exprs(expr_a)
    new_b = filter_exprs(expr_b)

    # If crossover stripped all statements, fall back to the originals.
    if isempty(new_a)
        new_a = deepcopy(a.expr)
    end
    if isempty(new_b)
        new_b = deepcopy(b.expr)
    end

    return (Individual(new_a, Inf, 0), Individual(new_b, Inf, 0))
end

"""
    evolve!(pop::Population, generations::Int;
            mutation_rate=0.3, crossover_rate=0.3,
            elitism=2, verbose=false)

Run the evolutionary loop for the specified number of generations.

- `mutation_rate`: probability that a selected individual is mutated
- `crossover_rate`: probability that two selected individuals undergo crossover
- `elitism`: number of top individuals carried forward unchanged
- `verbose`: if true, print generation stats
"""
function evolve!(pop::Population, generations::Int;
                 mutation_rate::Float64=0.3,
                 crossover_rate::Float64=0.3,
                 elitism::Int=2,
                 verbose::Bool=false)
    pop_size = length(pop.individuals)
    rng = pop.state.rng

    for gen in 1:generations
        # Evaluate any unevaluated individuals.
        for ind in pop.individuals
            if ind.fitness == Inf && ind.age == 0
                evaluate_individual!(pop, ind)
            end
        end

        # Sort by fitness ascending (best first).
        sort!(pop.individuals, by=x -> x.fitness)

        if verbose
            best_fit = pop.individuals[1].fitness
            finite_fits = [ind.fitness for ind in pop.individuals if isfinite(ind.fitness)]
            mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
            println("Generation $(pop.generation + gen): best=$(round(best_fit, digits=6)), mean=$(round(mean_fit, digits=6))")
        end

        # Build next generation.
        next_gen = Vector{Individual}(undef, pop_size)

        # Elitism: carry top individuals forward.
        for i in 1:min(elitism, pop_size)
            elite = pop.individuals[i]
            next_gen[i] = Individual(deepcopy(elite.expr), elite.fitness, elite.age + 1)
        end

        # Fill the rest via tournament selection + genetic operators.
        idx = elitism + 1
        while idx <= pop_size
            r = rand(rng)
            if r < crossover_rate && idx + 1 <= pop_size
                # Crossover.
                p1 = tournament_select(pop, 3)
                p2 = tournament_select(pop, 3)
                (child1, child2) = crossover_individuals(pop.state, p1, p2)
                next_gen[idx] = child1
                next_gen[idx + 1] = child2
                idx += 2
            elseif r < crossover_rate + mutation_rate
                # Mutation.
                parent = tournament_select(pop, 3)
                next_gen[idx] = mutate_individual(pop.state, parent)
                idx += 1
            else
                # Reproduction (copy).
                parent = tournament_select(pop, 3)
                next_gen[idx] = Individual(deepcopy(parent.expr), Inf, 0)
                idx += 1
            end
        end

        pop.individuals = next_gen
    end

    pop.generation += generations

    # Final evaluation pass.
    for ind in pop.individuals
        if ind.fitness == Inf && ind.age == 0
            evaluate_individual!(pop, ind)
        end
    end
    sort!(pop.individuals, by=x -> x.fitness)

    return pop
end
