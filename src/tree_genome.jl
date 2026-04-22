# tree_genome.jl — TreeGenome backed by DynamicExpressions.jl Node{T}
# for fast vectorized evaluation without @eval.
#
# TreeGenome is appropriate for pure function approximation (symbolic regression,
# system identification). For programs requiring control flow, mutable state, or
# side effects, use ExprGenome instead.

using DynamicExpressions
using DynamicExpressions: eval_tree_array, get_scalar_constants, set_scalar_constants!

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

"""
    TreeGenomeContext{T}

Per-island state carrier for `TreeGenome` under `IslandModel`. Parallels
`GenState` for `ExprGenome`: both carry `.rng` so that island-loop sites
reading `state.rng` work uniformly, and both are the second element of
the tuple returned from `_initialize_population`.

The extra `operators` and `n_features` fields let `from_migrant`
reconstruct a `TreeGenome{T}` by invoking `deserialize` with the
destination island's operator enum.
"""
struct TreeGenomeContext{T}
    rng::AbstractRNG
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

"""
    evaluate_cases(g::TreeGenome{T}, e::TreeFitnessEvaluator{T}) -> Vector{Float64}

Return per-sample squared error as a `Vector{Float64}` of length
`length(e.y)`. Non-finite samples (NaN/Inf after vectorised evaluation)
are reported as `Inf`. All samples `Inf` on evaluation failure.
Used by lexicase selection.
"""
function evaluate_cases(g::TreeGenome{T}, e::TreeFitnessEvaluator{T}) where T
    n = length(e.y)
    case_fitnesses = fill(Inf, n)
    try
        predictions = g.tree(e.X, e.operators)
        @inbounds for i in 1:n
            d = Float64(predictions[i]) - Float64(e.y[i])
            se = d * d
            case_fitnesses[i] = isfinite(se) ? se : Inf
        end
    catch err
        err isa InterruptException && rethrow()
        # leave as Inf
    end
    return case_fitnesses
end

# =============================================================================
# Version-safe operator accessors
# =============================================================================

# DynamicExpressions v2 stores operators in an `ops` field as (unary_tuple, binary_tuple).
function _get_unary_ops(operators::OperatorEnum)
    operators.ops[1]
end
function _get_binary_ops(operators::OperatorEnum)
    operators.ops[2]
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

function mutate(g::TreeGenome{T}, rng::AbstractRNG) where T
    r = rand(rng, 1:3)
    if r == 1
        return _point_mutate(g, rng)
    elseif r == 2
        return _constant_perturb(g, rng)
    else
        return _hoist_mutate(g, rng)
    end
end

function crossover(g1::TreeGenome{T}, g2::TreeGenome{T}, rng::AbstractRNG) where T
    _subtree_crossover(g1, g2, rng)
end

function distance(g1::TreeGenome{T}, g2::TreeGenome{T}) where T
    # Node count difference. Known limitation: less semantically meaningful
    # than ExprGenome's symmetric-difference metric.
    Float64(abs(count_nodes(g1.tree) - count_nodes(g2.tree)))
end

function complexity(g::TreeGenome{T}) where T
    Float64(count_nodes(g.tree))
end

function serialize(g::TreeGenome{T}) where T
    string_tree(g.tree, g.operators)
end

"""
    deserialize(::Type{TreeGenome{T}}, s, operators, n_features) -> Union{TreeGenome{T}, Nothing}

Parse a string representation of an expression tree back into a
`TreeGenome{T}`. Accepts both the infix form emitted by
`serialize` / DynamicExpressions' `string_tree` (e.g. `x1 + 1.0`,
`sin((x1 + 1.0) * x2)`) and the prefix s-expression form used by
older code paths (e.g. `+(x1, 1.0)`, `sin(*(x1, x2))`). Both forms
are accepted because `Meta.parse` normalizes them to the same
`Expr(:call, ...)` structure that `_expr_to_node` walks.

Returns `nothing` for unparseable input, unrecognized operators, or
out-of-range feature indices. The caller is responsible for any
fallback behavior.
"""
function deserialize(::Type{TreeGenome{T}}, s::AbstractString,
                             operators::OperatorEnum, n_features::Int) where T
    stripped = strip(s)
    isempty(stripped) && return nothing
    expr = try
        Meta.parse(stripped)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    tree = _expr_to_node(expr, operators, n_features, T)
    tree === nothing && return nothing
    return TreeGenome{T}(tree, operators, n_features)
end

# --- Operator dispatches for TreeGenome ---

function mutate(::SubtreeMutation, g::TreeGenome{T}, rng::AbstractRNG) where T
    mutate(g, rng)
end

function mutate(::PointMutation, g::TreeGenome{T}, rng::AbstractRNG) where T
    mutate(g, rng)
end

function crossover(::SubtreeCrossover, g1::TreeGenome{T}, g2::TreeGenome{T},
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
            TreeGenome{T}(t2, g2.operators, g2.n_features))
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
function solve(problem::GPProblem{TreeGenome{T}, E},
                       algorithm::GeneticProgramming;
                       verbose::Bool = false,
                       callback = nothing,
                       log::Union{Nothing, RunLog} = nothing) where {T, E<:TreeFitnessEvaluator}
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

        species_snapshot = log === nothing ? nothing : SpeciationSnapshot()
        selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                  algorithm.speciation, species_state, rng;
                                                  snapshot=species_snapshot)

        case_fitnesses = needs_cases(algorithm.selection) ?
            _compute_case_fitnesses(genomes, evaluator, algorithm.parallel) :
            nothing

        if log !== nothing
            record!(log, gen, fitnesses, genomes, time() - t0;
                    snapshot=species_snapshot)
        end

        next_genomes = Vector{TreeGenome{T}}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        # Elitism.
        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                 algorithm, rng, algorithm.elitism + 1;
                                 case_fitnesses=case_fitnesses)

        # Evaluate new individuals.
        for i in (algorithm.elitism + 1):pop_size
            raw = evaluate(evaluator, next_genomes[i])
            next_fitnesses[i] = (bp > 0.0 && isfinite(raw)) ? raw + bp * complexity(next_genomes[i]) : raw
        end

        # Periodic constant-optimization pass on top-K individuals. Refines
        # numeric literals against the dataset via BFGS with FD gradients; see
        # `src/constant_optimization.jl`. Zero overhead when disabled.
        _maybe_optimize_constants!(next_genomes, next_fitnesses, evaluator,
                                   algorithm, gen, bp)

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
        fitnesses[1] < algorithm.convergence_threshold
    )
end

# =============================================================================
# IslandModel integration
# =============================================================================

# Extend the solve.jl helper with a TreeGenome method so IslandModel
# (sequential, sync distributed, async distributed) can initialize
# TreeGenome populations. The second element of the returned tuple is a
# TreeGenomeContext whose .rng field parallels GenState.rng for uniform
# use by the island loop.
function _initialize_population(problem::GPProblem{TreeGenome{T}, E},
                                 algorithm::GeneticProgramming,
                                 rng::AbstractRNG) where {T, E}
    evaluator = problem.evaluator
    evaluator isa TreeFitnessEvaluator || error(
        "IslandModel with TreeGenome requires a TreeFitnessEvaluator " *
        "(got $(typeof(evaluator)))")

    ops = evaluator.operators
    n_feat = size(evaluator.X, 1)
    pop_size = algorithm.pop_size

    genomes = Vector{TreeGenome{T}}(undef, pop_size)
    for i in 1:pop_size
        method = i <= pop_size ÷ 2 ? :full : :grow
        depth = 2 + (i % 3)  # depths 2, 3, 4
        tree = _random_tree(rng, ops, n_feat, T, depth, method)
        genomes[i] = TreeGenome{T}(tree, ops, n_feat)
    end

    return (genomes, TreeGenomeContext{T}(rng, ops, n_feat))
end

# =============================================================================
# TreeGenome migration
# =============================================================================

"""
    to_migrant(g::TreeGenome{T}, fitness::Float64) -> MigrantGenome

Pack a TreeGenome's `Node{T}` into a `MigrantGenome` for cross-island
(and cross-process) migration. The `Node{T}` is carried directly rather
than going through the string-based `serialize`/`deserialize` path:
direct transport avoids any parse ambiguity, preserves exact bit
patterns of `Float32` constants, and is independent of
DynamicExpressions' `string_tree` output format. Julia's `Distributed`
serializer handles `Node{T}` natively.

The destination island's `OperatorEnum` is reattached in `from_migrant`.
Op indices stored in `Node{T}` are stable across islands because every
island holds the same `OperatorEnum` built from the problem.
"""
function to_migrant(g::TreeGenome{T}, fitness::Float64) where T
    MigrantGenome(deepcopy(g.tree), fitness, :TreeGenome)
end

"""
    from_migrant(m::MigrantGenome, ctx::TreeGenomeContext{T}) -> TreeGenome{T}

Reconstruct a `TreeGenome{T}` from a `MigrantGenome` by wrapping the
transported `Node{T}` with the destination island's `operators` and
`n_features` from `ctx`.
"""
function from_migrant(m::MigrantGenome, ctx::TreeGenomeContext{T}) where T
    m.genome_type === :TreeGenome || throw(ArgumentError(
        "Expected TreeGenome migrant, got $(m.genome_type)"))
    tree = m.data::Node{T}
    return TreeGenome{T}(deepcopy(tree), ctx.operators, ctx.n_features)
end

# =============================================================================
# Expression-tree walker for deserialize
# =============================================================================

"""
    _expr_to_node(x, operators, n_features, T) -> Union{Node{T}, Nothing}

Walk a `Meta.parse`d Julia expression and build a DynamicExpressions
`Node{T}`. Accepts both infix (`x1 + 1.0`, what `string_tree` emits)
and prefix (`+(x1, 1.0)`, what older callers used) because `Meta.parse`
normalizes both forms to the same `Expr(:call, op, args...)` structure.
Returns `nothing` on any unrecognized construct, unknown operator,
or out-of-range feature index.
"""
function _expr_to_node(x, operators::OperatorEnum,
                       n_features::Int, ::Type{T}) where T
    # Numeric literal: widen to T (handles Int, Float32, Float64, ...).
    if x isa Number
        return Node{T}(; val=T(x))
    end

    # Feature variable: :x1, :x2, ...
    if x isa Symbol
        m = match(r"^x(\d+)$", String(x))
        m === nothing && return nothing
        feat = parse(Int, m.captures[1])
        (feat < 1 || feat > n_features) && return nothing
        return Node{T}(; feature=UInt16(feat))
    end

    # Everything else must be a call expression.
    x isa Expr || return nothing
    x.head === :call || return nothing
    length(x.args) >= 2 || return nothing

    op_sym = x.args[1]
    op_sym isa Symbol || return nothing
    n_args = length(x.args) - 1

    if n_args == 2
        for (bi, bop) in enumerate(_get_binary_ops(operators))
            if Symbol(bop) == op_sym || Symbol(nameof(bop)) == op_sym
                l = _expr_to_node(x.args[2], operators, n_features, T)
                r = _expr_to_node(x.args[3], operators, n_features, T)
                (l === nothing || r === nothing) && return nothing
                return Node{T}(; op=UInt8(bi), l=l, r=r)
            end
        end
        return nothing
    elseif n_args == 1
        for (ui, uop) in enumerate(_get_unary_ops(operators))
            if Symbol(uop) == op_sym || Symbol(nameof(uop)) == op_sym
                child = _expr_to_node(x.args[2], operators, n_features, T)
                child === nothing && return nothing
                return Node{T}(; op=UInt8(ui), l=child)
            end
        end
        return nothing
    end

    return nothing
end

# =============================================================================
# SymbolicRegressionEvaluator convenience constructor
# =============================================================================

"""
    _default_operators(::Type{T}) -> OperatorEnum

Default operator set for symbolic regression.
"""
function _default_operators(::Type{T}) where T
    OperatorEnum(; binary_operators=[+, -, *, /],
                   unary_operators=[sin, cos, exp, abs])
end

"""
    SymbolicRegressionEvaluator(f; domain, points=20, operators=_default_operators(Float32), noise=0.0)

Convenience constructor for symbolic regression problems. Generates a
`TreeFitnessEvaluator` from a Julia function and domain specification.

# Arguments
- `f`: Target function (univariate: accepts `Float32`, multivariate: accepts `Vector{Float32}`)
- `domain`: `Tuple{T,T}` for univariate, `Vector{Tuple{T,T}}` for multivariate
- `points`: Sample points per dimension (default: 20)
- `operators`: `OperatorEnum` (default: +, -, *, / with sin, cos, exp, abs)
- `noise`: Gaussian noise standard deviation to add to targets (default: 0.0)
"""
function SymbolicRegressionEvaluator(f;
    domain::Union{Tuple, Vector},
    points::Int = 20,
    operators::OperatorEnum = _default_operators(Float32),
    noise::Float64 = 0.0
)
    if domain isa Tuple
        # Univariate
        lo, hi = Float32.(domain)
        xs = Float32.(range(lo, hi, length=points))
        X = reshape(xs, 1, :)
        y = Float32[f(x) for x in xs]
    else
        # Multivariate: domain is Vector of Tuples
        n_features = length(domain)
        grids = [Float32.(range(Float32(d[1]), Float32(d[2]), length=points)) for d in domain]
        # Use random sampling for multivariate (grid is exponential)
        rng = Random.MersenneTwister(42)
        n_samples = points * n_features
        X = zeros(Float32, n_features, n_samples)
        for j in 1:n_samples
            for i in 1:n_features
                X[i, j] = Float32(domain[i][1]) + rand(rng, Float32) * Float32(domain[i][2] - domain[i][1])
            end
        end
        y = Float32[f(X[:, j]) for j in 1:n_samples]
    end

    if noise > 0.0
        rng = Random.MersenneTwister(123)
        y .+= Float32.(noise .* randn(rng, length(y)))
    end

    return TreeFitnessEvaluator(X, y, operators)
end

# =============================================================================
# Periodic constant optimization (see src/constant_optimization.jl for config)
# =============================================================================
#
# Local BFGS on numeric literals in a TreeGenome's expression tree against a
# TreeFitnessEvaluator's dataset. Central finite-difference gradients, Armijo
# backtracking line search. Zygote-free (central FD avoids needing an AD
# package; DynamicExpressions' `eval_grad_tree_array` errors without Zygote
# and `differentiable_eval_tree_array` does not actually return gradients).

"""
    optimize_constants!(g::TreeGenome, e::TreeFitnessEvaluator; kwargs...) -> Float64

Apply BFGS with central finite-difference gradients to the constants of `g.tree`
against `e`'s dataset. Mutates `g.tree` in place; returns the post-optimization
MSE loss.

Returns the pre-optimization loss unchanged if the tree has zero constants or
if the initial evaluation produces `Inf` / `NaN`. Never makes the tree worse:
if BFGS diverges or line search fails, constants are restored and the original
loss is returned.

# Keyword arguments
- `max_iter::Int = 50`
- `tol::Float64 = 1e-8`
- `fd_step::Float64 = 1e-3`
"""
function optimize_constants!(g::TreeGenome{T}, e::TreeFitnessEvaluator{T};
                             max_iter::Int = 50,
                             tol::Float64 = 1e-8,
                             fd_step::Float64 = 1e-3) where T
    constants, refs = get_scalar_constants(g.tree)
    n = length(constants)
    initial_loss = _tree_mse(g.tree, e)
    n == 0 && return initial_loss
    isfinite(initial_loss) || return initial_loss

    initial_constants = copy(constants)
    h = T(fd_step)

    function f_and_grad(c::Vector{T})
        set_scalar_constants!(g.tree, c, refs)
        loss = _tree_mse(g.tree, e)
        isfinite(loss) || return (Inf, fill(Inf, n))
        grad = Vector{Float64}(undef, n)
        for i in 1:n
            orig = c[i]
            c[i] = orig + h
            set_scalar_constants!(g.tree, c, refs)
            loss_p = _tree_mse(g.tree, e)
            c[i] = orig - h
            set_scalar_constants!(g.tree, c, refs)
            loss_m = _tree_mse(g.tree, e)
            c[i] = orig
            grad[i] = (isfinite(loss_p) && isfinite(loss_m)) ?
                (loss_p - loss_m) / (2.0 * Float64(h)) : 0.0
        end
        set_scalar_constants!(g.tree, c, refs)
        return (loss, grad)
    end

    c_opt, loss_opt = _bfgs_minimize(f_and_grad, constants;
                                     max_iter=max_iter, tol=tol)

    # Accept only strict improvement — guard against roundoff / divergence.
    if isfinite(loss_opt) && loss_opt < initial_loss
        set_scalar_constants!(g.tree, c_opt, refs)
        return loss_opt
    else
        set_scalar_constants!(g.tree, initial_constants, refs)
        return initial_loss
    end
end

# Forward MSE on the evaluator's dataset. Returns Inf on evaluation failure.
function _tree_mse(tree::Node{T}, e::TreeFitnessEvaluator{T}) where T
    try
        predictions, complete = eval_tree_array(tree, e.X, e.operators)
        complete || return Inf
        n = length(e.y)
        sse = 0.0
        @inbounds for i in 1:n
            d = Float64(predictions[i]) - Float64(e.y[i])
            sse += d * d
        end
        mse = sse / n
        return isfinite(mse) ? mse : Inf
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end
end

"""
    _bfgs_minimize(f_and_grad, x0; max_iter, tol) -> (x_best, f_best)

Dense BFGS with Armijo backtracking. For small n (TreeGenome SR rarely
exceeds 20 constants) the dense inverse-Hessian is cheap.
"""
function _bfgs_minimize(f_and_grad::F, x0::Vector{T};
                        max_iter::Int, tol::Float64) where {F, T}
    x = copy(x0)
    n = length(x)
    H = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        H[i, j] = (i == j) ? 1.0 : 0.0
    end

    f, g = f_and_grad(x)
    (isfinite(f) && all(isfinite, g)) || return (x, f)
    best_x = copy(x)
    best_f = f

    for _ in 1:max_iter
        gnorm = sqrt(sum(v -> v * v, g))
        gnorm < tol && break

        p = -H * g
        # Safeguard: non-descent direction → reset Hessian, use steepest descent.
        if _co_dot(g, p) >= 0.0
            p = -Vector{Float64}(g)
            @inbounds for i in 1:n, j in 1:n
                H[i, j] = (i == j) ? 1.0 : 0.0
            end
        end

        alpha = 1.0
        c1 = 1e-4
        x_trial = copy(x)
        f_new = f
        g_new = g
        accepted = false
        for _ in 1:30
            @inbounds for i in 1:n
                x_trial[i] = x[i] + T(alpha * p[i])
            end
            f_new, g_new = f_and_grad(x_trial)
            if isfinite(f_new) && f_new <= f + c1 * alpha * _co_dot(g, p)
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break

        # BFGS inverse-Hessian rank-2 update.
        s = Vector{Float64}(undef, n)
        y = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            s[i] = Float64(x_trial[i] - x[i])
            y[i] = g_new[i] - g[i]
        end
        sy = _co_dot(s, y)
        if sy > 1e-12
            rho = 1.0 / sy
            Hy = H * y
            sHy = _co_dot(s, Hy)
            @inbounds for i in 1:n, j in 1:n
                H[i, j] += (rho + rho * rho * sHy) * s[i] * s[j] -
                           rho * (Hy[i] * s[j] + s[i] * Hy[j])
            end
        end

        @inbounds for i in 1:n
            x[i] = x_trial[i]
        end
        f = f_new
        g = g_new
        if f < best_f
            best_f = f
            @inbounds for i in 1:n
                best_x[i] = x[i]
            end
        end
    end

    return (best_x, best_f)
end

# Dot product local to the constant-opt module — avoids a LinearAlgebra dep.
function _co_dot(a::AbstractVector, b::AbstractVector)
    s = 0.0
    @inbounds for i in eachindex(a, b)
        s += Float64(a[i]) * Float64(b[i])
    end
    return s
end

"""
    _maybe_optimize_constants!(genomes, fitnesses, evaluator, algorithm, gen, bp)

If `algorithm.constant_optimization` is non-nothing and this generation is a
multiple of its `frequency`, run `optimize_constants!` on the top-K individuals.
Mutates `genomes[top-K]` and `fitnesses[top-K]` in place; preserves them
otherwise. Zero overhead when disabled.
"""
function _maybe_optimize_constants!(genomes::Vector{TreeGenome{T}},
                                    fitnesses::Vector{Float64},
                                    evaluator::TreeFitnessEvaluator{T},
                                    algorithm::GeneticProgramming,
                                    gen::Int,
                                    bp::Float64) where T
    cfg = algorithm.constant_optimization
    cfg === nothing && return nothing
    gen % cfg.frequency == 0 || return nothing

    order = sortperm(fitnesses)
    k = min(cfg.top_k, length(genomes))
    for rank in 1:k
        i = order[rank]
        g_copy = deepcopy(genomes[i])
        raw_after = optimize_constants!(g_copy, evaluator;
                                        max_iter=cfg.max_iter,
                                        tol=cfg.tol,
                                        fd_step=cfg.fd_step)
        fit_after = (bp > 0.0 && isfinite(raw_after)) ?
            raw_after + bp * complexity(g_copy) : raw_after
        if isfinite(fit_after) && fit_after < fitnesses[i]
            genomes[i] = g_copy
            fitnesses[i] = fit_after
        end
    end
    return nothing
end

"""
Default system prompt for TreeGenome LLM mutation (prefix notation).
"""
const DEFAULT_TREE_GP_SYSTEM_PROMPT = """
You are a genetic programming mutation operator for mathematical
expression trees. You will be given a mathematical expression in
prefix notation. Your task is to produce a meaningfully modified
variant that might better approximate the target function.

Rules:
- Return ONLY a single expression in prefix notation
- Use only these variable names: x1, x2, ..., xN
- Use floating-point constants where appropriate
- Do not include explanations, comments, or multiple expressions

Example input:  +(x1, *(2.0, x1))
Example output: +(*(x1, x1), *(3.0, x1))

Respond with only the prefix expression and nothing else.
"""

