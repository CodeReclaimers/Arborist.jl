# adf_genome.jl — Automatically Defined Functions (Koza 1994).
#
# An `ADFGenome{T}` carries a main expression tree plus N "automatically
# defined function" trees. Each ADF takes a fixed number of arguments
# (binary in this implementation: arity 2) and may be invoked from the
# main tree via reserved operator slots.
#
# Implementation strategy: **macro expansion**. ADF calls in the main
# tree are expanded inline (each call replaced by a copy of the ADF's
# body with ARG references substituted) before evaluation. This sidesteps
# the need for a custom interpreter or per-call thread-local state, at
# the cost of tree size growing with ADF body × number of calls. For
# Koza-style problems with 1-2 ADFs called a few times, the expansion is
# small.
#
# ARG encoding: ARG_i references inside an ADF body are encoded as feature
# indices above the user's `n_features`:
#   ARG0 := feature index n_features + 1
#   ARG1 := feature index n_features + 2
# `_random_adf_tree` enforces this when generating ADF bodies; main-tree
# generation uses only features [1, n_features].
#
# ADF call encoding: the OperatorEnum used for ADFGenome trees is
# extended at construction with N placeholder binary operators occupying
# slots `n_base_binary + 1 .. n_base_binary + n_adfs`. When the main tree
# uses one of these slots, expansion replaces the call with the
# corresponding ADF body (with its ARGs substituted). The placeholder
# function bodies should never be invoked — expansion runs first; if one
# is invoked it raises a clear error.
#
# Scope: this F.7 implementation supports fixed binary arity for all ADFs.
# Per-ADF variable arity and ADL (Automatically Defined Loops) are out of scope.

using DynamicExpressions
using DynamicExpressions: Node, OperatorEnum, eval_tree_array, copy_node,
                          count_nodes, count_depth

"""
    ADFGenome{T} <: AbstractGenome

Genome with a main expression tree and `N = length(adfs)` Automatically
Defined Function trees. Each ADF is a `Node{T}` that may reference the
ARG slots `ARG0..ARG{arity-1}` (encoded as features above the user's
`n_features`).

The main tree may invoke any ADF as a binary operator at slot
`n_base_binary + i`. ADFs themselves may reference features [1, n_features]
and ARG slots; nested ADF-from-ADF calls are not currently supported
(ADF body is generated without ADF placeholders).

# Fields
- `main::Node{T}`: main expression tree.
- `adfs::Vector{Node{T}}`: ADF body trees, one per ADF.
- `arity::Int`: shared arity of every ADF (default 2).
- `operators::OperatorEnum`: the *augmented* operator enum (base operators
  + N ADF placeholders). Use `base_operators(g)` to recover the user's
  original operator set.
- `n_features::Int`: number of real input features. ARG slots occupy
  `[n_features+1, n_features+arity]`.
- `n_adfs::Int`: convenience — `length(adfs)`.
"""
struct ADFGenome{T} <: AbstractGenome
    main::Node{T}
    adfs::Vector{Node{T}}
    arity::Int
    operators::OperatorEnum
    n_features::Int
    n_adfs::Int
end

# Stable error: invoked if expansion is skipped for any reason and an ADF
# placeholder ends up evaluated. Carries no payload — the error message
# alone is the contract.
function _adf_placeholder(a, b)
    error("ADF placeholder operator was invoked; expand_adfs must run before " *
          "eval_tree_array on an ADFGenome's main tree.")
end

"""
    augmented_operators(base::OperatorEnum, n_adfs::Int) -> OperatorEnum

Build the operator enum used by an `ADFGenome`'s trees: the user's binary
operators followed by `n_adfs` placeholder binary operators (one per ADF).
ADF body trees and the main tree share this enum. The placeholders are
never actually invoked — `expand_adfs` rewrites them before evaluation.
"""
function augmented_operators(base::OperatorEnum, n_adfs::Int)
    base_binary = collect(_get_binary_ops(base))
    base_unary  = collect(_get_unary_ops(base))
    placeholders = [_adf_placeholder for _ in 1:n_adfs]
    return OperatorEnum(binary_operators = vcat(base_binary, placeholders),
                        unary_operators  = base_unary)
end

"""
    initialize(::Type{ADFGenome{T}}, base_ops, n_features, n_adfs;
               arity=2, max_depth=4, rng) -> ADFGenome{T}

Construct a random `ADFGenome` with `n_adfs` ADFs. Main tree uses base
operators plus ADF placeholders; ADF bodies use only base operators
(nested ADF calls are not generated). ADF bodies may reference ARG slots
in addition to the user's features.
"""
function initialize_adf(::Type{T}, base_ops::OperatorEnum,
                        n_features::Int, n_adfs::Int;
                        arity::Int = 2,
                        max_depth::Int = 4,
                        rng::AbstractRNG) where T
    arity == 2 || throw(ArgumentError(
        "ADFGenome currently supports arity=2 only (got $arity)"))
    n_features >= 1 || throw(ArgumentError("n_features must be >= 1"))
    n_adfs >= 0 || throw(ArgumentError("n_adfs must be >= 0"))

    aug_ops = augmented_operators(base_ops, n_adfs)

    # ADF bodies: random tree using only base operators, may reference
    # features [1, n_features] AND ARG slots [n_features+1, n_features+arity].
    # We reuse `_random_tree` from tree_genome.jl with an extended n_features.
    adfs = Vector{Node{T}}(undef, n_adfs)
    for i in 1:n_adfs
        adfs[i] = _random_adf_body(rng, base_ops, n_features, arity, T, max_depth)
    end

    # Main tree: uses augmented ops (so ADF calls are reachable), references
    # only real features [1, n_features].
    main = _random_tree(rng, aug_ops, n_features, T, max_depth, :grow)

    return ADFGenome{T}(main, adfs, arity, aug_ops, n_features, n_adfs)
end

# ADF body generator: like _random_tree but uses base_ops (no ADF
# placeholders) and considers features in [1, n_features+arity] (real
# features + ARG slots).
function _random_adf_body(rng::AbstractRNG, base_ops::OperatorEnum,
                          n_features::Int, arity::Int,
                          ::Type{T}, depth::Int) where T
    extended_n = n_features + arity
    return _random_tree(rng, base_ops, extended_n, T, depth, :grow)
end

# ---------------------------------------------------------------------------
# Macro expansion: rewrite main tree's ADF calls inline.
# ---------------------------------------------------------------------------

"""
    expand_adfs(g::ADFGenome{T}) -> Node{T}

Produce a fully-expanded copy of `g.main` where every ADF call has been
replaced by the corresponding ADF body with ARG references substituted
for the call's argument subtrees. The result uses only the base operators
(no placeholders) and references only real features `[1, n_features]`.
Suitable for direct `eval_tree_array` evaluation against the user's
`base_operators`.
"""
function expand_adfs(g::ADFGenome{T}) where T
    n_base_binary = num_binary_ops(g.operators) - g.n_adfs
    return _expand_node(g.main, g.adfs, g.n_features, g.arity, n_base_binary)
end

# Walk a node, expanding ADF calls.
function _expand_node(node::Node{T}, adfs, n_features, arity, n_base_binary) where T
    if node.degree == 0
        # Constant or feature — copy as is. Features in main tree are real
        # features [1, n_features]; ARG slots can't appear in main.
        return copy_node(node)
    elseif node.degree == 1
        # Unary op — recurse into child.
        new_child = _expand_node(node.l, adfs, n_features, arity, n_base_binary)
        return Node{T}(; op=node.op, l=new_child)
    else
        # Binary op. Check if it's an ADF call.
        op_idx = Int(node.op)
        if op_idx > n_base_binary
            adf_idx = op_idx - n_base_binary
            # First expand the argument subtrees (they themselves may
            # contain ADF calls if we ever allow nested ADF-from-main).
            arg_l = _expand_node(node.l, adfs, n_features, arity, n_base_binary)
            arg_r = _expand_node(node.r, adfs, n_features, arity, n_base_binary)
            args = (arg_l, arg_r)
            # Substitute ARG references in the ADF body.
            return _substitute_args(adfs[adf_idx], args, n_features, T)
        else
            # Plain base binary operator.
            new_l = _expand_node(node.l, adfs, n_features, arity, n_base_binary)
            new_r = _expand_node(node.r, adfs, n_features, arity, n_base_binary)
            return Node{T}(; op=node.op, l=new_l, r=new_r)
        end
    end
end

# Substitute ARG references in an ADF body with copies of the corresponding
# argument subtrees. ARG_i is encoded as feature index n_features + i + 1
# (1-indexed, so ARG0 == n_features + 1, ARG1 == n_features + 2, ...).
function _substitute_args(adf_body::Node{T}, args, n_features, ::Type{T}) where T
    if adf_body.degree == 0
        if iszero(adf_body.constant)
            # Feature node. If it's an ARG slot, replace with the arg subtree.
            feat = Int(adf_body.feature)
            if feat > n_features
                arg_idx = feat - n_features
                if 1 <= arg_idx <= length(args)
                    return copy_node(args[arg_idx])
                end
            end
            return copy_node(adf_body)
        else
            return copy_node(adf_body)
        end
    elseif adf_body.degree == 1
        new_child = _substitute_args(adf_body.l, args, n_features, T)
        return Node{T}(; op=adf_body.op, l=new_child)
    else
        new_l = _substitute_args(adf_body.l, args, n_features, T)
        new_r = _substitute_args(adf_body.r, args, n_features, T)
        return Node{T}(; op=adf_body.op, l=new_l, r=new_r)
    end
end

# Note on the constant/feature distinction: DynamicExpressions' Node uses
# `constant::Bool` to discriminate. iszero(true) == false, iszero(false) ==
# true — so `iszero(adf_body.constant)` is true when the leaf is a feature.

# ---------------------------------------------------------------------------
# AbstractGenome interface
# ---------------------------------------------------------------------------

function complexity(g::ADFGenome)
    Float64(count_nodes(g.main) + sum(count_nodes(adf) for adf in g.adfs; init=0))
end

"""
    tree_depth(g::ADFGenome) -> Int

Maximum of `count_depth` across the main tree and every ADF body. This
captures the worst-case depth a caller might evaluate post-expansion; it
does not account for expansion-driven inlining, which can compose depths
up to `depth(main) + depth(any_adf) - 1` in the fully expanded form.
"""
function tree_depth(g::ADFGenome)
    d = count_depth(g.main)
    for adf in g.adfs
        dd = count_depth(adf)
        dd > d && (d = dd)
    end
    return d
end

function distance(a::ADFGenome, b::ADFGenome)
    # Sum of per-tree node-count differences. Coarse but symmetric.
    d = abs(count_nodes(a.main) - count_nodes(b.main))
    for i in 1:min(a.n_adfs, b.n_adfs)
        d += abs(count_nodes(a.adfs[i]) - count_nodes(b.adfs[i]))
    end
    # Penalize ADF-count mismatch.
    d += abs(a.n_adfs - b.n_adfs)
    return Float64(d)
end

function serialize(g::ADFGenome)
    io = IOBuffer()
    println(io, "ADFGenome n_features=$(g.n_features) n_adfs=$(g.n_adfs) arity=$(g.arity)")
    println(io, "MAIN: ", string_tree(g.main, g.operators))
    for (i, adf) in enumerate(g.adfs)
        println(io, "ADF$(i-1): ", string_tree(adf, g.operators))
    end
    return String(take!(io))
end

# Helper: count of binary ops in an OperatorEnum.
num_binary_ops(ops::OperatorEnum) = length(_get_binary_ops(ops))

# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

"""
    mutate(::SubtreeMutation, g::ADFGenome, rng) -> ADFGenome

Pick uniformly among (main, adf_1, ..., adf_N) and apply subtree mutation
to that tree. ADF bodies use the base operator set and may reference ARG
slots; main tree uses augmented operators and references only real features.
"""
function mutate(op::SubtreeMutation, g::ADFGenome{T}, rng::AbstractRNG) where T
    target = rand(rng, 0:g.n_adfs)  # 0 == main; 1..n_adfs == adfs[i]
    child = if target == 0
        new_main = _adf_subtree_mutate(g.main, rng, g.operators, g.n_features, T)
        ADFGenome{T}(new_main, [copy_node(a) for a in g.adfs],
                     g.arity, g.operators, g.n_features, g.n_adfs)
    else
        # ADF body — use base operators only, extended feature range.
        base_ops = _strip_adf_placeholders(g.operators, g.n_adfs)
        new_adfs = [copy_node(a) for a in g.adfs]
        new_adfs[target] = _adf_subtree_mutate(g.adfs[target], rng, base_ops,
                                               g.n_features + g.arity, T)
        ADFGenome{T}(copy_node(g.main), new_adfs,
                     g.arity, g.operators, g.n_features, g.n_adfs)
    end
    return _respect_caps(op, g, child)
end

# Subtree mutation that respects the n_features (features above this index
# are not generated, so it's safe for both main trees and ADF bodies when
# the n_features parameter is set appropriately).
function _adf_subtree_mutate(tree::Node{T}, rng::AbstractRNG,
                              ops::OperatorEnum, n_features::Int,
                              ::Type{T}) where T
    new_tree = copy_node(tree)
    nodes = collect(new_tree)
    isempty(nodes) && return new_tree
    target_idx = rand(rng, 1:length(nodes))
    replacement = _random_tree(rng, ops, n_features, T, 2, :grow)
    if target_idx == 1
        return replacement
    end
    _replace_nth_node!(new_tree, target_idx, replacement)
    return new_tree
end

# Strip the trailing N ADF placeholder operators from an augmented enum
# to recover the user's base operator set.
function _strip_adf_placeholders(aug::OperatorEnum, n_adfs::Int)
    binary = collect(_get_binary_ops(aug))
    n_base = length(binary) - n_adfs
    base_binary = binary[1:n_base]
    base_unary = collect(_get_unary_ops(aug))
    return OperatorEnum(binary_operators=base_binary, unary_operators=base_unary)
end

"""
    base_operators(g::ADFGenome) -> OperatorEnum

Recover the user's original operator enum (without the N ADF placeholder slots).
"""
base_operators(g::ADFGenome) = _strip_adf_placeholders(g.operators, g.n_adfs)

# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

"""
    crossover(::SubtreeCrossover, g1::ADFGenome, g2::ADFGenome, rng) -> Tuple

Same-index subtree crossover: pick uniformly among (main, adf_1, ..., adf_N)
and swap subtrees within the chosen tree pair. Requires both genomes to
share `n_features` and `n_adfs`.
"""
function crossover(op::SubtreeCrossover, g1::ADFGenome{T}, g2::ADFGenome{T},
                   rng::AbstractRNG) where T
    g1.n_adfs == g2.n_adfs || throw(ArgumentError(
        "ADFGenome crossover requires matching n_adfs (got $(g1.n_adfs) and $(g2.n_adfs))"))
    g1.n_features == g2.n_features || throw(ArgumentError(
        "ADFGenome crossover requires matching n_features"))

    target = rand(rng, 0:g1.n_adfs)
    (c1, c2) = if target == 0
        c1_main, c2_main = _swap_subtrees(g1.main, g2.main, rng, T)
        (ADFGenome{T}(c1_main, [copy_node(a) for a in g1.adfs],
                      g1.arity, g1.operators, g1.n_features, g1.n_adfs),
         ADFGenome{T}(c2_main, [copy_node(a) for a in g2.adfs],
                      g2.arity, g2.operators, g2.n_features, g2.n_adfs))
    else
        new_adfs1 = [copy_node(a) for a in g1.adfs]
        new_adfs2 = [copy_node(a) for a in g2.adfs]
        adf1, adf2 = _swap_subtrees(g1.adfs[target], g2.adfs[target], rng, T)
        new_adfs1[target] = adf1
        new_adfs2[target] = adf2
        (ADFGenome{T}(copy_node(g1.main), new_adfs1,
                      g1.arity, g1.operators, g1.n_features, g1.n_adfs),
         ADFGenome{T}(copy_node(g2.main), new_adfs2,
                      g2.arity, g2.operators, g2.n_features, g2.n_adfs))
    end
    return (_respect_caps(op, g1, c1), _respect_caps(op, g2, c2))
end

function _swap_subtrees(t1::Node{T}, t2::Node{T}, rng, ::Type{T}) where T
    a = copy_node(t1)
    b = copy_node(t2)
    nodes_a = collect(a)
    nodes_b = collect(b)
    idx_a = rand(rng, 1:length(nodes_a))
    idx_b = rand(rng, 1:length(nodes_b))
    sub_a = copy_node(nodes_a[idx_a])
    sub_b = copy_node(nodes_b[idx_b])
    if idx_a == 1
        a = sub_b
    else
        _replace_nth_node!(a, idx_a, sub_b)
    end
    if idx_b == 1
        b = sub_a
    else
        _replace_nth_node!(b, idx_b, sub_a)
    end
    return (a, b)
end

# ---------------------------------------------------------------------------
# Evaluation against (X, y) — convenience for tests / benchmarks.
# ---------------------------------------------------------------------------

"""
    evaluate_adf(g::ADFGenome{T}, X::Matrix{T}, y::Vector{T}) -> Float64

Expand ADF calls and compute MSE against `y` over `X`. Returns `Inf` on
expansion or evaluation failure (typical: ARG references with no enclosing
ADF context, i.e. ARG slots leaked into the main tree's expanded form).
"""
function evaluate_adf(g::ADFGenome{T}, X::Matrix{T}, y::AbstractVector{T}) where T
    expanded = try
        expand_adfs(g)
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end
    base_ops = base_operators(g)
    try
        predictions, complete = eval_tree_array(expanded, X, base_ops)
        complete || return Inf
        n = length(y)
        sse = 0.0
        @inbounds for i in 1:n
            d = Float64(predictions[i]) - Float64(y[i])
            sse += d * d
        end
        mse = sse / n
        return isfinite(mse) ? mse : Inf
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end
end
