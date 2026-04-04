You are conducting a focused adversarial review of a research project.

**Project:** Arborist.jl — a generic, extensible genetic programming framework for Julia. This iteration focuses on three genome types: TreeGenome (DynamicExpressions.jl-backed symbolic regression), GraphGenome (NEAT neural topology evolution), and AntGenome (side-effectful program synthesis).

Your role is to find flaws, untested assumptions, and failure modes that the development process has missed. You are the adversarial reviewer in a two-model workflow: you generate critical hypotheses, and another system tests them empirically.

## What you are NOT being asked to do

- Suggest wholesale alternatives or redesign the project
- Restate known limitations (listed below) unless you have a NEW argument
- Comment on writing style, documentation organization, or naming conventions
- Provide general praise or encouragement
- Suggest performance optimizations — this review is about correctness only
- Review examples/ directory code
- Review ExprGenome, the core solve loop, or operator selection logic (covered in iteration 001)

## Known limitations (do not re-discover these):
- AntGenome not thread-safe — uses module-level `Ref` for simulator state (documented, enforced with runtime error)
- GraphGenome.deserialize not implemented — returns `nothing` with a warning
- Distributed NEAT innovation ID collisions across workers
- TreeGenome serialize/deserialize format mismatch — `serialize` outputs infix, `deserialize` parses prefix
- ExprGenome serialize round-trip partial (~80% success rate)
- `GeneticProgramming.max_depth` field is declared but never read (confirmed dead code in iteration 001)

## Prior findings from iteration 001:
- CONFIRMED: No validation that crossover_rate + mutation_rate <= 1.0 (affects all solve paths including these genome types)
- CONFIRMED: Migrant ExprGenome genomes carry source island RNG via GenState (does not apply to TreeGenome/GraphGenome/AntGenome which don't carry GenState)

## What you ARE being asked to do

Find specific, testable problems in TreeGenome, GraphGenome (NEAT), and AntGenome. Focus especially on:

1. **Does the NEAT implementation match the NEAT specification?** (Stanley & Miikkulainen, 2002) — crossover alignment by innovation number, speciation, structural mutation, fitness sharing. Are there deviations?

2. **TreeGenome tree manipulation correctness** — Does `_replace_nth_node!` correctly find and replace the nth node in pre-order traversal? Are there off-by-one errors?

3. **AntGenome mutation correctness** — The mutation uses `unravel` + identity comparison (`===`) for replacement. Are there cases where replacement fails silently?

4. **NEAT distance formula** — The code uses `(c1 + c2) * disjoint_excess / N`. Standard NEAT uses separate counts for disjoint and excess genes with separate coefficients. Is this a meaningful deviation?

5. **GraphGenome crossover** — It produces two children but both call `_neat_crossover(fitter, other, rng)` with the same argument order. Does this produce two identical children?

6. **Network evaluation** — Does the topological sort handle all valid NEAT topologies correctly?

**Categories to look for:**
1. Circular dependencies
2. Context mismatch
3. Untested boundaries
4. Claims that don't follow from evidence
5. Scaling walls
6. Silent failure modes
7. Missing mechanisms

## Source Material

### src/tree_genome.jl — TreeGenome

```julia
struct TreeGenome{T} <: AbstractGenome
    tree::Node{T}
    operators::OperatorEnum
    n_features::Int
end

struct TreeFitnessEvaluator{T} <: AbstractEvaluator
    X::Matrix{T}
    y::Vector{T}
    operators::OperatorEnum
end

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

function _random_tree(rng::AbstractRNG, operators::OperatorEnum, n_features::Int,
                      ::Type{T}, depth::Int, method::Symbol) where T
    n_unary = length(_get_unary_ops(operators))
    n_binary = length(_get_binary_ops(operators))
    n_ops = n_unary + n_binary

    if depth <= 0 || (method == :grow && n_ops > 0 && rand(rng) < 0.3 && depth < 4)
        return _random_terminal(rng, n_features, T)
    end

    if n_ops == 0
        return _random_terminal(rng, n_features, T)
    end

    op_choice = rand(rng, 1:n_ops)
    if op_choice <= n_binary
        l = _random_tree(rng, operators, n_features, T, depth - 1, method)
        r = _random_tree(rng, operators, n_features, T, depth - 1, method)
        return Node{T}(; op=UInt8(op_choice), l=l, r=r)
    else
        ui = op_choice - n_binary
        child = _random_tree(rng, operators, n_features, T, depth - 1, method)
        return Node{T}(; op=UInt8(ui), l=child)
    end
end

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

function _point_mutate(g::TreeGenome{T}, rng::AbstractRNG) where T
    new_tree = copy_node(g.tree)
    nodes = collect(new_tree)
    isempty(nodes) && return g
    target_idx = rand(rng, 1:length(nodes))
    replacement = _random_tree(rng, g.operators, g.n_features, T, 2, :grow)
    if target_idx == 1
        return TreeGenome{T}(replacement, g.operators, g.n_features)
    end
    _replace_nth_node!(new_tree, target_idx, replacement)
    return TreeGenome{T}(new_tree, g.operators, g.n_features)
end

function _replace_nth_node!(tree::Node{T}, target_idx::Int, replacement::Node{T}) where T
    counter = Ref(0)
    _replace_nth_recursive!(tree, target_idx, replacement, counter)
end

function _replace_nth_recursive!(node::Node{T}, target_idx::Int,
                                  replacement::Node{T}, counter::Ref{Int}) where T
    counter[] += 1
    if node.degree >= 1
        counter_before_l = counter[]
        if counter_before_l + 1 == target_idx
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

function _subtree_crossover(g1::TreeGenome{T}, g2::TreeGenome{T}, rng::AbstractRNG) where T
    t1 = copy_node(g1.tree)
    t2 = copy_node(g2.tree)
    nodes1 = collect(t1)
    nodes2 = collect(t2)
    idx1 = rand(rng, 1:length(nodes1))
    idx2 = rand(rng, 1:length(nodes2))
    sub1 = copy_node(nodes1[idx1])
    sub2 = copy_node(nodes2[idx2])
    if idx1 == 1
        t1 = sub2
    else
        _replace_nth_node!(t1, idx1, sub2)
    end
    if idx2 == 1
        t2 = sub1
    else
        _replace_nth_node!(t2, idx2, sub1)
    end
    return (TreeGenome{T}(t1, g1.operators, g1.n_features),
            TreeGenome{T}(t2, g2.operators, g2.n_features))
end

function distance(g1::TreeGenome{T}, g2::TreeGenome{T}) where T
    Float64(abs(count_nodes(g1.tree) - count_nodes(g2.tree)))
end

# Operator dispatches for TreeGenome — delegate to genome methods
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
```

### src/genome/graph_genome.jl — GraphGenome (NEAT)

```julia
const _innovation_counter = Ref{Int}(0)
const _innovation_lock = ReentrantLock()

function _next_innovation!()::Int
    lock(_innovation_lock) do
        _innovation_counter[] += 1
        _innovation_counter[]
    end
end

function reset_innovation_counter!()
    lock(_innovation_lock) do
        _innovation_counter[] = 0
    end
end

struct NodeGene
    id::Int
    type::Symbol          # :input, :hidden, :output, :bias
    activation::Symbol    # :sigmoid, :tanh, :relu, :identity
end

mutable struct ConnectionGene
    in_node::Int
    out_node::Int
    weight::Float64
    enabled::Bool
    innovation::Int
end

mutable struct GraphGenome <: AbstractGenome
    nodes::Dict{Int, NodeGene}
    connections::Dict{Int, ConnectionGene}
    n_inputs::Int
    n_outputs::Int
    fitness::Float64
end

function initialize(::Type{GraphGenome}, n_inputs::Int, n_outputs::Int,
                    rng::AbstractRNG)
    nodes = Dict{Int, NodeGene}()
    next_id = 1
    for i in 1:n_inputs
        nodes[next_id] = NodeGene(next_id, :input, :identity)
        next_id += 1
    end
    bias_id = next_id
    nodes[bias_id] = NodeGene(bias_id, :bias, :identity)
    next_id += 1
    output_ids = Int[]
    for i in 1:n_outputs
        nodes[next_id] = NodeGene(next_id, :output, :sigmoid)
        push!(output_ids, next_id)
        next_id += 1
    end
    connections = Dict{Int, ConnectionGene}()
    input_ids = [i for i in 1:n_inputs]
    push!(input_ids, bias_id)
    for in_id in input_ids
        for out_id in output_ids
            inn = _next_innovation!()
            connections[inn] = ConnectionGene(in_id, out_id,
                                              randn(rng) * 0.5, true, inn)
        end
    end
    return GraphGenome(nodes, connections, n_inputs, n_outputs, Inf)
end

function mutate(g::GraphGenome, rng::AbstractRNG)
    new_g = _copy_graph(g)
    r = rand(rng)
    if r < 0.8
        _mutate_weights!(new_g, rng)
    elseif r < 0.9
        _mutate_weight_replace!(new_g, rng)
    elseif r < 0.95
        _mutate_add_connection!(new_g, rng)
    elseif r < 0.98
        _mutate_add_node!(new_g, rng)
    else
        _mutate_toggle_connection!(new_g, rng)
    end
    return new_g
end

function crossover(g1::GraphGenome, g2::GraphGenome, rng::AbstractRNG)
    if g1.fitness <= g2.fitness
        child1 = _neat_crossover(g1, g2, rng)
        child2 = _neat_crossover(g1, g2, rng)
    else
        child1 = _neat_crossover(g2, g1, rng)
        child2 = _neat_crossover(g2, g1, rng)
    end
    return (child1, child2)
end

function _neat_crossover(fitter::GraphGenome, other::GraphGenome,
                          rng::AbstractRNG)
    child_nodes = Dict{Int, NodeGene}()
    child_connections = Dict{Int, ConnectionGene}()
    all_innovations = union(keys(fitter.connections), keys(other.connections))
    for inn in all_innovations
        has_f = haskey(fitter.connections, inn)
        has_o = haskey(other.connections, inn)
        if has_f && has_o
            c = rand(rng, Bool) ? fitter.connections[inn] : other.connections[inn]
            child_connections[inn] = ConnectionGene(c.in_node, c.out_node,
                                                     c.weight, c.enabled, c.innovation)
        elseif has_f
            c = fitter.connections[inn]
            child_connections[inn] = ConnectionGene(c.in_node, c.out_node,
                                                     c.weight, c.enabled, c.innovation)
        end
    end
    for c in values(child_connections)
        for nid in (c.in_node, c.out_node)
            if !haskey(child_nodes, nid)
                if haskey(fitter.nodes, nid)
                    child_nodes[nid] = fitter.nodes[nid]
                elseif haskey(other.nodes, nid)
                    child_nodes[nid] = other.nodes[nid]
                end
            end
        end
    end
    for n in values(fitter.nodes)
        if n.type in (:input, :output, :bias)
            child_nodes[n.id] = n
        end
    end
    return GraphGenome(child_nodes, child_connections,
                       fitter.n_inputs, fitter.n_outputs, Inf)
end

function _neat_distance(g1::GraphGenome, g2::GraphGenome;
                        c1::Float64=1.0, c2::Float64=1.0, c3::Float64=0.4)
    inns1 = Set(keys(g1.connections))
    inns2 = Set(keys(g2.connections))
    if isempty(inns1) && isempty(inns2)
        return 0.0
    end
    matching = intersect(inns1, inns2)
    disjoint_excess = length(symdiff(inns1, inns2))
    W = 0.0
    if !isempty(matching)
        for inn in matching
            W += abs(g1.connections[inn].weight - g2.connections[inn].weight)
        end
        W /= length(matching)
    end
    N = max(length(inns1), length(inns2))
    N = N < 20 ? 1.0 : Float64(N)
    return (c1 + c2) * disjoint_excess / N + c3 * W
end

function _mutate_add_connection!(g::GraphGenome, rng::AbstractRNG)
    node_ids = collect(keys(g.nodes))
    length(node_ids) < 2 && return
    for _ in 1:20
        from_id = rand(rng, node_ids)
        to_id = rand(rng, node_ids)
        from_node = g.nodes[from_id]
        to_node = g.nodes[to_id]
        to_node.type in (:input, :bias) && continue
        from_node.type == :output && continue
        from_id == to_id && continue
        exists = any(c -> c.in_node == from_id && c.out_node == to_id,
                     values(g.connections))
        exists && continue
        inn = _next_innovation!()
        g.connections[inn] = ConnectionGene(from_id, to_id,
                                            randn(rng) * 0.5, true, inn)
        return
    end
end

function _mutate_add_node!(g::GraphGenome, rng::AbstractRNG)
    enabled_conns = [c for c in values(g.connections) if c.enabled]
    isempty(enabled_conns) && return
    old_conn = rand(rng, enabled_conns)
    old_conn.enabled = false
    new_id = maximum(keys(g.nodes)) + 1
    activation = rand(rng, [:sigmoid, :tanh, :relu])
    g.nodes[new_id] = NodeGene(new_id, :hidden, activation)
    inn1 = _next_innovation!()
    g.connections[inn1] = ConnectionGene(old_conn.in_node, new_id,
                                         1.0, true, inn1)
    inn2 = _next_innovation!()
    g.connections[inn2] = ConnectionGene(new_id, old_conn.out_node,
                                         old_conn.weight, true, inn2)
end

function _mutate_weights!(g::GraphGenome, rng::AbstractRNG)
    for c in values(g.connections)
        if c.enabled && rand(rng) < 0.9
            c.weight += randn(rng) * 0.3
        end
    end
end

function _topological_sort(g::GraphGenome)
    in_degree = Dict{Int, Int}()
    for n in keys(g.nodes)
        in_degree[n] = 0
    end
    adj = Dict{Int, Vector{Int}}()
    for n in keys(g.nodes)
        adj[n] = Int[]
    end
    for c in values(g.connections)
        c.enabled || continue
        haskey(g.nodes, c.in_node) || continue
        haskey(g.nodes, c.out_node) || continue
        push!(adj[c.in_node], c.out_node)
        in_degree[c.out_node] = get(in_degree, c.out_node, 0) + 1
    end
    queue = [n for (n, d) in in_degree if d == 0]
    sort!(queue)
    result = Int[]
    while !isempty(queue)
        n = popfirst!(queue)
        push!(result, n)
        for m in sort!(adj[n])
            in_degree[m] -= 1
            if in_degree[m] == 0
                push!(queue, m)
            end
        end
    end
    length(result) == length(g.nodes) ? result : nothing
end

function evaluate_genome(g::GraphGenome, e::GraphEvaluator)
    try
        n_samples = size(e.input_data, 2)
        total_se = 0.0
        eval_order = _topological_sort(g)
        eval_order === nothing && return Inf
        input_ids = sort!([n.id for n in values(g.nodes) if n.type == :input])
        output_ids = sort!([n.id for n in values(g.nodes) if n.type == :output])
        bias_ids = [n.id for n in values(g.nodes) if n.type == :bias]
        for s in 1:n_samples
            node_vals = Dict{Int, Float64}()
            for (idx, nid) in enumerate(input_ids)
                node_vals[nid] = e.input_data[idx, s]
            end
            for nid in bias_ids
                node_vals[nid] = 1.0
            end
            for nid in eval_order
                haskey(g.nodes, nid) || continue
                node = g.nodes[nid]
                node.type in (:input, :bias) && continue
                total = 0.0
                for c in values(g.connections)
                    if c.enabled && c.out_node == nid
                        total += c.weight * get(node_vals, c.in_node, 0.0)
                    end
                end
                act_fn = get(e.activation_fns, node.activation, identity)
                node_vals[nid] = act_fn(total)
            end
            for (idx, nid) in enumerate(output_ids)
                predicted = get(node_vals, nid, 0.0)
                expected = e.output_data[idx, s]
                se = (predicted - expected)^2
                isfinite(se) || return Inf
                total_se += se
            end
        end
        return total_se / (n_samples * length(output_ids))
    catch e
        e isa InterruptException && rethrow()
        return Inf
    end
end
```

### src/genome/ant_genome.jl — AntGenome

```julia
const _ant_sim_ref = Ref{Any}(nothing)

mutable struct AntSimulator
    grid::Matrix{Bool}
    row::Int
    col::Int
    direction::Int   # 0=east, 1=south, 2=west, 3=north
    moves::Int
    food_eaten::Int
    max_moves::Int
end

const _ANT_DR = [0, 1, 0, -1]  # east, south, west, north
const _ANT_DC = [1, 0, -1, 0]

function _ant_ahead(ant::AntSimulator)
    gs = size(ant.grid, 1)
    r = mod1(ant.row + _ANT_DR[ant.direction + 1], gs)
    c = mod1(ant.col + _ANT_DC[ant.direction + 1], gs)
    return (r, c)
end

function gp_ant_move(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    ant.moves >= ant.max_moves && return false
    r, c = _ant_ahead(ant)
    ant.row = r; ant.col = c; ant.moves += 1
    if ant.grid[r, c]
        ant.food_eaten += 1; ant.grid[r, c] = false
    end
    return true
end

function gp_ant_left(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    ant.moves >= ant.max_moves && return false
    ant.direction = mod(ant.direction + 3, 4)
    ant.moves += 1
    return true
end

function gp_ant_right(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    ant.moves >= ant.max_moves && return false
    ant.direction = mod(ant.direction + 1, 4)
    ant.moves += 1
    return true
end

struct AntGenome <: AbstractGenome
    program::Expr
    primitives::Vector{Symbol}
    conditions::Vector{Symbol}
    max_depth::Int
end

function mutate(g::AntGenome, rng::AbstractRNG)
    new_prog = deepcopy(g.program)
    nodes = unravel(new_prog)
    if isempty(nodes)
        return AntGenome(_random_ant_program(g.primitives, g.conditions,
                                             g.max_depth, rng),
                         g.primitives, g.conditions, g.max_depth)
    end
    target_idx = rand(rng, 1:length(nodes))
    replacement = _random_ant_program(g.primitives, g.conditions,
                                      max(1, g.max_depth - 2), rng)
    if target_idx == 1
        return AntGenome(replacement, g.primitives, g.conditions, g.max_depth)
    end
    for i in 1:length(new_prog.args)
        if new_prog.args[i] isa Expr && new_prog.args[i] === nodes[target_idx]
            new_prog.args[i] = replacement
            return AntGenome(new_prog, g.primitives, g.conditions, g.max_depth)
        end
    end
    for node in nodes
        if node !== nodes[target_idx] && node isa Expr
            for i in 1:length(node.args)
                if node.args[i] isa Expr && node.args[i] === nodes[target_idx]
                    node.args[i] = replacement
                    return AntGenome(new_prog, g.primitives, g.conditions, g.max_depth)
                end
            end
        end
    end
    return AntGenome(new_prog, g.primitives, g.conditions, g.max_depth)
end

function crossover(g1::AntGenome, g2::AntGenome, rng::AbstractRNG)
    p1 = deepcopy(g1.program)
    p2 = deepcopy(g2.program)
    nodes1 = unravel(p1)
    nodes2 = unravel(p2)
    if isempty(nodes1) || isempty(nodes2)
        return (g1, g2)
    end
    idx1 = rand(rng, 1:length(nodes1))
    idx2 = rand(rng, 1:length(nodes2))
    sub1 = deepcopy(nodes1[idx1])
    sub2 = deepcopy(nodes2[idx2])
    c1_prog = idx1 == 1 ? sub2 : begin
        replace_subtree!(p1, nodes1[idx1], sub2); p1
    end
    c2_prog = idx2 == 1 ? sub1 : begin
        replace_subtree!(p2, nodes2[idx2], sub1); p2
    end
    return (AntGenome(c1_prog, g1.primitives, g1.conditions, g1.max_depth),
            AntGenome(c2_prog, g1.primitives, g1.conditions, g1.max_depth))
end

function evaluate(e::AntEvaluator, f::Function)
    ant = AntSimulator(e.food_positions, e.max_moves)
    _ant_sim_ref[] = ant
    n_food = length(e.food_positions)
    max_calls = e.max_moves * 2
    calls = 0
    while ant.moves < ant.max_moves && ant.food_eaten < n_food
        moves_before = ant.moves
        try
            Base.invokelatest(f)
        catch e
            e isa InterruptException && rethrow()
            break
        end
        calls += 1
        if ant.moves == moves_before || calls >= max_calls
            break
        end
    end
    return Float64(n_food - ant.food_eaten)
end
```

### Relevant paper observations (arborist_paper_observations.md excerpts)

**Section 1.3 — Dual genome types:**
> TreeGenome: 8.4x faster evaluation than ExprGenome on the Koza symbolic regression suite.
> ExprGenome: Supports arbitrary Julia control flow (loops, conditionals, mutable state, side effects).

**Section 3.3 — XOR NEAT benchmark:**
> The fitness sharing sign error (division instead of multiplication) was found and fixed.
> After fixing, XOR converged 4/5 seeds to fitness < 0.01 within 150 generations.
> Networks grew organically from 4 to 7-11 nodes via structural mutation.

**Section 4.4 — DynamicExpressions.jl:**
> TreeGenome wraps DynamicExpressions.jl's `Node{T}` type, which is the backbone of SymbolicRegression.jl and PySR.

## Output Format

Produce numbered findings. For each:

```
### Finding N: [short title]

**Tag:** [STRUCTURAL | PARAMETER | UNTESTED | CIRCULAR | SCALING | SILENT_FAILURE | MISSING_MECHANISM]
**Confidence:** [HIGH | MEDIUM | LOW]
**Severity:** [CRITICAL — breaks a core claim | SIGNIFICANT — wrong but fixable | MINOR — edge case or refinement]

**Claim being challenged:**
[The specific claim or assumption you think is wrong or undertested]

**Why it might be wrong:**
[Your argument. Be specific. Reference the source material.]

**Suggested test:**
[A concrete investigation that would determine whether you're right.
Include what the expected result would be if the project is correct
vs. if your concern is valid.]
```

Aim for 5-15 findings. Quality over quantity.
