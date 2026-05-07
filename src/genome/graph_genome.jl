# graph_genome.jl — NEAT-style neural topology genome.

# =============================================================================
# Innovation tracking (global, thread-safe)
# =============================================================================

const _innovation_counter = Ref{Int}(0)
const _innovation_lock = ReentrantLock()

"""
    _next_innovation!() -> Int

Get the next global innovation number (thread-safe).
"""
function _next_innovation!()::Int
    lock(_innovation_lock) do
        _innovation_counter[] += 1
        _innovation_counter[]
    end
end

"""
    reset_innovation_counter!()

Reset the global innovation counter to 0. Must be called at the start
of each `solve()` call for GraphGenome problems.
"""
function reset_innovation_counter!()
    init_innovation_range!(0)
end

"""
    init_innovation_range!(offset::Int)

Set the module-local innovation counter to `offset`. Used by the
distributed island model to give each worker a disjoint range of
innovation IDs so that NEAT crossover on migrants does not align
structurally unrelated genes under the same innovation number.

Callers in distributed mode typically use offsets like
`(island_id - 1) * 10^9` — disjoint as long as no single worker allocates
more than `10^9` structural mutations in a run. The sequential island
model does not need this: all islands share the same process-global
counter, which already ensures uniqueness.
"""
function init_innovation_range!(offset::Int)
    lock(_innovation_lock) do
        _innovation_counter[] = offset
    end
end

# =============================================================================
# Node and connection genes
# =============================================================================

"""
    GraphGenomeContext

Per-island state carrier for `GraphGenome` under `IslandModel`. Parallels
`GenState` (ExprGenome) and `TreeGenomeContext` (TreeGenome): all three
carry `.rng` so that island-loop sites reading `state.rng` work uniformly,
and all three are the second element of the tuple returned from
`_initialize_population`.

The extra `n_inputs` / `n_outputs` fields are kept available for future use
(e.g. cross-island initialization) but aren't currently consulted — migrant
GraphGenomes carry their own `n_inputs` / `n_outputs`.
"""
struct GraphGenomeContext
    rng::AbstractRNG
    n_inputs::Int
    n_outputs::Int
end

"""
    NodeGene

A single node in a neural network topology genome.
"""
struct NodeGene
    id::Int
    type::Symbol          # :input, :hidden, :output, :bias
    activation::Symbol    # :sigmoid, :tanh, :relu, :identity
end

"""
    ConnectionGene

A directed connection between two nodes.
"""
mutable struct ConnectionGene
    in_node::Int
    out_node::Int
    weight::Float64
    enabled::Bool
    innovation::Int
end

# =============================================================================
# GraphGenome struct
# =============================================================================

"""
    GraphGenome <: AbstractGenome

A genome representing a neural network topology, following the NEAT
encoding (Stanley & Miikkulainen, 2002). Supports structural mutation
(add node, add connection) and weight mutation, plus crossover aligned
by innovation number.

# Fields
- `nodes::Dict{Int, NodeGene}`: node genes keyed by node ID
- `connections::Dict{Int, ConnectionGene}`: connection genes keyed by innovation number
- `n_inputs::Int`: number of input nodes (not counting bias)
- `n_outputs::Int`: number of output nodes
- `fitness::Float64`: cached fitness value

# Known limitations

- **Distributed NEAT innovation matching is disjoint-range, not
  content-aware.** Under `IslandModel(distributed=true)`, each worker
  gets a unique innovation ID range via
  `init_innovation_range!((island_id - 1) * INNOVATION_STRIDE)` so
  IDs don't collide. The cost: structurally identical mutations on
  different workers receive different IDs and are treated as disjoint
  by NEAT crossover rather than aligned. Per-generation cross-worker
  innovation dedup is not implemented.
"""
mutable struct GraphGenome <: AbstractGenome
    nodes::Dict{Int, NodeGene}
    connections::Dict{Int, ConnectionGene}
    n_inputs::Int
    n_outputs::Int
    fitness::Float64
end

# --- Display ---------------------------------------------------------------

function Base.show(io::IO, g::GraphGenome)
    n_enabled = count(c -> c.enabled, values(g.connections))
    print(io, "GraphGenome(nodes=", length(g.nodes),
              ", conns=", n_enabled, "/", length(g.connections), ")")
end

function Base.show(io::IO, ::MIME"text/plain", g::GraphGenome)
    types = Dict{Symbol, Int}()
    activations = Dict{Symbol, Int}()
    for n in values(g.nodes)
        types[n.type] = get(types, n.type, 0) + 1
        activations[n.activation] = get(activations, n.activation, 0) + 1
    end
    n_enabled = count(c -> c.enabled, values(g.connections))
    n_disabled = length(g.connections) - n_enabled
    println(io, "GraphGenome")
    println(io, "  inputs:      ", get(types, :input, 0))
    println(io, "  outputs:     ", get(types, :output, 0))
    println(io, "  bias:        ", get(types, :bias, 0))
    println(io, "  hidden:      ", get(types, :hidden, 0))
    println(io, "  connections: ", length(g.connections),
                " (", n_enabled, " enabled, ", n_disabled, " disabled)")
    if !isempty(activations)
        keys_sorted = sort!(collect(keys(activations)))
        parts = String[]
        for k in keys_sorted
            push!(parts, string(k, "(", activations[k], ")"))
        end
        println(io, "  activations: ", join(parts, ", "))
    end
    print(io,   "  fitness:     ", _fmt_fitness(g.fitness))
end

# =============================================================================
# Activation functions
# =============================================================================

"""
    ACTIVATION_FNS

Dictionary mapping activation `Symbol` names to their unary `Function`
implementations, used by `GraphEvaluator` when propagating values through a
`GraphGenome`. The built-in set is:

- `:sigmoid`  — NEAT-style steepened logistic `1 / (1 + exp(-4.9·x))`.
- `:tanh`     — hyperbolic tangent.
- `:relu`     — rectified linear, `max(0, x)`.
- `:identity` — `x` (pass-through).
- `:gauss`    — Gaussian bump `exp(-x²)`. Common in CPPN / HyperNEAT work.
- `:sin`      — plain `sin(x)`. Substrate or network is expected to supply any
                frequency scaling.
- `:abs`      — absolute value `|x|`.
- `:step`     — Heaviside step, `1.0` for `x > 0`, else `0.0`.

New activations can be added by assigning into this dict before solving; each
`NodeGene` stores the activation as a `Symbol` and looks the function up here
at evaluation time.

Note: the NEAT mutation operators (`AddNodeMutation`, `NEATDefaultMutation`)
only draw from `:sigmoid`, `:tanh`, `:relu` by default when adding a new
hidden node. To make CPPN activations available to those operators, pass
`hidden_activations=[:sigmoid, :tanh, :gauss, :sin, :abs]` (or similar) at
construction.
"""
const ACTIVATION_FNS = Dict{Symbol, Function}(
    :sigmoid  => x -> 1.0 / (1.0 + exp(-4.9 * x)),
    :tanh     => x -> tanh(x),
    :relu     => x -> max(0.0, x),
    :identity => x -> x,
    :gauss    => x -> exp(-x * x),
    :sin      => x -> sin(x),
    :abs      => x -> abs(x),
    :step     => x -> x > 0.0 ? 1.0 : 0.0,
)

# =============================================================================
# Initialization
# =============================================================================

"""
    initialize(::Type{GraphGenome}, n_inputs, n_outputs, rng) -> GraphGenome

Create a minimal fully-connected network: all inputs connected to all
outputs with random weights, no hidden nodes. Includes a bias node.
"""
function initialize(::Type{GraphGenome}, n_inputs::Int, n_outputs::Int,
                    rng::AbstractRNG)
    nodes = Dict{Int, NodeGene}()
    next_id = 1

    # Input nodes
    for i in 1:n_inputs
        nodes[next_id] = NodeGene(next_id, :input, :identity)
        next_id += 1
    end

    # Bias node
    bias_id = next_id
    nodes[bias_id] = NodeGene(bias_id, :bias, :identity)
    next_id += 1

    # Output nodes
    output_ids = Int[]
    for i in 1:n_outputs
        nodes[next_id] = NodeGene(next_id, :output, :sigmoid)
        push!(output_ids, next_id)
        next_id += 1
    end

    # Fully connected: each input + bias -> each output
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

# =============================================================================
# AbstractGenome interface
# =============================================================================
#
# Mutation and crossover on GraphGenome go through the framework's operator
# dispatch (`AbstractMutationOperator` / `AbstractCrossoverOperator`). See
# `src/operators/neat_mutation.jl` for `NEATDefaultMutation` (canonical
# Stanley & Miikkulainen branching) and the individual-branch operators, and
# `src/operators/crossover.jl` for `NEATCrossover`. Use `neat_defaults()` to
# get a ready-made (mutation_ops, crossover_ops) tuple.

function distance(g1::GraphGenome, g2::GraphGenome)
    _neat_distance(g1, g2)
end

function complexity(g::GraphGenome)
    Float64(count(c -> c.enabled, values(g.connections)))
end

function serialize(g::GraphGenome)
    io = IOBuffer()
    for (id, n) in sort!(collect(g.nodes), by=first)
        println(io, "N $(n.id) $(n.type) $(n.activation)")
    end
    for (inn, c) in sort!(collect(g.connections), by=first)
        println(io, "C $(c.in_node)->$(c.out_node) w=$(c.weight) en=$(c.enabled) i=$(c.innovation)")
    end
    return String(take!(io))
end

"""
    deserialize(::Type{GraphGenome}, s::AbstractString, n_inputs, n_outputs;
                reassign_innovations=false) -> Union{GraphGenome, Nothing}

Parse the text emitted by `serialize(::GraphGenome)` back into a
`GraphGenome`. The format is line-oriented:

- `N <id> <type> <activation>` — node line
- `C <in>-><out> w=<weight> en=<true|false> i=<innovation>` — connection line

Lines not starting with `N` or `C` are skipped (tolerates LLM commentary,
code fences, etc). Returns `nothing` when a malformed line is encountered,
when a connection references an undefined node, or when `n_inputs` /
`n_outputs` disagree with the decoded node set.

Preserves node IDs and innovation numbers verbatim — required for
content-aware distributed migration and NEAT crossover alignment. Pass
`reassign_innovations=true` to issue a fresh innovation ID to every
connection via `_next_innovation!()`; the LLM mutation path uses this to
prevent LLM-generated IDs from colliding with the parent pool's history.
"""
function deserialize(::Type{GraphGenome}, s::AbstractString,
                     n_inputs::Int, n_outputs::Int;
                     reassign_innovations::Bool = false)
    nodes = Dict{Int, NodeGene}()
    connections = Dict{Int, ConnectionGene}()

    for raw_line in split(s, '\n')
        line = strip(raw_line)
        isempty(line) && continue
        c0 = first(line)
        if c0 == 'N'
            node = _parse_node_line(line)
            node === nothing && return nothing
            haskey(nodes, node.id) && return nothing
            nodes[node.id] = node
        elseif c0 == 'C'
            conn = _parse_conn_line(line)
            conn === nothing && return nothing
            haskey(connections, conn.innovation) && return nothing
            connections[conn.innovation] = conn
        else
            # Ignore everything else — markdown fences, LLM preamble, blank runs, etc.
            continue
        end
    end

    # Validate that every connection points at defined nodes.
    for c in values(connections)
        haskey(nodes, c.in_node) || return nothing
        haskey(nodes, c.out_node) || return nothing
    end

    # Validate declared input/output counts match the decoded node set.
    n_in_decoded  = count(n -> n.type == :input,  values(nodes))
    n_out_decoded = count(n -> n.type == :output, values(nodes))
    n_in_decoded  == n_inputs  || return nothing
    n_out_decoded == n_outputs || return nothing

    if reassign_innovations && !isempty(connections)
        new_conns = Dict{Int, ConnectionGene}()
        for c in values(connections)
            fresh = _next_innovation!()
            new_conns[fresh] = ConnectionGene(c.in_node, c.out_node,
                                              c.weight, c.enabled, fresh)
        end
        connections = new_conns
    end

    return GraphGenome(nodes, connections, n_inputs, n_outputs, Inf)
end

# Line parsers. Return `nothing` on any mismatch.

function _parse_node_line(line::AbstractString)
    # "N <id> <type> <activation>"
    parts = split(line)
    length(parts) == 4 || return nothing
    parts[1] == "N" || return nothing
    id = tryparse(Int, parts[2])
    id === nothing && return nothing
    nt  = Symbol(parts[3])
    act = Symbol(parts[4])
    nt in (:input, :hidden, :output, :bias) || return nothing
    return NodeGene(id, nt, act)
end

function _parse_conn_line(line::AbstractString)
    # "C <in>-><out> w=<weight> en=<true|false> i=<innovation>"
    parts = split(line)
    length(parts) == 5 || return nothing
    parts[1] == "C" || return nothing

    m_edge = match(r"^(\-?\d+)->(\-?\d+)$", parts[2])
    m_edge === nothing && return nothing
    in_node  = tryparse(Int, m_edge.captures[1])
    out_node = tryparse(Int, m_edge.captures[2])
    (in_node === nothing || out_node === nothing) && return nothing

    startswith(parts[3], "w=") || return nothing
    weight = tryparse(Float64, SubString(parts[3], 3))
    weight === nothing && return nothing

    startswith(parts[4], "en=") || return nothing
    enabled = if parts[4] == "en=true"
        true
    elseif parts[4] == "en=false"
        false
    else
        return nothing
    end

    startswith(parts[5], "i=") || return nothing
    innovation = tryparse(Int, SubString(parts[5], 3))
    innovation === nothing && return nothing

    return ConnectionGene(in_node, out_node, weight, enabled, innovation)
end

# =============================================================================
# Mutation operators
# =============================================================================

function _mutate_weights!(g::GraphGenome, rng::AbstractRNG;
                          perturb_prob::Float64=0.9, perturb_sigma::Float64=0.3)
    # Perturb each enabled weight independently with `perturb_prob` probability.
    # Sort by innovation number for deterministic RNG consumption order.
    for c in sort!(collect(values(g.connections)), by=c -> c.innovation)
        if c.enabled && rand(rng) < perturb_prob
            c.weight += randn(rng) * perturb_sigma
        end
    end
end

function _mutate_weight_replace!(g::GraphGenome, rng::AbstractRNG;
                                 replace_sigma::Float64=2.0)
    conns = sort!(collect(values(g.connections)), by=c -> c.innovation)
    isempty(conns) && return
    c = rand(rng, conns)
    c.weight = randn(rng) * replace_sigma
end

function _mutate_add_connection!(g::GraphGenome, rng::AbstractRNG;
                                 max_attempts::Int=20)
    node_ids = sort!(collect(keys(g.nodes)))
    length(node_ids) < 2 && return

    for _ in 1:max_attempts
        from_id = rand(rng, node_ids)
        to_id = rand(rng, node_ids)
        from_node = g.nodes[from_id]
        to_node = g.nodes[to_id]

        # Don't connect to input/bias nodes or from output nodes
        to_node.type in (:input, :bias) && continue
        from_node.type == :output && continue
        from_id == to_id && continue

        # Check if connection already exists
        exists = any(c -> c.in_node == from_id && c.out_node == to_id,
                     values(g.connections))
        exists && continue

        inn = _next_innovation!()
        g.connections[inn] = ConnectionGene(from_id, to_id,
                                            randn(rng) * 0.5, true, inn)
        return
    end
end

function _mutate_add_node!(g::GraphGenome, rng::AbstractRNG;
                           hidden_activations::Vector{Symbol}=Symbol[:sigmoid, :tanh, :relu])
    enabled_conns = sort!([c for c in values(g.connections) if c.enabled], by=c -> c.innovation)
    isempty(enabled_conns) && return

    old_conn = rand(rng, enabled_conns)
    old_conn.enabled = false

    # New hidden node
    new_id = maximum(keys(g.nodes)) + 1
    activation = rand(rng, hidden_activations)
    g.nodes[new_id] = NodeGene(new_id, :hidden, activation)

    # Two new connections: in_node -> new_node -> out_node
    inn1 = _next_innovation!()
    g.connections[inn1] = ConnectionGene(old_conn.in_node, new_id,
                                         1.0, true, inn1)
    inn2 = _next_innovation!()
    g.connections[inn2] = ConnectionGene(new_id, old_conn.out_node,
                                         old_conn.weight, true, inn2)
end

function _mutate_toggle_connection!(g::GraphGenome, rng::AbstractRNG)
    conns = sort!(collect(values(g.connections)), by=c -> c.innovation)
    isempty(conns) && return
    c = rand(rng, conns)
    c.enabled = !c.enabled
end

# =============================================================================
# NEAT crossover
# =============================================================================

function _neat_crossover(fitter::GraphGenome, other::GraphGenome,
                          rng::AbstractRNG)
    child_nodes = Dict{Int, NodeGene}()
    child_connections = Dict{Int, ConnectionGene}()

    # All innovation numbers from both parents, sorted for deterministic RNG order.
    all_innovations = sort!(collect(union(keys(fitter.connections), keys(other.connections))))

    for inn in all_innovations
        has_f = haskey(fitter.connections, inn)
        has_o = haskey(other.connections, inn)

        if has_f && has_o
            # Matching gene: inherit randomly
            c = rand(rng, Bool) ? fitter.connections[inn] : other.connections[inn]
            child_connections[inn] = ConnectionGene(c.in_node, c.out_node,
                                                     c.weight, c.enabled, c.innovation)
        elseif has_f
            # Disjoint/excess from fitter parent
            c = fitter.connections[inn]
            child_connections[inn] = ConnectionGene(c.in_node, c.out_node,
                                                     c.weight, c.enabled, c.innovation)
        end
        # Disjoint/excess from other parent: skip (inherit from fitter)
    end

    # Collect all referenced nodes
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

    # Ensure all input/output/bias nodes are present
    for n in values(fitter.nodes)
        if n.type in (:input, :output, :bias)
            child_nodes[n.id] = n
        end
    end

    return GraphGenome(child_nodes, child_connections,
                       fitter.n_inputs, fitter.n_outputs, Inf)
end

# =============================================================================
# NEAT distance
# =============================================================================

"""
    _neat_distance(g1, g2; c1=1.0, c2=1.0, c3=0.4) -> Float64

NEAT compatibility distance: `δ = c1 * E / N + c2 * D / N + c3 * W`
where E = excess genes (beyond the other genome's max innovation),
D = disjoint genes (within range but not matching), W = mean weight
difference of matching genes, N = max genome size (1.0 if < 20).

Matches the formula from Stanley & Miikkulainen (2002).
"""
function _neat_distance(g1::GraphGenome, g2::GraphGenome;
                        c1::Float64=1.0, c2::Float64=1.0, c3::Float64=0.4)
    inns1 = Set(keys(g1.connections))
    inns2 = Set(keys(g2.connections))

    if isempty(inns1) && isempty(inns2)
        return 0.0
    end

    matching = intersect(inns1, inns2)

    # Separate disjoint (within range) from excess (beyond range) genes.
    max1 = isempty(inns1) ? 0 : maximum(inns1)
    max2 = isempty(inns2) ? 0 : maximum(inns2)
    non_matching = symdiff(inns1, inns2)
    n_excess = 0
    n_disjoint = 0
    for inn in non_matching
        if inn in inns1
            # Gene in g1 but not g2: excess if beyond g2's max
            inn > max2 ? (n_excess += 1) : (n_disjoint += 1)
        else
            # Gene in g2 but not g1: excess if beyond g1's max
            inn > max1 ? (n_excess += 1) : (n_disjoint += 1)
        end
    end

    # Mean weight difference of matching genes
    W = 0.0
    if !isempty(matching)
        for inn in matching
            W += abs(g1.connections[inn].weight - g2.connections[inn].weight)
        end
        W /= length(matching)
    end

    N = max(length(inns1), length(inns2))
    N = N < 20 ? 1.0 : Float64(N)

    return c1 * n_excess / N + c2 * n_disjoint / N + c3 * W
end

# =============================================================================
# Network evaluation
# =============================================================================

"""
    GraphEvaluator <: AbstractEvaluator

Evaluates a `GraphGenome` by building the neural network from the genome
topology, running it on input data, and computing MSE against target outputs.

# Fields
- `input_data::Matrix{Float64}`: `n_inputs × n_samples`. In recurrent mode,
  samples are treated as a time sequence and node activations persist across
  samples.
- `output_data::Matrix{Float64}`: `n_outputs × n_samples`.
- `activation_fns::Dict{Symbol, Function}`: activation function lookup.
- `allow_recurrent::Bool`: when `true`, cycles in the genome are allowed and
  evaluation uses a relaxation loop with state that persists across samples.
  Default `false` — cycles return `Inf`, state resets per sample.
- `relaxation_passes::Int`: number of activation sweeps per sample when
  `allow_recurrent=true`. Default `1`. Higher values let information
  propagate further through the network within a single sample.
"""
struct GraphEvaluator <: AbstractEvaluator
    input_data::Matrix{Float64}    # n_inputs × n_samples
    output_data::Matrix{Float64}   # n_outputs × n_samples
    activation_fns::Dict{Symbol, Function}
    allow_recurrent::Bool
    relaxation_passes::Int
end

"""
    GraphEvaluator(input_data, output_data;
                   activation_fns=ACTIVATION_FNS,
                   allow_recurrent=false,
                   relaxation_passes=1)

Construct a `GraphEvaluator`. Defaults match the original feedforward
behavior: cycles return `Inf`, per-sample state reset, single forward pass.
Pass `allow_recurrent=true` for sequence/memory tasks where node
activations should persist across samples (and cycles are legal).
"""
function GraphEvaluator(input_data::Matrix{Float64}, output_data::Matrix{Float64};
                        activation_fns::Dict{Symbol, Function}=ACTIVATION_FNS,
                        allow_recurrent::Bool=false,
                        relaxation_passes::Int=1)
    relaxation_passes >= 1 || throw(ArgumentError(
        "relaxation_passes must be >= 1 (got $relaxation_passes)"))
    GraphEvaluator(input_data, output_data, activation_fns,
                   allow_recurrent, relaxation_passes)
end

input_signature(e::GraphEvaluator) = Dict(Symbol("x$i") => Float64 for i in 1:size(e.input_data, 1))
output_signature(e::GraphEvaluator) = Dict(Symbol("y$i") => Float64 for i in 1:size(e.output_data, 1))

"""
    evaluate_genome(g::GraphGenome, e::GraphEvaluator) -> Float64

Evaluate a GraphGenome by propagating inputs through the network. Returns
mean squared error against target outputs.

- Feedforward mode (`e.allow_recurrent=false`, default): topologically
  sorts the network; returns `Inf` on cycle. Each sample is independent —
  node activations reset between samples.
- Recurrent mode (`e.allow_recurrent=true`): cycles are allowed. Node
  activations **persist across samples** (samples are treated as a time
  sequence). Each sample runs `e.relaxation_passes` activation sweeps over
  all non-input nodes in sorted-id order, reading from the previous pass's
  values for inputs from cyclic edges.
"""
function evaluate_genome(g::GraphGenome, e::GraphEvaluator)
    try
        if e.allow_recurrent
            return _evaluate_recurrent(g, e)
        else
            return _evaluate_feedforward(g, e)
        end
    catch e
        e isa InterruptException && rethrow()
        return Inf
    end
end

"""
    evaluate_cases(g::GraphGenome, e::GraphEvaluator) -> Vector{Float64}

Return per-sample mean squared error (averaged across outputs) as a
`Vector{Float64}` of length `size(e.input_data, 2)`. Any sample that
raises or produces a non-finite squared error is reported as `Inf`.
Used by lexicase selection.

Feedforward mode only: recurrent evaluators have persistent state across
samples (samples form a time sequence) so per-sample cases are not
independent. Calling this on a recurrent evaluator raises `ArgumentError`.
"""
function evaluate_cases(g::GraphGenome, e::GraphEvaluator)
    e.allow_recurrent && throw(ArgumentError(
        "evaluate_cases is not defined for recurrent GraphEvaluator: " *
        "samples are not independent when node state persists across them. " *
        "Lexicase selection requires per-case independence."))

    n_samples = size(e.input_data, 2)
    case_fitnesses = fill(Inf, n_samples)

    eval_order = try
        _topological_sort(g)
    catch err
        err isa InterruptException && rethrow()
        return case_fitnesses
    end
    eval_order === nothing && return case_fitnesses

    input_ids = sort!([n.id for n in values(g.nodes) if n.type == :input])
    output_ids = sort!([n.id for n in values(g.nodes) if n.type == :output])
    bias_ids = [n.id for n in values(g.nodes) if n.type == :bias]
    ff_order = [nid for nid in eval_order
                if haskey(g.nodes, nid) && !(g.nodes[nid].type in (:input, :bias))]
    n_outputs = length(output_ids)

    for s in 1:n_samples
        se_total = 0.0
        ok = true
        try
            node_vals = Dict{Int, Float64}()
            for (idx, nid) in enumerate(input_ids)
                node_vals[nid] = e.input_data[idx, s]
            end
            for nid in bias_ids
                node_vals[nid] = 1.0
            end
            _apply_node_activations!(node_vals, node_vals, ff_order, g, e.activation_fns)
            for (idx, nid) in enumerate(output_ids)
                predicted = get(node_vals, nid, 0.0)
                expected = e.output_data[idx, s]
                se = (predicted - expected)^2
                if !isfinite(se)
                    ok = false
                    break
                end
                se_total += se
            end
        catch err
            err isa InterruptException && rethrow()
            ok = false
        end
        if ok && n_outputs > 0
            case_fitnesses[s] = se_total / n_outputs
        end
    end

    return case_fitnesses
end

# ---------------------------------------------------------------------------
# Shared forward-pass helper (used by _evaluate_feedforward, _evaluate_recurrent,
# and EpisodicEvaluator). For each node in `update_order`, sums weighted inputs
# from `source`, applies the node's activation, and writes the result to
# `dest`. `dest` and `source` may be the same Dict (feedforward mode: reads of
# an in_node that was already updated this sweep see the new value) or
# different Dicts (recurrent mode: reads come from a prior snapshot).
# `update_order` should be pre-filtered to exclude :input and :bias nodes —
# callers seed those directly.
# ---------------------------------------------------------------------------

function _apply_node_activations!(
    dest::Dict{Int, Float64},
    source::Dict{Int, Float64},
    update_order::AbstractVector{Int},
    g::GraphGenome,
    activation_fns::Dict{Symbol, Function},
)
    for nid in update_order
        haskey(g.nodes, nid) || continue
        node = g.nodes[nid]
        total = 0.0
        for c in values(g.connections)
            if c.enabled && c.out_node == nid
                total += c.weight * get(source, c.in_node, 0.0)
            end
        end
        act_fn = get(activation_fns, node.activation, identity)
        dest[nid] = act_fn(total)
    end
    return dest
end

function _evaluate_feedforward(g::GraphGenome, e::GraphEvaluator)
    n_samples = size(e.input_data, 2)
    total_se = 0.0

    # Topological sort of enabled connections
    eval_order = _topological_sort(g)
    eval_order === nothing && return Inf

    # Get sorted input and output node IDs
    input_ids = sort!([n.id for n in values(g.nodes) if n.type == :input])
    output_ids = sort!([n.id for n in values(g.nodes) if n.type == :output])
    bias_ids = [n.id for n in values(g.nodes) if n.type == :bias]

    # Pre-filter topologically-sorted order to skip :input / :bias — those
    # are seeded directly per-sample and don't get activated.
    ff_order = [nid for nid in eval_order
                if haskey(g.nodes, nid) && !(g.nodes[nid].type in (:input, :bias))]

    for s in 1:n_samples
        # Initialize node values
        node_vals = Dict{Int, Float64}()

        for (idx, nid) in enumerate(input_ids)
            node_vals[nid] = e.input_data[idx, s]
        end
        for nid in bias_ids
            node_vals[nid] = 1.0
        end

        # Forward propagation in topological order — updates dest in place,
        # reading from the same dict so later nodes see earlier updates.
        _apply_node_activations!(node_vals, node_vals, ff_order, g, e.activation_fns)

        # Compute squared error for outputs
        for (idx, nid) in enumerate(output_ids)
            predicted = get(node_vals, nid, 0.0)
            expected = e.output_data[idx, s]
            se = (predicted - expected)^2
            isfinite(se) || return Inf
            total_se += se
        end
    end

    return total_se / (n_samples * length(output_ids))
end

function _evaluate_recurrent(g::GraphGenome, e::GraphEvaluator)
    n_samples = size(e.input_data, 2)
    total_se = 0.0

    input_ids  = sort!([n.id for n in values(g.nodes) if n.type == :input])
    output_ids = sort!([n.id for n in values(g.nodes) if n.type == :output])
    bias_ids   = [n.id for n in values(g.nodes) if n.type == :bias]

    # Sort all non-input/non-bias nodes by id for deterministic update order.
    # Recurrent edges consume the previous pass's value of their in_node;
    # forward edges consume the current pass's value if updated earlier in
    # the sweep. Sorted-id order matches NEAT's add-node id-growth pattern
    # (new hidden nodes get the next largest id), so the natural reading
    # order roughly matches topological order where it exists.
    update_ids = sort!([n.id for n in values(g.nodes) if !(n.type in (:input, :bias))])

    # Persistent state across samples (the "recurrent" in allow_recurrent).
    # Start at zero.
    node_vals = Dict{Int, Float64}(nid => 0.0 for nid in keys(g.nodes))

    for s in 1:n_samples
        # Overwrite inputs and bias for this timestep.
        for (idx, nid) in enumerate(input_ids)
            node_vals[nid] = e.input_data[idx, s]
        end
        for nid in bias_ids
            node_vals[nid] = 1.0
        end

        # Relaxation: run `relaxation_passes` sweeps, each reading from a
        # snapshot of the previous pass.
        for _ in 1:e.relaxation_passes
            prev = copy(node_vals)
            _apply_node_activations!(node_vals, prev, update_ids, g, e.activation_fns)
        end

        # Error at the end of the relaxation.
        for (idx, nid) in enumerate(output_ids)
            predicted = get(node_vals, nid, 0.0)
            expected = e.output_data[idx, s]
            se = (predicted - expected)^2
            isfinite(se) || return Inf
            total_se += se
        end
    end

    return total_se / (n_samples * length(output_ids))
end

"""Topological sort of network nodes. Returns nothing if cycle detected."""
function _topological_sort(g::GraphGenome)
    # Build adjacency and in-degree from enabled connections
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

    # Kahn's algorithm
    queue = [n for (n, d) in in_degree if d == 0]
    sort!(queue)  # deterministic ordering
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

"""Deep copy a GraphGenome."""
function _copy_graph(g::GraphGenome)
    new_nodes = Dict{Int, NodeGene}(id => n for (id, n) in g.nodes)
    new_conns = Dict{Int, ConnectionGene}()
    for (inn, c) in g.connections
        new_conns[inn] = ConnectionGene(c.in_node, c.out_node,
                                         c.weight, c.enabled, c.innovation)
    end
    GraphGenome(new_nodes, new_conns, g.n_inputs, g.n_outputs, g.fitness)
end

# =============================================================================
# EpisodicEvaluator — closed-loop control task evaluator
# =============================================================================

"""
    EpisodicEvaluator{FInit,FDyn,FRew,FDone,FObs,FDec} <: AbstractEvaluator

Evaluates a `GraphGenome` as a closed-loop policy on an episodic
environment defined by declarative callables. The network is treated as
`obs -> action`, and the evaluator drives the loop:

    for ep in 1:n_episodes
        rng   = MersenneTwister(episode_seed_base + ep)
        state = initial_state(rng)
        for step in 1:max_steps
            obs          = observe(state)
            net_output   = forward_network(state, obs)
            action       = decode_action(net_output)
            next_state   = dynamics(state, action)
            total       += reward(state, action, next_state)
            state        = next_state
            done(state) && break
        end
    end

Fitness is `-mean_reward_per_episode` (framework convention is
lower-is-better, so episodic tasks that want to *maximise* reward are
negated). On cycle detection in `allow_recurrent=false` mode, returns `Inf`.

# Fields
- `n_inputs::Int` / `n_outputs::Int` — dimensions the network expects,
  must match `length(observe(state))` and `length(net_output)`.
- `initial_state::FInit` — `rng -> state`. Must be reproducible from rng.
- `dynamics::FDyn` — `(state, action) -> next_state`.
- `reward::FRew` — `(state, action, next_state) -> Float64`.
- `done::FDone` — `state -> Bool`. Stops the episode early when `true`.
- `observe::FObs` — `state -> Vector{Float64}` of length `n_inputs`.
- `decode_action::FDec` — `Vector{Float64}` of length `n_outputs` → action.
- `max_steps::Int` — per-episode step cap.
- `n_episodes::Int` — rollouts averaged per `evaluate_genome` call.
- `episode_seed_base::Int` — `rng = MersenneTwister(base + ep_index)`.
- `activation_fns::Dict{Symbol,Function}` — defaults to `ACTIVATION_FNS`.
- `allow_recurrent::Bool` — defaults `true` (episodic tasks usually want
  persistent hidden-node state across timesteps).
- `relaxation_passes::Int` — recurrent-mode sweeps per step; default `1`.

# Design
The shape is declarative / pure-functional by default (see
memory/episodic_evaluator_design.md). For environments with heavy
reusable state (physics-engine handle, loaded dataset), a future
`StatefulEpisodicEvaluator` subtype can offer the `reset!`/`step!`
idiom; it is intentionally not built yet.

# Known limitations

- **Not parallel-safe for stateful environments.** The declarative
  API is structurally thread-safe when every callable is pure, but
  a `dynamics` closure that captures mutable state will race under
  `GeneticProgramming(; parallel=true)`. Use `parallel=false` for
  stateful environments until `StatefulEpisodicEvaluator` lands.
"""
struct EpisodicEvaluator{FInit,FDyn,FRew,FDone,FObs,FDec} <: AbstractEvaluator
    n_inputs::Int
    n_outputs::Int
    initial_state::FInit
    dynamics::FDyn
    reward::FRew
    done::FDone
    observe::FObs
    decode_action::FDec
    max_steps::Int
    n_episodes::Int
    episode_seed_base::Int
    activation_fns::Dict{Symbol,Function}
    allow_recurrent::Bool
    relaxation_passes::Int
end

"""
    EpisodicEvaluator(n_inputs, n_outputs, initial_state, dynamics, reward, done,
                      observe, decode_action; max_steps, n_episodes, ...)

Outer constructor. Keyword-argument defaults:

- `max_steps = 1000`
- `n_episodes = 1`
- `episode_seed_base = 0`
- `activation_fns = ACTIVATION_FNS`
- `allow_recurrent = true`
- `relaxation_passes = 1`
"""
function EpisodicEvaluator(n_inputs::Int, n_outputs::Int,
                            initial_state, dynamics, reward, done,
                            observe, decode_action;
                            max_steps::Int = 1000,
                            n_episodes::Int = 1,
                            episode_seed_base::Int = 0,
                            activation_fns::Dict{Symbol,Function} = ACTIVATION_FNS,
                            allow_recurrent::Bool = true,
                            relaxation_passes::Int = 1)
    EpisodicEvaluator(n_inputs, n_outputs,
                      initial_state, dynamics, reward, done,
                      observe, decode_action,
                      max_steps, n_episodes, episode_seed_base,
                      activation_fns, allow_recurrent, relaxation_passes)
end

input_signature(e::EpisodicEvaluator) =
    Dict(Symbol("x$i") => Float64 for i in 1:e.n_inputs)
output_signature(e::EpisodicEvaluator) =
    Dict(Symbol("y$i") => Float64 for i in 1:e.n_outputs)

"""
    evaluate_genome(g::GraphGenome, e::EpisodicEvaluator) -> Float64

Run `e.n_episodes` closed-loop rollouts of `g` as a policy on the
environment described by `e`, return `-mean_reward_per_episode`.
"""
function evaluate_genome(g::GraphGenome, e::EpisodicEvaluator)
    # Precompute node partitions used every step.
    input_ids  = sort!([n.id for n in values(g.nodes) if n.type == :input])
    output_ids = sort!([n.id for n in values(g.nodes) if n.type == :output])
    bias_ids   = [n.id for n in values(g.nodes) if n.type == :bias]

    length(input_ids) == e.n_inputs || return Inf
    length(output_ids) == e.n_outputs || return Inf

    local ff_order::Vector{Int}
    local update_ids::Vector{Int}
    if e.allow_recurrent
        update_ids = sort!([n.id for n in values(g.nodes)
                            if !(n.type in (:input, :bias))])
    else
        order = _topological_sort(g)
        order === nothing && return Inf   # cycle: illegal in feedforward mode
        ff_order = [nid for nid in order
                    if haskey(g.nodes, nid) && !(g.nodes[nid].type in (:input, :bias))]
    end

    total_reward = 0.0
    for ep in 1:e.n_episodes
        rng = Random.MersenneTwister(e.episode_seed_base + ep)
        state = e.initial_state(rng)

        # Persistent per-episode node state. For feedforward mode these
        # non-input/bias/output values are overwritten every step; for
        # recurrent mode they carry forward (the intended memory channel).
        node_vals = Dict{Int, Float64}(nid => 0.0 for nid in keys(g.nodes))

        ep_reward = 0.0
        step_count = 0
        while step_count < e.max_steps && !e.done(state)
            step_count += 1

            obs_vec = e.observe(state)
            length(obs_vec) == e.n_inputs || return Inf

            for (idx, nid) in enumerate(input_ids)
                node_vals[nid] = obs_vec[idx]
            end
            for nid in bias_ids
                node_vals[nid] = 1.0
            end

            if e.allow_recurrent
                for _ in 1:e.relaxation_passes
                    prev = copy(node_vals)
                    _apply_node_activations!(node_vals, prev, update_ids,
                                              g, e.activation_fns)
                end
            else
                _apply_node_activations!(node_vals, node_vals, ff_order,
                                          g, e.activation_fns)
            end

            output_vec = [get(node_vals, nid, 0.0) for nid in output_ids]
            action = e.decode_action(output_vec)

            next_state = e.dynamics(state, action)
            r = e.reward(state, action, next_state)
            isfinite(r) || return Inf
            ep_reward += r
            state = next_state
        end

        total_reward += ep_reward
    end

    return -(total_reward / e.n_episodes)
end

# =============================================================================
# GraphGenome solve method
# =============================================================================

"""
    solve(problem::GPProblem{GraphGenome}, algorithm::GeneticProgramming; ...) -> GPResult

Run NEAT-style evolution with GraphGenome. Handles initialization,
mutation, crossover with innovation-aligned genes, and speciation.

Accepts any `AbstractEvaluator` that implements `evaluate_genome(::GraphGenome, e)`
and whose `input_signature(e)` / `output_signature(e)` lengths match the
intended network dimensions — `GraphEvaluator` for table-based tasks,
`EpisodicEvaluator` for closed-loop control tasks.
"""
function solve(problem::GPProblem{GraphGenome, E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing,
               log::Union{Nothing, RunLog} = nothing) where {E<:AbstractEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    _validate_ops(algorithm.mutation_ops, algorithm.crossover_ops, GraphGenome;
                  mutation_rate=algorithm.mutation_rate,
                  crossover_rate=algorithm.crossover_rate)
    reset_innovation_counter!()

    evaluator = problem.evaluator
    n_in = length(input_signature(evaluator))
    n_out = length(output_signature(evaluator))
    pop_size = algorithm.pop_size
    bp = algorithm.bloat_penalty

    # Initialize population
    genomes = [initialize(GraphGenome, n_in, n_out, rng) for _ in 1:pop_size]
    fitnesses = fill(Inf, pop_size)

    for i in 1:pop_size
        fitnesses[i] = _evaluate_with_penalty(genomes[i], evaluator, bp)
        genomes[i].fitness = fitnesses[i]
    end

    species_state = _init_species_state(algorithm.speciation)
    fitness_history = Float64[]
    mean_history = Float64[]
    init_best = argmin(fitnesses)
    best_genome_all_time = deepcopy(genomes[init_best])
    best_fitness_all_time = fitnesses[init_best]
    t0 = time()

    for gen in 1:algorithm.generations
        order = sortperm(fitnesses)
        genomes = genomes[order]
        fitnesses = fitnesses[order]

        push!(fitness_history, fitnesses[1])
        finite_fits = filter(isfinite, fitnesses)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        if verbose
            println("Generation $gen: best=$(round(fitnesses[1], digits=6)), mean=$(round(mean_fit, digits=4))")
            flush(stdout)
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

        next_genomes = Vector{GraphGenome}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                 algorithm, rng, algorithm.elitism + 1;
                                 case_fitnesses=case_fitnesses)

        for i in (algorithm.elitism + 1):pop_size
            next_fitnesses[i] = _evaluate_with_penalty(next_genomes[i], evaluator, bp)
            next_genomes[i].fitness = next_fitnesses[i]  # NEAT crossover uses cached fitness
        end

        cur_best = argmin(next_fitnesses)
        if next_fitnesses[cur_best] < best_fitness_all_time
            best_fitness_all_time = next_fitnesses[cur_best]
            best_genome_all_time = deepcopy(next_genomes[cur_best])
        end

        genomes = next_genomes
        fitnesses = next_fitnesses
    end

    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    return GPResult{GraphGenome}(
        best_fitness_all_time < fitnesses[1] ? best_genome_all_time : genomes[1],
        min(best_fitness_all_time, fitnesses[1]),
        genomes,
        fitness_history, mean_history,
        algorithm.generations, time() - t0,
        _converged(min(best_fitness_all_time, fitnesses[1]),
                   algorithm.convergence_threshold)
    )
end

# =============================================================================
# IslandModel support — _initialize_population for GraphGenome
# =============================================================================
#
# The generic IslandModel solve path (src/solve.jl) calls
# `_initialize_population(problem, alg, rng)` per island and expects a
# `(genomes, state)` tuple where `state.rng` is consulted by the per-island
# loop. Reset of the innovation counter is the caller's responsibility —
# sequential IslandModel resets once before spawning islands so all islands
# share a coherent innovation history.
function _initialize_population(problem::GPProblem{GraphGenome, E},
                                 algorithm::GeneticProgramming,
                                 rng::AbstractRNG) where {E<:AbstractEvaluator}
    evaluator = problem.evaluator
    n_in  = length(input_signature(evaluator))
    n_out = length(output_signature(evaluator))
    pop_size = algorithm.pop_size

    genomes = Vector{GraphGenome}(undef, pop_size)
    for i in 1:pop_size
        genomes[i] = initialize(GraphGenome, n_in, n_out, rng)
    end

    return (genomes, GraphGenomeContext(rng, n_in, n_out))
end
