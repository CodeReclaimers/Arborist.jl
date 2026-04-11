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
    lock(_innovation_lock) do
        _innovation_counter[] = 0
    end
end

# =============================================================================
# Node and connection genes
# =============================================================================

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
"""
mutable struct GraphGenome <: AbstractGenome
    nodes::Dict{Int, NodeGene}
    connections::Dict{Int, ConnectionGene}
    n_inputs::Int
    n_outputs::Int
    fitness::Float64
end

# =============================================================================
# Activation functions
# =============================================================================

"""
    ACTIVATION_FNS

Dictionary mapping activation `Symbol` names to their unary `Function`
implementations, used by `GraphEvaluator` when propagating values through a
`GraphGenome`. The default set is `:sigmoid` (NEAT-style steepened logistic),
`:tanh`, `:relu`, and `:identity`. New activations can be added by assigning
into this dict before solving; each `NodeGene` stores the activation as a
`Symbol` and looks the function up here at evaluation time.
"""
const ACTIVATION_FNS = Dict{Symbol, Function}(
    :sigmoid  => x -> 1.0 / (1.0 + exp(-4.9 * x)),
    :tanh     => x -> tanh(x),
    :relu     => x -> max(0.0, x),
    :identity => x -> x,
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
    # NEAT crossover: disjoint/excess genes come from the fitter parent.
    # Use cached fitness to determine which parent is fitter (lower = better).
    # Standard NEAT produces one child; the framework needs two.
    # The second child swaps parent order so that disjoint/excess genes
    # from the less-fit parent are also preserved in the population.
    if g1.fitness <= g2.fitness
        child1 = _neat_crossover(g1, g2, rng)
        child2 = _neat_crossover(g2, g1, rng)
    else
        child1 = _neat_crossover(g2, g1, rng)
        child2 = _neat_crossover(g1, g2, rng)
    end
    return (child1, child2)
end

# Operator-dispatch fallbacks for shared _breed_next_generation!.
crossover(::AbstractCrossoverOperator, g1::GraphGenome, g2::GraphGenome, rng::AbstractRNG) = crossover(g1, g2, rng)
mutate(::AbstractMutationOperator, g::GraphGenome, rng::AbstractRNG) = mutate(g, rng)

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

function deserialize(::Type{GraphGenome}, s::String,
                     n_inputs::Int, n_outputs::Int)
    @warn "GraphGenome deserialization is not yet implemented"
    return nothing
end

# =============================================================================
# Mutation operators
# =============================================================================

function _mutate_weights!(g::GraphGenome, rng::AbstractRNG)
    # Perturb each enabled weight independently with 90% probability.
    # Standard NEAT: each weight has an independent chance of perturbation.
    # Sort by innovation number for deterministic RNG consumption order.
    for c in sort!(collect(values(g.connections)), by=c -> c.innovation)
        if c.enabled && rand(rng) < 0.9
            c.weight += randn(rng) * 0.3
        end
    end
end

function _mutate_weight_replace!(g::GraphGenome, rng::AbstractRNG)
    conns = sort!(collect(values(g.connections)), by=c -> c.innovation)
    isempty(conns) && return
    c = rand(rng, conns)
    c.weight = randn(rng) * 2.0
end

function _mutate_add_connection!(g::GraphGenome, rng::AbstractRNG)
    node_ids = sort!(collect(keys(g.nodes)))
    length(node_ids) < 2 && return

    # Try up to 20 times to find a valid new connection
    for _ in 1:20
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

function _mutate_add_node!(g::GraphGenome, rng::AbstractRNG)
    enabled_conns = sort!([c for c in values(g.connections) if c.enabled], by=c -> c.innovation)
    isempty(enabled_conns) && return

    old_conn = rand(rng, enabled_conns)
    old_conn.enabled = false

    # New hidden node
    new_id = maximum(keys(g.nodes)) + 1
    activation = rand(rng, [:sigmoid, :tanh, :relu])
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
"""
struct GraphEvaluator <: AbstractEvaluator
    input_data::Matrix{Float64}    # n_inputs × n_samples
    output_data::Matrix{Float64}   # n_outputs × n_samples
    activation_fns::Dict{Symbol, Function}
end

"""
    GraphEvaluator(input_data, output_data)

Construct a `GraphEvaluator` with default activation functions.
"""
function GraphEvaluator(input_data::Matrix{Float64}, output_data::Matrix{Float64})
    GraphEvaluator(input_data, output_data, ACTIVATION_FNS)
end

input_signature(e::GraphEvaluator) = Dict(Symbol("x$i") => Float64 for i in 1:size(e.input_data, 1))
output_signature(e::GraphEvaluator) = Dict(Symbol("y$i") => Float64 for i in 1:size(e.output_data, 1))

"""
    evaluate_genome(g::GraphGenome, e::GraphEvaluator) -> Float64

Evaluate a GraphGenome by forward-propagating inputs through the network.
Returns mean squared error against target outputs. Returns Inf if the
network contains cycles or evaluation throws.
"""
function evaluate_genome(g::GraphGenome, e::GraphEvaluator)
    try
        n_samples = size(e.input_data, 2)
        total_se = 0.0

        # Topological sort of enabled connections
        eval_order = _topological_sort(g)
        eval_order === nothing && return Inf

        # Get sorted input and output node IDs
        input_ids = sort!([n.id for n in values(g.nodes) if n.type == :input])
        output_ids = sort!([n.id for n in values(g.nodes) if n.type == :output])
        bias_ids = [n.id for n in values(g.nodes) if n.type == :bias]

        for s in 1:n_samples
            # Initialize node values
            node_vals = Dict{Int, Float64}()

            for (idx, nid) in enumerate(input_ids)
                node_vals[nid] = e.input_data[idx, s]
            end
            for nid in bias_ids
                node_vals[nid] = 1.0
            end

            # Forward propagation in topological order
            for nid in eval_order
                haskey(g.nodes, nid) || continue
                node = g.nodes[nid]
                node.type in (:input, :bias) && continue

                # Sum incoming weights
                total = 0.0
                for c in values(g.connections)
                    if c.enabled && c.out_node == nid
                        total += c.weight * get(node_vals, c.in_node, 0.0)
                    end
                end

                # Apply activation
                act_fn = get(e.activation_fns, node.activation, identity)
                node_vals[nid] = act_fn(total)
            end

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
    catch e
        e isa InterruptException && rethrow()
        return Inf
    end
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
# GraphGenome solve method
# =============================================================================

"""
    solve(problem::GPProblem{GraphGenome}, algorithm::GeneticProgramming; ...) -> GPResult

Run NEAT-style evolution with GraphGenome. Handles initialization,
mutation, crossover with innovation-aligned genes, and speciation.
"""
function solve(problem::GPProblem{GraphGenome, E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing) where {E<:GraphEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    reset_innovation_counter!()

    evaluator = problem.evaluator
    n_in = size(evaluator.input_data, 1)
    n_out = size(evaluator.output_data, 1)
    pop_size = algorithm.pop_size

    # Initialize population
    genomes = [initialize(GraphGenome, n_in, n_out, rng) for _ in 1:pop_size]
    fitnesses = fill(Inf, pop_size)

    for i in 1:pop_size
        fitnesses[i] = evaluate_genome(genomes[i], evaluator)
        genomes[i].fitness = fitnesses[i]
    end

    species_state = _init_species_state(algorithm.speciation)
    fitness_history = Float64[]
    mean_history = Float64[]
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

        selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                  algorithm.speciation, species_state, rng)

        next_genomes = Vector{GraphGenome}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                 algorithm, rng, algorithm.elitism + 1)

        for i in (algorithm.elitism + 1):pop_size
            next_fitnesses[i] = evaluate_genome(next_genomes[i], evaluator)
            next_genomes[i].fitness = next_fitnesses[i]  # NEAT crossover uses cached fitness
        end

        genomes = next_genomes
        fitnesses = next_fitnesses
    end

    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    return GPResult{GraphGenome}(
        genomes[1], fitnesses[1], genomes,
        fitness_history, mean_history,
        algorithm.generations, time() - t0,
        fitnesses[1] < algorithm.convergence_threshold
    )
end
