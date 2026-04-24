# Graphviz DOT-format export for genome visualization.
#
# Each `to_dot(g)` method returns a self-contained DOT document as a `String`
# that can be piped through the `dot` binary or any Graphviz-consuming tool:
#
#     open("tree.dot", "w") do io
#         print(io, to_dot(tree_genome))
#     end
#     run(`dot -Tsvg tree.dot -o tree.svg`)
#
# Alternatively, `to_dot(io, g)` streams directly to an `IO` target.
#
# The `dot` binary itself is NOT a runtime dependency — these functions only
# produce the source document. The user invokes Graphviz separately.

"""
    to_dot(g) -> String
    to_dot(io::IO, g)

Produce a Graphviz DOT document describing the genome `g`. Supported inputs
are `TreeGenome`, `ExprGenome`, `ADFGenome`, `AntGenome`, and `GraphGenome`.

For tree-structured genomes the output is a directed acyclic graph with
ellipse-shaped nodes labeled by operator / constant / variable. For
`GraphGenome` the output is a left-to-right network diagram with distinct
node shapes by role (input, output, bias, hidden) and edges labeled by
connection weight, with disabled connections shown dashed and gray.

The function returns the document as a `String`. The two-argument form
writes to `io` and returns `io` for chaining. Tree-genome methods do not
throw on NaN / Inf constants; they are rendered literally.
"""
function to_dot end

to_dot(g) = sprint(to_dot, g)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Escape double quotes and backslashes for DOT label safety.
_dot_escape(s) = replace(string(s), "\\" => "\\\\", "\"" => "\\\"")

# Round a floating-point value to 2 decimals for edge/weight labels.
_dot_num(x::Real) = isfinite(x) ? string(round(float(x); digits=2)) : string(x)
_dot_num(x) = _dot_escape(string(x))

# ---------------------------------------------------------------------------
# TreeGenome (DynamicExpressions.Node)
# ---------------------------------------------------------------------------

function to_dot(io::IO, g::TreeGenome)
    println(io, "digraph TreeGenome {")
    println(io, "  rankdir=TB;")
    println(io, "  node [shape=ellipse];")
    counter = Ref(0)
    _tree_to_dot!(io, g.tree, g.operators, counter)
    println(io, "}")
    return io
end

# Walk a DynamicExpressions node; emit one DOT node and recurse into children.
# Returns the integer id assigned to the node so the caller can draw the edge.
function _tree_to_dot!(io::IO, node, ops::OperatorEnum, counter::Ref{Int})
    counter[] += 1
    id = counter[]
    if node.degree == 0
        label = node.constant ? _dot_num(node.val) : "x$(node.feature)"
        println(io, "  node_$id [label=\"", label, "\"];")
    elseif node.degree == 1
        op_name = string(ops.unaops[node.op])
        println(io, "  node_$id [label=\"", _dot_escape(op_name), "\"];")
        child_id = _tree_to_dot!(io, node.l, ops, counter)
        println(io, "  node_$id -> node_$child_id;")
    else  # degree == 2
        op_name = string(ops.binops[node.op])
        println(io, "  node_$id [label=\"", _dot_escape(op_name), "\"];")
        left_id  = _tree_to_dot!(io, node.l, ops, counter)
        right_id = _tree_to_dot!(io, node.r, ops, counter)
        println(io, "  node_$id -> node_$left_id;")
        println(io, "  node_$id -> node_$right_id;")
    end
    return id
end

# ---------------------------------------------------------------------------
# ExprGenome (Julia Expr)
# ---------------------------------------------------------------------------

function to_dot(io::IO, g::ExprGenome)
    println(io, "digraph ExprGenome {")
    println(io, "  rankdir=TB;")
    println(io, "  node [shape=ellipse];")
    counter = Ref(0)
    for (i, stmt) in enumerate(g.body)
        println(io, "  subgraph cluster_stmt_$i {")
        println(io, "    label=\"stmt $i\";")
        println(io, "    style=dashed;")
        _expr_to_dot!(io, stmt, counter)
        println(io, "  }")
    end
    println(io, "}")
    return io
end

# Convert a Julia Expr / Symbol / Number to DOT. Non-Expr values are
# leaves labeled by their `repr`. Returns the integer id of the emitted node.
function _expr_to_dot!(io::IO, x, counter::Ref{Int})
    counter[] += 1
    id = counter[]
    if x isa Expr
        # Use the head as the label for control-flow constructs (:=, :if,
        # :while, :block, ...); for :call, use the callee name.
        label = x.head === :call && !isempty(x.args) ? string(x.args[1]) : string(x.head)
        println(io, "    node_$id [label=\"", _dot_escape(label), "\"];")
        # Skip args[1] for :call (it's the function name, already the label).
        start_idx = x.head === :call ? 2 : 1
        for i in start_idx:length(x.args)
            cid = _expr_to_dot!(io, x.args[i], counter)
            println(io, "    node_$id -> node_$cid;")
        end
    else
        println(io, "    node_$id [label=\"", _dot_escape(string(x)),
                    "\", shape=box, style=filled, fillcolor=lightyellow];")
    end
    return id
end

# ---------------------------------------------------------------------------
# AntGenome (single Expr tree)
# ---------------------------------------------------------------------------

function to_dot(io::IO, g::AntGenome)
    println(io, "digraph AntGenome {")
    println(io, "  rankdir=TB;")
    println(io, "  node [shape=ellipse];")
    counter = Ref(0)
    _expr_to_dot!(io, g.program, counter)
    println(io, "}")
    return io
end

# ---------------------------------------------------------------------------
# ADFGenome (main + ADF subgraphs)
# ---------------------------------------------------------------------------

function to_dot(io::IO, g::ADFGenome)
    base_ops = base_operators(g)
    println(io, "digraph ADFGenome {")
    println(io, "  rankdir=TB;")
    println(io, "  node [shape=ellipse];")
    counter = Ref(0)
    # Main tree in its own cluster.
    println(io, "  subgraph cluster_main {")
    println(io, "    label=\"main\";")
    println(io, "    style=dashed;")
    _tree_to_dot!(io, g.main, g.operators, counter)
    println(io, "  }")
    # Each ADF body in its own cluster, using the base (non-augmented) ops.
    for (i, adf) in enumerate(g.adfs)
        println(io, "  subgraph cluster_adf_$(i - 1) {")
        println(io, "    label=\"ADF", i - 1, "\";")
        println(io, "    style=dashed;")
        _tree_to_dot!(io, adf, base_ops, counter)
        println(io, "  }")
    end
    println(io, "}")
    return io
end

# ---------------------------------------------------------------------------
# GraphGenome (NEAT-style network)
# ---------------------------------------------------------------------------

function to_dot(io::IO, g::GraphGenome)
    println(io, "digraph GraphGenome {")
    println(io, "  rankdir=LR;")
    println(io, "  node [fontname=\"Helvetica\"];")
    println(io, "  edge [fontsize=10, fontname=\"Helvetica\"];")

    # Group nodes by role for deterministic ordering + rank grouping.
    input_ids  = Int[]
    output_ids = Int[]
    bias_ids   = Int[]
    hidden_ids = Int[]
    for id in sort!(collect(keys(g.nodes)))
        n = g.nodes[id]
        if     n.type === :input;  push!(input_ids, id)
        elseif n.type === :output; push!(output_ids, id)
        elseif n.type === :bias;   push!(bias_ids, id)
        else                       push!(hidden_ids, id)
        end
    end

    # Render each node with a role-specific style.
    for id in input_ids
        println(io, "  node_$id [label=\"in$id\", shape=box, ",
                     "style=filled, fillcolor=lightblue];")
    end
    for id in bias_ids
        println(io, "  node_$id [label=\"bias\", shape=diamond, ",
                     "style=filled, fillcolor=lightgray];")
    end
    for id in hidden_ids
        act = g.nodes[id].activation
        println(io, "  node_$id [label=\"", act, "\", shape=circle];")
    end
    for id in output_ids
        act = g.nodes[id].activation
        println(io, "  node_$id [label=\"out$id\\n", act,
                    "\", shape=box, style=filled, fillcolor=lightcoral];")
    end

    # Rank-group inputs and outputs to put them at the extremes.
    if !isempty(input_ids) || !isempty(bias_ids)
        ids = vcat(input_ids, bias_ids)
        print(io, "  { rank=min; ")
        for id in ids; print(io, "node_$id; "); end
        println(io, "}")
    end
    if !isempty(output_ids)
        print(io, "  { rank=max; ")
        for id in output_ids; print(io, "node_$id; "); end
        println(io, "}")
    end

    # Emit connections sorted by innovation number for determinism.
    for inn in sort!(collect(keys(g.connections)))
        c = g.connections[inn]
        w = _dot_num(c.weight)
        if c.enabled
            println(io, "  node_$(c.in_node) -> node_$(c.out_node) ",
                        "[label=\"", w, "\"];")
        else
            println(io, "  node_$(c.in_node) -> node_$(c.out_node) ",
                        "[label=\"", w, "\", style=dashed, color=gray];")
        end
    end

    println(io, "}")
    return io
end
