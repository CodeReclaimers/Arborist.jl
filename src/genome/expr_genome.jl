"""
    ExprGenome <: AbstractGenome

Genome representation based on Julia `Expr` trees. Wraps the existing
`codegen.jl` / `evolution.jl` infrastructure.

# Fields
- `body::Vector{Expr}`: body statements (not yet wrapped in a function harness)
- `state::GenState`: type context carrying variable types, function set, etc.
"""
struct ExprGenome <: AbstractGenome
    body::Vector{Expr}
    state::GenState
end

"""
    GPProblem{G<:AbstractGenome, E<:AbstractEvaluator}

Problem specification for genetic programming. Combines an evaluator
(which defines the fitness landscape) with a genome type and configuration.

# Fields
- `evaluator::E`: the fitness evaluator
- `genome_type::Type{G}`: the genome type to evolve
- `function_set::FunctionSet`: available functions for code generation
- `num_temps::Int`: number of temporary variables per genome
- `seed::Union{Int, Nothing}`: random seed for reproducibility (`nothing` for no seeding)
"""
struct GPProblem{G<:AbstractGenome, E<:AbstractEvaluator}
    evaluator::E
    genome_type::Type{G}
    function_set::FunctionSet
    num_temps::Int
    seed::Union{Int, Nothing}
end

"""
    GPProblem(evaluator, ::Type{G}; function_set, num_temps, seed) -> GPProblem

Construct a `GPProblem` with keyword arguments and sensible defaults.
"""
function GPProblem(evaluator::E, ::Type{G};
                   function_set::FunctionSet=default_function_set(),
                   num_temps::Int=4,
                   seed::Union{Int, Nothing}=nothing) where {G<:AbstractGenome, E<:AbstractEvaluator}
    GPProblem{G,E}(evaluator, G, function_set, num_temps, seed)
end

# --- AbstractGenome interface implementation for ExprGenome ---

"""
    initialize(::Type{ExprGenome}, problem::GPProblem) -> ExprGenome

Create a random `ExprGenome` using the problem's function set and evaluator signatures.
"""
function initialize(::Type{ExprGenome}, problem::GPProblem)
    inputs = input_signature(problem.evaluator)
    outputs = output_signature(problem.evaluator)
    state = GenState(problem.function_set, inputs, outputs, problem.num_temps)
    body = [create_random_assignment(state) for _ in 1:3]
    return ExprGenome(body, state)
end

"""
    mutate(g::ExprGenome, rng::AbstractRNG) -> ExprGenome

Produce a mutated copy of the genome by applying a random point mutation
to a randomly selected sub-expression.
"""
function mutate(g::ExprGenome, rng::AbstractRNG)
    return mutate(PointMutation(), g, rng)
end

"""
    crossover(g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG) -> Tuple{ExprGenome, ExprGenome}

Produce two offspring via subtree crossover.
"""
function crossover(g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG)
    return crossover(SubtreeCrossover(), g1, g2, rng)
end

"""
    distance(g1::ExprGenome, g2::ExprGenome) -> Float64

Structural compatibility distance. Counts Expr nodes appearing in one
program but not the other after type-normalizing.
"""
function distance(g1::ExprGenome, g2::ExprGenome)
    set1 = _expr_node_set(g1.body)
    set2 = _expr_node_set(g2.body)
    return Float64(length(symdiff(set1, set2)))
end

"""
    complexity(g::ExprGenome) -> Float64

Total node count across all body statements, measured via `unravel`.
"""
function complexity(g::ExprGenome)
    count = 0
    for stmt in g.body
        count += length(unravel(stmt))
    end
    return Float64(count)
end

"""
    serialize(g::ExprGenome) -> String

Convert an ExprGenome body to a human-readable Julia source string
suitable for inclusion in an LLM prompt. Each statement is printed
on its own line using Julia's standard pretty-printer.
"""
function serialize(g::ExprGenome)::String
    io = IOBuffer()
    for (i, stmt) in enumerate(g.body)
        print(io, repr(stmt))
        i < length(g.body) && print(io, "\n")
    end
    return String(take!(io))
end

"""
    deserialize(::Type{ExprGenome}, s::String, state::GenState) -> Union{ExprGenome, Nothing}

Parse a string of Julia statements into an ExprGenome. Returns `nothing`
if zero valid statements survive parsing and type-checking.

Each line is parsed with `Meta.parse`. Successfully parsed assignment
expressions are type-checked against the GenState using `get_lvalue_type`
and `get_rvalue_type`. Lines that fail parsing or type-checking are
skipped (partial recovery) rather than rejecting the whole genome.

Does not eval anything; parse only.
"""
function deserialize(::Type{ExprGenome}, s::String,
                     state::GenState)::Union{ExprGenome, Nothing}
    lines = filter(!isempty, strip.(split(s, "\n")))
    valid_stmts = Expr[]
    for line in lines
        expr = try
            Meta.parse(line)
        catch
            nothing
        end
        expr isa Expr || continue
        # Unwrap QuoteNode from repr()-style :() output.
        # repr(:(y = x)) produces ":(y = x)" which Meta.parse returns
        # as Expr(:quote, :(y = x)).
        if expr.head == :quote && length(expr.args) == 1 && expr.args[1] isa Expr
            expr = expr.args[1]
        end
        # Verify it is a valid assignment with consistent types.
        if _is_valid_assignment(expr, state)
            push!(valid_stmts, expr)
        end
    end
    isempty(valid_stmts) && return nothing
    return ExprGenome(valid_stmts, state)
end

"""
    deserialize(::Type{ExprGenome}, s::String; state::Union{GenState, Nothing}=nothing) -> Union{ExprGenome, Nothing}

Backward-compatible keyword-argument version. Delegates to the positional
version when `state` is provided; returns `nothing` when it is not.
"""
function deserialize(::Type{ExprGenome}, s::String; state::Union{GenState, Nothing}=nothing)
    state === nothing && return nothing
    return deserialize(ExprGenome, s, state)
end

"""
    _is_valid_assignment(expr::Expr, state::GenState) -> Bool

Check that an expression is a valid assignment with type-consistent
lvalue and rvalue according to the GenState.
"""
function _is_valid_assignment(expr::Expr, state::GenState)::Bool
    expr.head == :(=) || return false
    length(expr.args) == 2 || return false
    try
        lhs = expr.args[1]
        rhs = expr.args[2]
        ltype = get_lvalue_type(state, lhs)
        rtype = get_rvalue_type(state, rhs)
        return ltype == rtype
    catch
        return false
    end
end


# --- Internal helpers ---

"""
    _normalize_expr(expr::Expr) -> Expr

Type-normalize an expression: replace all literals with a Symbol representing
their type. Used for structural comparison in `distance`.
"""
function _normalize_expr(expr::Expr)
    new_args = map(expr.args) do a
        if a isa Expr
            _normalize_expr(a)
        elseif a isa Number
            Symbol(typeof(a))
        else
            a
        end
    end
    return Expr(expr.head, new_args...)
end

"""
    _expr_node_set(body::Vector{Expr}) -> Set{String}

Collect the set of normalized Expr node string representations from a body.
"""
function _expr_node_set(body::Vector{Expr})
    nodes = Set{String}()
    for stmt in body
        for node in unravel(stmt)
            push!(nodes, string(_normalize_expr(node)))
        end
    end
    return nodes
end

"""
    evaluate_genome(g::ExprGenome, evaluator::AbstractEvaluator) -> Float64

Compile and evaluate an ExprGenome against the given evaluator.
Returns `Inf` on any compilation or evaluation failure.
"""
function evaluate_genome(g::ExprGenome, evaluator::AbstractEvaluator)
    fname = gensym("evolved")
    try
        checked_body = add_loop_checks(g.body)
        harness = create_harness(g.state, checked_body, fname)
        f = @eval $harness
        return evaluate(evaluator, f)
    catch
        return Inf
    end
end
