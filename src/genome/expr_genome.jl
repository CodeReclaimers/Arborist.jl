"""
    ExprGenome <: AbstractGenome

Genome representation based on Julia `Expr` trees. Wraps the existing
`codegen.jl` / `evolution.jl` infrastructure.

# Fields
- `body::Vector{Expr}`: body statements (not yet wrapped in a function harness)
- `state::GenState`: type context carrying variable types, function set, etc.

# Known limitations

- **`serialize` / `deserialize` round-trip is ~80% reliable.**
  `repr()`-style `Float32(literal)` forms produced by the Julia
  printer fail type-checking on round-trip. The LLM operator falls
  back silently to a classical operator, but checkpoint/resume or
  cross-process migration can lose a fraction of individuals.
- **`@eval` grows Julia's method table monotonically** across
  generations. Long runs (thousands of generations × hundreds of
  individuals) accumulate tens of thousands of methods, slowing
  dispatch. Use `TreeGenome` for long runs where applicable.
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

Accepts assignments, `while` loops, `if`/`if-else` statements, `for` loops,
blocks, `break`, `continue`, and standalone function calls. Multi-line
control flow is supported by parsing the entire string as a block.

Statements that fail parsing or type-checking are skipped (partial recovery)
rather than rejecting the whole genome.

Does not eval anything; parse only.
"""
function deserialize(::Type{ExprGenome}, s::String,
                     state::GenState)::Union{ExprGenome, Nothing}
    # Try to parse the whole string as a block to handle multi-line control flow.
    stmts = _parse_statements(s)

    valid_stmts = Expr[]
    for expr in stmts
        # Unwrap QuoteNode from repr()-style :() output.
        if expr isa Expr && expr.head == :quote && length(expr.args) == 1 && expr.args[1] isa Expr
            expr = expr.args[1]
        end
        if expr isa Expr && _is_valid_statement(expr, state)
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
    _parse_statements(s::String) -> Vector{Any}

Parse a string into a list of top-level statements. First tries to parse
the whole string as a block (to handle multi-line control flow like
`while...end`). Falls back to line-by-line parsing if block parsing fails.
"""
function _parse_statements(s::String)
    # Try block parse for multi-line control flow.
    block = try
        Meta.parse("begin\n" * s * "\nend")
    catch e
        e isa InterruptException && rethrow()
        nothing
    end
    if block isa Expr && block.head == :block
        # Extract non-LineNumberNode statements.
        return [a for a in block.args if !(a isa LineNumberNode)]
    end

    # Fallback: line-by-line parsing.
    lines = filter(!isempty, strip.(split(s, "\n")))
    stmts = Any[]
    for line in lines
        expr = try
            Meta.parse(line)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        expr === nothing || push!(stmts, expr)
    end
    return stmts
end

"""
    _is_valid_statement(expr::Expr, state::GenState) -> Bool

Check that an expression is a valid statement: assignment, while loop,
if/if-else, for loop, block, or standalone function call. Validates
recursively for compound statements.
"""
function _is_valid_statement(expr::Expr, state::GenState)::Bool
    h = expr.head
    h == :(=) && return _is_valid_assignment(expr, state)
    h == :while && return _is_valid_while(expr, state)
    h == :if && return _is_valid_if(expr, state)
    h == :for && return _is_valid_for(expr, state)
    h == :block && return _is_valid_block(expr, state)
    h == :call && return _is_valid_call(expr, state)
    h == :break && return true
    h == :continue && return true
    return false
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

        # If the rvalue is a bare numeric literal and the types don't match,
        # try to coerce it to the lvalue's type. LLMs commonly produce `0`
        # (Int64) instead of `Int32(0)`, or `0.0` (Float64) instead of
        # `Float32(0.0)`. Coercion is safe for literals — no precision loss
        # for small integers, and Float64→Float32 is an explicit narrowing
        # the user would write anyway.
        if rhs isa Number
            rtype = typeof(rhs)
            if ltype != rtype
                coerced = try
                    convert(ltype, rhs)
                catch
                    return false
                end
                expr.args[2] = coerced
            end
            return true
        end

        rtype = get_rvalue_type(state, rhs)
        return ltype == rtype
    catch e
        e isa InterruptException && rethrow()
        return false
    end
end

"""Check that a while loop has a Bool-typed condition and valid body."""
function _is_valid_while(expr::Expr, state::GenState)::Bool
    length(expr.args) == 2 || return false
    try
        cond_type = get_rvalue_type(state, expr.args[1])
        cond_type == Bool || return false
    catch e
        e isa InterruptException && rethrow()
        return false
    end
    body = expr.args[2]
    body isa Expr || return true  # empty body is valid
    return _is_valid_body(body, state)
end

"""Check that an if/if-else has a Bool-typed condition and valid branches."""
function _is_valid_if(expr::Expr, state::GenState)::Bool
    length(expr.args) >= 2 || return false
    try
        cond_type = get_rvalue_type(state, expr.args[1])
        cond_type == Bool || return false
    catch e
        e isa InterruptException && rethrow()
        return false
    end
    # Validate then-branch.
    then_branch = expr.args[2]
    if then_branch isa Expr && !_is_valid_body(then_branch, state)
        return false
    end
    # Validate else-branch if present.
    if length(expr.args) >= 3
        else_branch = expr.args[3]
        if else_branch isa Expr
            # else-branch can be another :if (elseif) or a :block
            if else_branch.head == :if
                return _is_valid_if(else_branch, state)
            elseif !_is_valid_body(else_branch, state)
                return false
            end
        end
    end
    return true
end

"""Check that a for loop has a valid iterator and body."""
function _is_valid_for(expr::Expr, state::GenState)::Bool
    length(expr.args) == 2 || return false
    # args[1] is the iterator assignment (e.g., :(i = 1:10))
    iter = expr.args[1]
    iter isa Expr && iter.head == :(=) || return false
    # We don't type-check the iterator variable — it's loop-local.
    body = expr.args[2]
    body isa Expr || return true
    return _is_valid_body(body, state)
end

"""Check that a block contains at least one valid statement."""
function _is_valid_block(expr::Expr, state::GenState)::Bool
    expr.head == :block || return false
    for a in expr.args
        a isa LineNumberNode && continue
        a isa Expr || continue
        if _is_valid_statement(a, state)
            return true
        end
    end
    return false
end

"""Check that a standalone function call uses a known function."""
function _is_valid_call(expr::Expr, state::GenState)::Bool
    expr.head == :call || return false
    length(expr.args) >= 1 || return false
    fn = expr.args[1]
    fn isa Symbol || return false
    # Check against the function set.
    for fd in state.funcs.funcs
        if fd.name == fn
            return true
        end
    end
    return false
end

"""Validate the body of a control flow statement (block or single statement)."""
function _is_valid_body(body::Expr, state::GenState)::Bool
    if body.head == :block
        # At least one statement must be valid; invalid ones are tolerated.
        for a in body.args
            a isa LineNumberNode && continue
            if a isa Expr && _is_valid_statement(a, state)
                return true
            end
            # break/continue as bare Symbols
            if a isa Symbol && a in (:break, :continue)
                return true
            end
        end
        return false
    else
        return _is_valid_statement(body, state)
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
    catch e
        e isa InterruptException && rethrow()
        return Inf
    end
end

"""
    evaluate_cases(g::ExprGenome, e::TableFitnessEvaluator) -> Vector{Float64}

Compile the genome and return per-row squared error via the
`TableFitnessEvaluator` case evaluator. All rows `Inf` on compilation failure.
"""
function evaluate_cases(g::ExprGenome, e::TableFitnessEvaluator)
    fname = gensym("evolved")
    try
        checked_body = add_loop_checks(g.body)
        harness = create_harness(g.state, checked_body, fname)
        f = @eval $harness
        return evaluate_cases(e, f)
    catch err
        err isa InterruptException && rethrow()
        return fill(Inf, length(e.input_rows))
    end
end
