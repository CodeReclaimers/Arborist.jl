"""
    ASTSanitizer

Validates ExprGenome expression trees against a whitelist of permitted
function calls before `@eval` compilation. Rejects any expression
containing calls to functions outside the whitelist.

This is a defense-in-depth measure for use with `LLMMutationOperator`.
For purely classical GP (no LLM operator), the function set already
constrains what can appear, but sanitization adds an explicit check.

# Fields
- `allowed_calls::Set{Symbol}`: whitelist of permitted function call symbols
- `allow_literals::Bool`: whether to allow literal values (default: true)
- `allow_variables::Bool`: whether to allow variable references (default: true)
"""
struct ASTSanitizer
    allowed_calls::Set{Symbol}
    allow_literals::Bool
    allow_variables::Bool
end

"""
    ASTSanitizer(; allowed_calls=DEFAULT_SAFE_CALLS, allow_literals=true, allow_variables=true)

Construct an `ASTSanitizer` with the default mathematical/logical whitelist.
"""
function ASTSanitizer(; allowed_calls::Set{Symbol}=copy(DEFAULT_SAFE_CALLS),
                       allow_literals::Bool=true,
                       allow_variables::Bool=true)
    ASTSanitizer(allowed_calls, allow_literals, allow_variables)
end

"""
Default whitelist of safe function calls — mathematical and logical operations only.
"""
const DEFAULT_SAFE_CALLS = Set{Symbol}([
    # Arithmetic
    :+, :-, :*, :/, :^, :%, :div, :mod, :rem,
    # Math functions
    :sin, :cos, :tan, :exp, :log, :log2, :log10,
    :sqrt, :abs, :sign, :floor, :ceil, :round,
    :min, :max, :clamp,
    # Comparison
    :>, :<, :(==), :!=, :>=, :<=,
    # Boolean
    :&, :|, :!, :xor,
    # Type conversion
    :Float32, :Float64, :Int32, :Int64, :Bool,
    # Safe Julia builtins
    :ifelse, :typemax, :typemin, :zero, :one,
    # Arborist boolean operators
    :gp_nand, :gp_nor,
    # Range construction (used in for loops)
    :(:),
])

"""
    sanitize(san::ASTSanitizer, expr::Expr) -> Bool

Return `true` if the expression tree is safe (all function calls are in the
whitelist), `false` if it contains any unsafe call. Walks the entire AST
recursively.

Flags as unsafe:
- `:call` nodes where `args[1]` is a Symbol not in `allowed_calls`
- `:call` nodes where `args[1]` is a qualified name (e.g., `Base.run`)
- `:macrocall` nodes
- `:quote` or `:\$` interpolation nodes

Does NOT flag: assignment, block, if, while, for, literal values, variable symbols.
"""
function sanitize(san::ASTSanitizer, expr::Expr)::Bool
    # Reject macros, quotes, and interpolation
    if expr.head in (:macrocall, :quote, :$)
        return false
    end

    if expr.head == :call
        fn = expr.args[1]
        # Reject qualified calls (e.g., Base.run, Sys.exit)
        if fn isa Expr && fn.head == :.
            return false
        end
        # Check against whitelist
        if fn isa Symbol && fn ∉ san.allowed_calls
            return false
        end
    end

    # Recursively check all Expr arguments
    for arg in expr.args
        if arg isa Expr
            sanitize(san, arg) || return false
        end
    end

    return true
end

"""
    sanitize(san::ASTSanitizer, body::Vector{Expr}) -> Bool

Check all statements in a genome body.
"""
function sanitize(san::ASTSanitizer, body::Vector{Expr})::Bool
    for stmt in body
        sanitize(san, stmt) || return false
    end
    return true
end
