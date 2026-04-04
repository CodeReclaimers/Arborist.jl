# Investigation 06: Uncaught Exception Paths in LLM Mutation Operator

## Claim Under Test

"All LLM failures fall back to a classical operator" -- the `mutate` function in `llm_operator.jl` (lines 135-195) claims to be robust to any LLM failure mode.

## Summary of Findings

The "graceful degradation" claim is **almost entirely correct** but has **one confirmed gap** (StackOverflowError) and **one theoretical gap** (malformed non-Expr parse results). Neither is likely to occur in practice, but both violate the stated contract.

---

## Path 1: `deserialize(ExprGenome, text, g.state)` at line 181

### `_parse_statements(s)` (expr_genome.jl lines 176-202)

Both `Meta.parse` calls are wrapped in try/catch. The catch blocks re-throw `InterruptException` and return `nothing` for everything else. A `nothing` result is filtered out (line 199: `expr === nothing || push!(stmts, expr)`).

**Verdict: Safe.** Cannot throw an unhandled exception (excluding InterruptException, which should propagate).

### `_is_valid_statement` and its callees (expr_genome.jl lines 211-348)

- `_is_valid_assignment` (line 230): Calls `get_lvalue_type` and `get_rvalue_type` inside a try/catch that returns `false` on any non-InterruptException error. **Safe.**
- `_is_valid_while` (line 246): Calls `get_rvalue_type` inside try/catch. **Safe.**
- `_is_valid_if` (line 261): Calls `get_rvalue_type` inside try/catch. **Safe.**
- `_is_valid_for` (line 291): Does NOT call `get_rvalue_type` or `get_lvalue_type` -- only checks structural properties. **Safe.**
- `_is_valid_block` (line 303): Iterates `expr.args`, delegates to `_is_valid_statement`. **Safe.**
- `_is_valid_call` (line 316): Checks `fn isa Symbol` and iterates `state.funcs.funcs`. **Safe.**

### The `get_lvalue_type` and `get_rvalue_type` throw paths (codegen.jl)

These functions CAN and DO throw:
- `get_lvalue_type(s, value)` at line 122: raw Dict lookup `s.all_vars[value]` -- throws `KeyError` for unknown variables.
- `get_rvalue_type(s, value)` at line 171 (fallback): throws `ArgumentError` for unexpected value types.
- `get_rvalue_type(s, value::Symbol)` at line 180: raw Dict lookup `s.all_vars[value]` -- throws `KeyError`.
- `get_rvalue_type(s, value::Expr)` at lines 188, 194: throws `ErrorException` or `ArgumentError`.

**BUT** every call site in the `_is_valid_*` family wraps these in try/catch. So these throws are always caught.

### Potential gap: Non-Expr parse results

`_parse_statements` can return non-Expr elements. The loop in `deserialize` (lines 145-153) checks `expr isa Expr` before calling `_is_valid_statement`, so non-Expr items (Symbols, Numbers, etc.) are silently skipped.

However, lines 147-149 do `expr.head` and `expr.args[1]` on the outer `expr` BEFORE the `expr isa Expr` guard on line 150. The QuoteNode unwrap at line 147 is protected by `expr isa Expr &&` short-circuit. **Safe.**

### Empty/whitespace input to `deserialize`

If `_extract_response_text` returns `""` or `"   \n  "`:
- `_parse_statements("")` wraps it as `"begin\n\nend"`, which parses to an empty block. Returns `[]`.
- The `for expr in stmts` loop does nothing. `valid_stmts` stays empty.
- Line 154: `isempty(valid_stmts) && return nothing`.
- Back in `mutate`, line 183 catches the `nothing` and falls back.

**Verdict: Safe.** Empty/whitespace input is handled gracefully.

---

## Path 2: `sanitize(ASTSanitizer(), result.body)` at line 189

### Can `sanitize` throw on valid Expr nodes?

The function accesses `expr.head`, `expr.args`, and does `fn isa Symbol` and set membership checks. For any well-formed `Expr` object (which Julia's `Meta.parse` always produces), these are safe field accesses that cannot throw.

### Can `sanitize` throw on malformed Expr nodes?

`deserialize` only passes expressions through `_is_valid_statement`, which checks structural properties (head values, args counts). All expressions reaching `sanitize` have been validated to have correct `.head` values and appropriate `.args` lengths.

However, `sanitize` recurses into ALL `expr.args` children (line 101-104), not just the ones checked by `_is_valid_statement`. If a deeply nested sub-expression has an unexpected structure, the field accesses are still safe because Julia's `Expr` type guarantees `.head` and `.args` exist.

### CONFIRMED GAP: StackOverflowError from deeply nested expressions

`sanitize` recurses via:
```julia
for arg in expr.args
    if arg isa Expr
        sanitize(san, arg) || return false
    end
end
```

There is no depth limit. An LLM could return a response that parses into a deeply nested expression tree (e.g., `(((((((...)))))))`), causing `StackOverflowError`. This error is NOT caught by the try/catch in `mutate` because **there is no try/catch around the `sanitize` call at line 189**.

The same StackOverflowError risk exists in `_is_valid_statement` -> `_is_valid_body` -> `_is_valid_statement` recursion, and this path is ALSO not wrapped in try/catch at line 181 (the `deserialize` call).

`StackOverflowError` is a subtype of `Exception`, so a `try/catch` would catch it. But neither call site has one.

**Severity: Low.** Julia's `Meta.parse` has its own stack depth limits and would need an astronomically deep nesting to overflow the runtime stack. In practice, LLM-generated code rarely exceeds 10-20 levels of nesting. But the theoretical violation of the "never throws" contract exists.

### Can `sanitize` receive a non-Expr in `body`?

`sanitize(san::ASTSanitizer, body::Vector{Expr})` at line 115 is type-constrained to `Vector{Expr}`. Since `deserialize` returns `ExprGenome(valid_stmts, state)` where `valid_stmts::Vector{Expr}`, and `result.body` is typed as `Vector{Expr}`, this is guaranteed safe by Julia's type system.

---

## Path 3: `serialize(g)` at line 138

`serialize` calls `repr(stmt)` on each `Expr` in `g.body`. Julia's `repr` for `Expr` objects is a built-in that handles all valid Expr trees without throwing.

### Empty body case

If `g.body` is empty, the loop body never executes, and `String(take!(io))` returns `""`. This empty string flows to the LLM prompt, which would return some response, which would likely fail to deserialize, triggering the fallback at line 183-185.

**Verdict: Safe.** The empty body case degrades gracefully through the normal fallback path.

---

## Path 4: Line 189 -- Missing try/catch

The critical finding: lines 181 and 189 are NOT wrapped in try/catch.

```julia
# Line 181 - no try/catch
result = deserialize(ExprGenome, text, g.state)

# Line 189 - no try/catch
if !sanitize(ASTSanitizer(), result.body)
```

Contrast with line 165-170, where the HTTP call IS wrapped:
```julia
response_text = try
    _http_post[](op.endpoint, headers, body, op.timeout_seconds)
catch e
    ...
end
```

### What could escape from `deserialize`?

After tracing every code path:
- `_parse_statements`: Fully try/catch protected. **Cannot throw.**
- `_is_valid_statement` tree: All `get_*_type` calls wrapped in try/catch. **Cannot throw** (except StackOverflowError on deeply nested input, or InterruptException).
- The loop and `isempty` check: Standard Julia operations on arrays. **Cannot throw.**

**Verdict: `deserialize` is effectively safe**, but the safety depends on the internal try/catch blocks in `_is_valid_assignment`, `_is_valid_while`, and `_is_valid_if` catching everything. If any of those internal try/catch blocks were refactored to be more selective (e.g., catching only `KeyError`), the outer call would become vulnerable.

### What could escape from `sanitize`?

- `StackOverflowError` from deep recursion (confirmed gap, low probability).
- No other identified throw paths for well-formed `Expr` objects.

---

## Severity Assessment

| Gap | Could it crash the evolutionary loop? | Probability in practice | Fix difficulty |
|-----|--------------------------------------|------------------------|----------------|
| StackOverflowError in `sanitize` | Yes | Very low | Easy -- wrap lines 181 and 189 in try/catch |
| StackOverflowError in `deserialize` recursion | Yes | Very low | Same fix |

## Recommended Fix

Wrap the deserialize + sanitize block (lines 181-192) in a single try/catch that falls back to the classical operator on any exception. This would make the "graceful degradation" claim unconditionally true and protect against future refactoring of the internal validation functions:

```julia
# In mutate() at line 180, replace lines 181-192 with:
result, safe = try
    r = deserialize(ExprGenome, text, g.state)
    if r === nothing
        (nothing, false)
    else
        (r, sanitize(ASTSanitizer(), r.body))
    end
catch e
    e isa InterruptException && rethrow()
    @warn "LLMMutationOperator: deserialize/sanitize failed" exception=e
    (nothing, false)
end

if result === nothing || !safe
    return mutate(op.fallback_op, g, rng)
end

return result
```

## Conclusion

The graceful degradation claim is **substantively correct but not formally guaranteed**. The internal structure of `deserialize` and `sanitize` happens to not throw for any input that `Meta.parse` can produce under normal conditions. However, the absence of a try/catch around these calls means:

1. A `StackOverflowError` from adversarially deep nesting would crash the loop.
2. Future refactoring of the internal validation functions could introduce throw paths that propagate unhandled.

The fix is trivial (one try/catch block) and would make the contract airtight.
