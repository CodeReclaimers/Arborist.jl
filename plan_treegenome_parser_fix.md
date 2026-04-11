# Plan: Fix TreeGenome serialize/deserialize round-trip

## Context

`serialize(::TreeGenome)` and `deserialize(::Type{TreeGenome{T}}, ...)` in
`src/tree_genome.jl` disagree about the string format they exchange, so
any code path that round-trips a tree through a string silently loses
information for binary operators.

- **`serialize` at `src/tree_genome.jl:189-191`** delegates to
  DynamicExpressions' `string_tree`, which emits **real
  Julia-parseable infix** — for example, `(x1 + 1.0) * sin(x2)` or
  `(x1 + 1.0) / x1`.
- **`deserialize` at `src/tree_genome.jl:193-198`** calls the
  hand-rolled tokenizer `_parse_prefix_expr` (`src/tree_genome.jl:531-646`),
  which only recognizes **prefix s-expressions** — `+(x1, 1.0)`,
  `sin(*(x1, x2))`. When handed `x1 + 1.0` the tokenizer matches the
  feature-variable branch on the first token, returns a bare
  `Node(; feature=1)`, and silently drops `+ 1.0`. Unary operators
  happen to round-trip because `sin(x1)` is textually identical in
  both notations.

**Evidence:**
- `CLAUDE.md` "Known Limitations" section already lists this:
  "TreeGenome serialize/deserialize format mismatch: `serialize`
  outputs infix (`x1 + 1.0`), `deserialize` parses prefix
  (`+(x1, 1.0)`). Unary ops round-trip; binary ops do not."
- The existing test at
  `test/integration/test_tree_genome.jl:98-135` documents the bug in
  its own comment block (lines 101-104: "So serialize -> deserialize
  is not a lossless round-trip for binary ops. Test that deserialize
  works with prefix format strings.") and only exercises the prefix
  form — effectively locking the bug in as "working as written."

**Who is affected today:**
1. **`LLMMutationOperator` is unusable with TreeGenome** end-to-end.
   The operator currently only dispatches on `ExprGenome` at
   `src/llm_operator.jl:153`, but even if it were extended to
   `TreeGenome` it would fail the first time the LLM returned a
   well-formed infix expression — `deserialize` would silently
   return `nothing` and the evolution loop would fall back to
   classical mutation on every LLM call.
2. **Any direct caller of `serialize`/`deserialize`** for
   TreeGenome: users inspecting evolved trees, writing custom
   logging, or implementing their own migration backend would hit
   the same silent drop.
3. **Not `IslandModel`+`TreeGenome` migration** — the session of
   2026-04-11 deliberately routed migration through
   `to_migrant(::TreeGenome)` / `from_migrant(::TreeGenomeContext)`
   which pack `Node{T}` directly into `MigrantGenome.data` and never
   touch the string path. That's why the bug was visible but not
   blocking for the island feature.

**Intended outcome:**
`serialize` → `deserialize` is a lossless round-trip for any tree
DynamicExpressions can evaluate. A call like
`serialize(g)` followed by `deserialize(TreeGenome{Float32}, s, ops, n_features)`
returns a `TreeGenome` whose predictions on any matrix are
bit-identical to the source. The fix is backward-compatible with the
old prefix form — existing tests continue to pass without
modification — because `Meta.parse` normalizes both forms to the
same `Expr(:call, op, args...)`.

## Goal

Replace the hand-rolled prefix tokenizer in `src/tree_genome.jl` with
an `Expr`-walker built on `Meta.parse`. Update the docstring of
`deserialize` to reflect the new accepted input range. Add
round-trip tests that `string_tree` output parses correctly.
Preserve every existing test at
`test/integration/test_tree_genome.jl:98-135` unchanged.

## Out of scope — deferred follow-on

**`LLMMutationOperator` dispatch on `TreeGenome` is NOT part of this
plan.** That is a separate ~1.5 hours of work consisting of:
- Adding `mutate(::LLMMutationOperator, ::TreeGenome{T}, rng)` to
  `src/llm_operator.jl` (currently only
  `mutate(::LLMMutationOperator, ::ExprGenome, ...)` at `:153`).
- Serializing the TreeGenome via `serialize` (works today), prompting,
  parsing the reply via the fixed `deserialize` (works after this
  plan), and falling back to `SubtreeMutation`/`_point_mutate` on
  any failure, matching the ExprGenome pattern.
- Adding an integration test with mocked HTTP in
  `test/integration/test_llm_operator.jl`.

Do not scope-creep into it. The parser fix is independently useful
and the LLM enablement sits on top cleanly.

**Also out of scope:** reverting `IslandModel`+`TreeGenome` migration
to the string path. `Node{T}`-direct is strictly better for
migration (no parse ambiguity, exact bit-preservation, no
dependency on DynamicExpressions' output format) and should stay.

## Critical files

| File | Change |
|---|---|
| `src/tree_genome.jl` | Replace `_parse_prefix_expr` / `_tokenize_prefix` / `_parse_prefix_token` (lines 521-646) with a single `_expr_to_node` helper. Update `deserialize` at `:193-198` to call `Meta.parse` and dispatch through the walker. Update docstring. |
| `test/integration/test_tree_genome.jl` | Update the stale comment at `:101-104`, add round-trip tests that `serialize → deserialize → eval` matches the source predictions on a data matrix. Existing prefix-form tests at `:107-134` stay as-is. |
| `src/llm_operator.jl` | **Not modified in this plan.** Deferred to follow-on. |

## Design

### Core: replace the tokenizer with a Meta.parse-based walker

The new `deserialize` is five lines plus a recursive walker:

```julia
function deserialize(::Type{TreeGenome{T}}, s::AbstractString,
                     operators::OperatorEnum, n_features::Int) where T
    stripped = strip(s)
    isempty(stripped) && return nothing
    expr = try
        Meta.parse(stripped)
    catch
        return nothing
    end
    tree = _expr_to_node(expr, operators, n_features, T)
    tree === nothing && return nothing
    return TreeGenome{T}(tree, operators, n_features)
end
```

Note the signature widens from `s::String` to `s::AbstractString` —
this is intentional. It fixes a latent issue flagged in
`progress-20260327.md:123` where `strip()` returned `SubString{String}`
and caused a `MethodError` that was silently swallowed by the
try/catch, so every `deserialize` call returned `nothing`. That bug
was worked around by stripping earlier in the call chain; folding
the fix in here eliminates the workaround and the footgun.

### The walker

```julia
"""
    _expr_to_node(x, operators, n_features, T) -> Union{Node{T}, Nothing}

Walk a `Meta.parse`d expression and build a DynamicExpressions
`Node{T}`. Accepts both infix (`x1 + 1.0`, what `string_tree` emits)
and prefix (`+(x1, 1.0)`, what the old deserializer required) because
`Meta.parse` normalizes both forms to the same `Expr(:call, ...)`
structure. Returns `nothing` on any unrecognized construct.
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
```

**Reuses existing helpers:**
- `_get_unary_ops(::OperatorEnum)` at `src/tree_genome.jl:92-94`
- `_get_binary_ops(::OperatorEnum)` at `src/tree_genome.jl:95-97`
- The dual-symbol match pattern `Symbol(bop) == op_sym || Symbol(nameof(bop)) == op_sym`
  from the existing `_parse_prefix_token` at `:602-603`.
- `Node{T}(; val=...)`, `Node{T}(; feature=...)`, `Node{T}(; op=..., l=..., r=...)` — same constructors the current parser uses.

**Deleted code:** lines 521-646 in `src/tree_genome.jl`
(`_parse_prefix_expr`, `_tokenize_prefix`, `_parse_prefix_token`, and
their surrounding comments). The new `_expr_to_node` is ~50 lines
and replaces ~125 lines of tokenizer. Net reduction.

## Edge cases

| Case | Handling |
|---|---|
| Integer literals from `Meta.parse` (`x1 + 2` → `Expr(:call, :+, :x1, 2)`) | `T(x)` widens in the `Number` branch. Existing parser already does this. |
| Scientific notation (`1.5e-3`) | `Meta.parse` handles natively, lands as `Float64` in the `Number` branch, `T(x)` widens. |
| Negative literals (`-1.5`) | `Meta.parse("-1.5")` produces the bare negative number, not an `Expr(:call, :-, ...)`. Lands in the `Number` branch directly. Confirmed. |
| Unary minus on a variable (`-x1`) | `Meta.parse("-x1")` → `Expr(:call, :-, :x1)` — an arity-1 call. Only resolves if `-` is in the enum's **unary** operator list. Otherwise returns `nothing`. This is strictly safer than the current tokenizer which would silently produce garbage. |
| Binary subtraction (`x1 - 1.0`) | `Expr(:call, :-, :x1, 1.0)` — arity-2, goes through the binary branch, matches `-` against the binary ops. |
| Operator not in the enum | Falls through all the `for` loops, returns `nothing`. Deserialize returns `nothing`. Caller falls back. |
| Empty / whitespace-only input | Handled at the `deserialize` entry point via `isempty(stripped)`. |
| Non-`:call` Expr heads (`:block`, `:tuple`, ...) | Rejected by the `x.head === :call` check. Returns `nothing`. |
| `Meta.parse` throws (malformed Julia) | Caught by the try/catch in `deserialize`. Returns `nothing`. |

## Tests

### Existing tests — keep unchanged

`test/integration/test_tree_genome.jl:107-134` exercises the old
prefix form (`+(x1, 1.0)`, `sin(*(x1, x2))`, `x1`, `3.14`, and
negative cases). These pass unchanged after the fix because
`Meta.parse` normalizes both forms. Do **not** delete them — they
become the regression guard that the prefix form still works.

### Update the comment block

The comment at `test/integration/test_tree_genome.jl:101-104`
currently documents the bug. Replace it with a comment explaining
that `deserialize` now accepts both infix (from `serialize` /
`string_tree`) and prefix (from older code paths) forms via
`Meta.parse`.

### Add round-trip tests

Append a new testset (either within the existing
`@testset "TreeGenome serialize/deserialize round-trip"` or as a
sibling) that actually tests what the name claims:

```julia
@testset "TreeGenome serialize -> deserialize round-trip via string_tree" begin
    ops = OperatorEnum(; binary_operators=[+, -, *, /],
                        unary_operators=[sin, cos, abs])
    X = reshape(Float32.(range(-2, 2, length=25)), 1, :)

    # Build a collection of trees exercising every operator arity
    # and a few nested structures.
    function round_trip_eq(g::TreeGenome{Float32})
        s = serialize(g)
        g2 = deserialize(TreeGenome{Float32}, s, g.operators, g.n_features)
        g2 === nothing && return false
        p1 = g.tree(X, g.operators)
        p2 = g2.tree(X, g2.operators)
        return p1 == p2
    end

    # Single terminal
    @test round_trip_eq(TreeGenome{Float32}(Node{Float32}(; feature=1), ops, 1))
    @test round_trip_eq(TreeGenome{Float32}(Node{Float32}(; val=2.5f0), ops, 1))

    # Binary (previously broken)
    let
        t = Node{Float32}(; op=1, l=Node{Float32}(; feature=1),
                             r=Node{Float32}(; val=1.0f0))  # x1 + 1
        @test round_trip_eq(TreeGenome{Float32}(t, ops, 1))
    end

    # All four binary ops in isolation
    for bi in 1:4
        t = Node{Float32}(; op=UInt8(bi),
                             l=Node{Float32}(; feature=1),
                             r=Node{Float32}(; val=2.0f0))
        @test round_trip_eq(TreeGenome{Float32}(t, ops, 1))
    end

    # All three unary ops in isolation
    for ui in 1:3
        t = Node{Float32}(; op=UInt8(ui), l=Node{Float32}(; feature=1))
        @test round_trip_eq(TreeGenome{Float32}(t, ops, 1))
    end

    # Nested: sin((x1 + 1) * x1)
    let
        inner = Node{Float32}(; op=1, l=Node{Float32}(; feature=1),
                                 r=Node{Float32}(; val=1.0f0))
        prod  = Node{Float32}(; op=3, l=inner, r=Node{Float32}(; feature=1))
        top   = Node{Float32}(; op=1, l=prod)  # sin(...)
        @test round_trip_eq(TreeGenome{Float32}(top, ops, 1))
    end

    # Deep random tree via _random_tree
    rng = Random.MersenneTwister(42)
    for _ in 1:10
        tree = Arborist._random_tree(rng, ops, 1, Float32, 4, :grow)
        g = TreeGenome{Float32}(tree, ops, 1)
        @test round_trip_eq(g)
    end
end
```

**Why `p1 == p2` and not `p1 ≈ p2`:** the reconstructed tree uses
the same op indices as the source against the same `OperatorEnum`,
so evaluation is deterministic and bit-identical. If `≈` is needed
(e.g. because an integer-literal path introduces a rounding step),
that's a sign something is wrong.

### Do not assert golden string output

Avoid tests of the form `@test serialize(g) == "x1 + 1.0"`. The
string format is DynamicExpressions' responsibility and can change
between minor versions. Test the semantic round-trip (predictions
match), not the syntactic one.

## Verification

1. **Fast tier:**
   ```
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
   Expect **3143 + new tests pass / 1 broken** (pre-existing
   `@test_skip` for `ANTHROPIC_API_KEY`). Wall time ~1m10s + a few
   seconds for the new round-trip asserts.

2. **Existing prefix-form tests still pass:** the specific testset
   `"TreeGenome serialize/deserialize round-trip"` at
   `test/integration/test_tree_genome.jl:98` must stay green.
   This is the backward-compatibility check — if it fails, the
   `Meta.parse` normalization assumption is wrong.

3. **LLM-operator fallback tests still pass:** several tests in
   `test/integration/test_llm_operator.jl` stub the HTTP layer and
   assert that `deserialize` returns `nothing` on malformed input
   (logged as `"LLMMutationOperator: deserialize returned nothing,
   falling back"`). The new parser must continue to return
   `nothing` on genuinely unparseable input — confirm these warning
   counts are unchanged.

4. **Full benchmark tier:**
   ```
   ARBORIST_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'
   ```
   Expect **3165 + new tests pass / 1 broken**. The symbolic
   regression benchmarks (`koza_regression.jl`,
   `symbolic_regression.jl`) do not currently round-trip trees
   through strings, so no behavior change is expected there — but
   run the full tier anyway to confirm no hidden coupling.

5. **End-to-end smoke:** a one-off script that builds several
   `TreeGenome{Float32}` values using `_random_tree`, round-trips
   each through `serialize → deserialize`, and asserts the
   reconstructed tree evaluates identically on a test matrix.
   Include at least one tree per binary operator in the default
   set (+, -, *, /) and one per unary operator (sin, cos, abs).

## Risks and gotchas

- **DynamicExpressions version coupling.** `string_tree` output
  format is owned by DynamicExpressions and can change across
  minor versions. `Project.toml` pins `DynamicExpressions = "2"`
  and the installed version is 2.5.2. Mitigation: semantic
  round-trip tests (predictions match), not golden-string tests.
  If a future DynExpr release adds unusual formatting (e.g. extra
  parenthesization, keyword args in calls), `Meta.parse` still
  handles it as long as the output is valid Julia.

- **Symbol equality for function objects.** The existing parser
  uses `Symbol(bop) == op_sym || Symbol(nameof(bop)) == op_sym` at
  `:602-603` to tolerate two different ways Julia might stringify
  the same operator. Preserve this pattern exactly — dropping
  either clause breaks a corner case.

- **`strip` → `SubString`.** Widening the input type to
  `AbstractString` fixes the latent issue from
  `progress-20260327.md:123` where a `::String` annotation caused
  a silent `MethodError` every time `strip` was called on the
  input. Check for any remaining `::String` annotations in the
  call chain and widen them too.

- **`catch` swallowing `InterruptException`.** The existing
  `deserialize` catches all exceptions. Per the fix from Tier 2
  of the 2026-03-25 pre-registration pass
  (`progress-20260325.md:208`), new catch blocks should use
  `catch e; e isa InterruptException && rethrow()`. Apply the same
  pattern here.

- **Test name already says "round-trip".** The existing testset
  name at `test_tree_genome.jl:98` is "TreeGenome
  serialize/deserialize round-trip" but currently only tests the
  deserialize half. After the fix the name becomes accurate;
  either rename the existing testset to "deserialize accepts
  prefix form" and add a new "round-trip" testset, or fold the
  new round-trip asserts into the existing testset. Prefer the
  fold-in for minimal churn.

## Follow-on: LLM-TreeGenome enablement (separate session)

After the parser fix lands, the natural next step is to enable
`LLMMutationOperator` for `TreeGenome`. This is its own ~1.5 hours
of work and should not be bundled in.

Sketch for the follow-on session:
1. Add `mutate(op::LLMMutationOperator, g::TreeGenome{T}, rng::AbstractRNG)`
   in `src/llm_operator.jl`, paralleling the existing
   `mutate(::LLMMutationOperator, ::ExprGenome, ...)` at `:153`.
   Serialize via `serialize(g)` (works today, emits infix).
   Deserialize via `deserialize(TreeGenome{T}, text, g.operators, g.n_features)`
   (works after this plan).
2. Fallback operator: `SubtreeMutation()` works via the existing
   TreeGenome operator-dispatch at `src/tree_genome.jl:202-214`.
3. Sanitizer: the AST sanitizer at `src/sanitizer.jl` targets Julia
   `Expr` trees, not DynamicExpressions `Node{T}`. The parser
   already restricts accepted expressions to a whitelist of
   operators present in the enum — no external sanitizer call is
   needed. Document this decision in the new method's docstring.
4. Update `docs/src/llm_operator.md` to mention TreeGenome support.
5. Add an integration test with mocked HTTP that feeds a valid
   infix expression back through the pipeline and asserts the
   evolved genome has the expected tree structure. The mock
   infrastructure in `test/mocks/mock_http.jl` already supports
   this pattern.
6. Verify end-to-end against a live Ollama endpoint (gated behind
   an environment variable, matching the existing
   `ANTHROPIC_API_KEY` gate).

**Related deferred items** that do NOT need to be bundled with the
LLM enablement:
- Extending `LLMMutationOperator` to `AntGenome`/`GraphGenome` —
  blocked by the same upstream limitations as IslandModel.
- Token-usage tracking (`TODO` from
  `progress-20260410.md:121-143`) — orthogonal.
