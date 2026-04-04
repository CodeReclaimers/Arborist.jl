# Iteration 003 — Synthesis

## Confirmed Findings

### 1. GraphGenome non-deterministic Dict iteration — 6 mutation/crossover sites (Finding 3 → inv_07: CONFIRMED)

Six functions in `graph_genome.jl` iterate over Dict keys/values with RNG consumption in non-deterministic order:
- `_mutate_weights!` (line 220)
- `_mutate_weight_replace!` (line 228)
- `_mutate_add_connection!` (line 235)
- `_mutate_add_node!` (line 263)
- `_mutate_toggle_connection!` (line 284)
- `_neat_crossover` (line 302 — iterates a Set from `union`)

This violates the project's documented convention: "Collection sampling uses deterministic sorted ordering." It means GraphGenome evolution is NOT reproducible from a fixed seed across different Julia sessions (Dict hash seeds differ). The `_topological_sort` and `evaluate_genome` functions ARE correctly deterministic.

**Severity:** High for reproducibility. Every GraphGenome generation compounds the non-determinism.

### 2. JSON unescape order bug (Finding 1 + 5 → inv_08: CONFIRMED, minor)

`_find_last_json_string` unescapes `\\\\` AFTER `\\n`, `\\t`, `\\"`. This means the literal string `\n` (backslash followed by 'n') in a JSON response would be incorrectly unescaped to a newline. The correct order is to unescape `\\\\` first.

Additionally, `_json_escape` doesn't handle `\b`, `\f`, or other control characters per RFC 8259. `_find_last_json_string` doesn't unescape `\b`, `\f`, `\uXXXX`.

**Severity:** Low. In practice, LLM-generated Julia code won't contain these sequences. The unescape order bug is the most likely to matter.

### 3. LLM operator missing try/catch around deserialize+sanitize (Finding 2 → inv_06: CONFIRMED, low probability)

The `deserialize` and `sanitize` calls in `mutate(::LLMMutationOperator, ...)` are not wrapped in try/catch. While `deserialize`'s internals are effectively safe (all sub-calls have their own try/catch), a `StackOverflowError` from deeply nested LLM output could crash the loop. The absence of the outer try/catch also makes the graceful degradation claim fragile against future refactoring.

**Severity:** Low (StackOverflowError from LLM output is extremely unlikely). The fix is trivial — one try/catch block.

## Not Investigated / Classified as Known

### 4. Sanitizer doesn't block infinite loops (Finding 4: KNOWN)

The `add_loop_checks` function in codegen.jl already instruments all loops with iteration counters and `LoopLimitExceeded` exceptions (default limit: 10,000). The sanitizer doesn't need to block loop constructs because the evaluation harness already prevents infinite loops.

### 5. Fitness sharing claim is accurate (Finding 6: REFUTED)

Gemini claimed the paper's description of `:linear` sharing inverting selection pressure was misleading. However, the paper is correct: with minimization (lower = better), `shared = raw * species_size` means a species of 5 at fitness 1.10 gets shared fitness 5.5, while a singleton at fitness 2.0 stays at 2.0. The singleton (worse raw fitness) wins in tournament selection because 2.0 < 5.5. This IS inversion — it penalizes membership in good species.

## Paper Observations Accuracy Check

| Claim | Status |
|-------|--------|
| 1.5: All LLM failures fall back | Substantively correct but not formally guaranteed (missing try/catch) |
| 1.6: Deterministic sorted ordering for collections | VIOLATED in GraphGenome (6 sites) |
| 1.6: Explicit rng everywhere | Partially violated (see iteration 001 re: GenState.rng contamination in ExprGenome) |
| 2.5: Sharing formula inverts selection for singletons | Correct as stated |
| 3.1: Three sharing formulas implemented | Correct — `:linear`, `:sqrt`, `:log2` all present in speciation.jl |

## Follow-up Candidates

1. **Distributed island model review** — `distributed_island.jl` not yet audited
2. **The `defaults.jl` function sets** — Do the default function sets match what's documented?
3. **TreeGenome `_replace_nth_node!` correctness** — The pre-order index counting logic is complex and was flagged but not deeply tested
