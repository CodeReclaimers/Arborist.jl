# Iteration 003 — Investigation Plan

## Gemini Finding Classification

| # | Title | Classification | Rationale |
|---|-------|---------------|-----------|
| 1 | Incomplete JSON string escaping | ACTIONABLE | Testable: check if control chars U+0000-U+001F beyond \n\r\t are escaped |
| 2 | Uncaught exceptions in LLM operator | ACTIONABLE | Testable: trace whether deserialize or sanitize can throw exceptions that bypass the fallback |
| 3 | Non-deterministic Dict iteration in GraphGenome | ACTIONABLE (partially pre-confirmed) | Already identified as follow-up in iteration 002. Need to enumerate all affected sites. |
| 4 | Sanitizer bypassed with infinite loop | KNOWN | The code already uses `add_loop_checks` to instrument all loops with iteration counters and `LoopLimitExceeded` exceptions. The sanitizer doesn't need to block loops because the evaluation harness already limits them. |
| 5 | Incomplete JSON response un-escaping | ACTIONABLE | Minor but real — \b, \f, \uXXXX not unescaped. In practice, LLMs rarely produce these in code, but it's a correctness gap. |
| 6 | Fitness sharing formula claim misleading | ACTIONABLE | Need to re-read the full claim in context — the paper observations explain WHY linear sharing hurts in the bin packing scenario with ~100 singleton species. |

## Investigations

### Investigation 06: LLM operator exception handling (Finding 2)

**Hypothesis:** `deserialize` or `sanitize` could throw an exception that isn't caught, violating the graceful degradation claim.

**Mode:** Formal analysis — trace all exception paths in deserialize and sanitize.

### Investigation 07: GraphGenome non-deterministic Dict iteration (Finding 3)

**Hypothesis:** Multiple GraphGenome mutation functions iterate over Dict values/keys without sorting, violating the deterministic-ordering convention.

**Mode:** Code search — enumerate all affected sites.

### Investigation 08: JSON escaping completeness (Finding 1 + 5)

**Hypothesis:** `_json_escape` doesn't handle all RFC 8259 required escapes; `_find_last_json_string` doesn't unescape all JSON escapes.

**Mode:** Formal analysis against RFC 8259.
