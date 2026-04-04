# Investigation 08: JSON Escaping and Un-escaping Completeness

## What Was Tested

Formal analysis of `_json_escape` (llm_operator.jl line 215-222) and `_find_last_json_string` (lines 235-250) against RFC 8259 JSON string requirements.

## Key Findings

### _json_escape — missing control character escapes

RFC 8259 requires escaping of ALL control characters U+0000 through U+001F. The function handles:
- `\\` → `\\\\` ✓
- `"` → `\\"` ✓
- `\n` (U+000A) → `\\n` ✓
- `\r` (U+000D) → `\\r` ✓
- `\t` (U+0009) → `\\t` ✓

Missing:
- `\b` (U+0008, backspace)
- `\f` (U+000C, form feed)
- U+0000-U+0007, U+000B, U+000E-U+001F (all other control characters)

### Practical impact: None for current usage

- The `system_prompt` is a hardcoded constant string containing only ASCII printable characters and `\n`. No control chars.
- The `source` is produced by `serialize(g)` which calls `repr()` on Julia Expr objects. `repr()` produces ASCII-only output with escaped special characters. No raw control chars.
- An LLM model name (e.g., "claude-sonnet-4-20250514") contains no control chars.

So the missing escapes are a theoretical RFC 8259 compliance gap, not a practical bug.

### _find_last_json_string — missing JSON unescapes

The function unescapes:
- `\\n` → `\n` ✓
- `\\t` → `\t` ✓
- `\\"` → `"` ✓
- `\\\\` → `\\` ✓

Missing:
- `\\b` → backspace
- `\\f` → form feed
- `\\uXXXX` → Unicode code point
- `\\/` → `/` (optional per RFC but common)

### Practical impact: Negligible

LLM-generated Julia code will not contain `\b`, `\f`, or Unicode escapes in the `text`/`content` field. Julia source code uses ASCII identifiers. The missing unescapes could theoretically cause a parse failure if an LLM response contained these sequences, but this would trigger the fallback (return nothing from deserialize).

### Un-escape order issue

The current order is:
1. `\\n` → `\n`
2. `\\t` → `\t`
3. `\\"` → `"`
4. `\\\\` → `\\`

This is INCORRECT order. The `\\\\` → `\\` replacement should happen FIRST, before other escapes. Otherwise, the literal sequence `\\n` in the JSON (meaning a backslash followed by 'n') would be incorrectly unescaped to a newline.

Example: JSON contains `"\\\\n"` (which represents the string `\n` literally, not a newline).
- Step 1: `\\n` → `\n` — converts `\\n` to a newline (WRONG, this was an escaped backslash followed by 'n')
- Step 4: `\\\\` → `\\` — no remaining `\\\\` to replace

Correct order would unescape `\\\\` first, producing `\n` as a literal string, then skip the `\\n` replacement because the `\` is no longer preceded by another `\`.

**However:** The regex pattern `((?:[^\"\\\\]|\\\\.)*)` already handles this at the extraction level — it matches escaped sequences as atomic units. The issue is only in the post-extraction un-escaping, where the chained `replace` calls can interfere. In practice, LLM code output rarely contains literal backslash-n sequences, so this is an edge case.

## Verdict

**CONFIRMED (minor):**
1. Missing control character escapes in `_json_escape` — RFC non-compliance, but no practical impact
2. Missing JSON unescapes in `_find_last_json_string` — negligible practical impact
3. Incorrect unescape order — `\\\\` should be unescaped before `\\n`, `\\t`, `\\"` to avoid double-unescaping

**Severity: Low.** All three issues are edge cases that don't affect the current usage pattern (LLM-generated Julia code). The unescape order issue is the most likely to manifest in real usage.
