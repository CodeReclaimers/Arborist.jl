# Iteration 003 — Document Updates

Per RESEARCH.md scope: **No project files were modified.**

## Findings to record when approved:

1. **GraphGenome non-deterministic Dict iteration (6 sites)** — `_mutate_weights!`, `_mutate_weight_replace!`, `_mutate_add_connection!`, `_mutate_add_node!`, `_mutate_toggle_connection!`, `_neat_crossover` all iterate Dict values/keys/sets with RNG consumption in non-deterministic order. Violates the documented determinism convention. Fix: sort before iteration.

2. **JSON unescape order bug** — `_find_last_json_string` unescapes `\\\\` after `\\n`/`\\t`/`\\"`, causing double-unescaping of literal backslash sequences. Fix: move `\\\\` unescape to first position.

3. **LLM operator missing try/catch** — `deserialize` and `sanitize` calls not wrapped in try/catch. Formally violates graceful degradation claim despite being practically safe. Fix: one try/catch block around lines 181-192.

4. **Paper observations claim 1.6 accuracy** — "Collection sampling uses deterministic sorted ordering" is violated by GraphGenome. Either fix the code or amend the claim to specify which genome types guarantee determinism.
