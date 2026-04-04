# Iteration 002 — Synthesis

## Confirmed Findings

### 1. GraphGenome crossover produces identical children when parents are fully disjoint (Finding 1 + Finding 5 → inv_03: CONFIRMED)

`crossover(g1, g2, rng)` calls `_neat_crossover(fitter, other, rng)` twice with the same parent order. When parents have no matching innovation numbers, both calls are deterministic and produce identical children — both are copies of the fitter parent's topology. Even with matching innovations, the less-fit parent's disjoint/excess genes are systematically excluded from BOTH children.

This is a NEAT adaptation issue: standard NEAT produces one child per crossover, but the framework requires two. A better adaptation would swap parent order for the second child.

**Severity:** Medium. The XOR benchmark works (4/5 seeds converge), but diversity is reduced.

### 2. NEAT distance formula merges disjoint+excess with combined coefficient (Finding 4 → inv_05: CONFIRMED, minor)

The formula `(c1 + c2) * (E+D) / N` deviates from standard `c1*E/N + c2*D/N`. With defaults c1=c2=1.0, the structural distance term is 2x the standard formula. This changes the effective scale of speciation thresholds compared to NEAT literature values.

**Severity:** Minor. A documentation gap, not a bug. Parameters are internally consistent.

## Partially Confirmed

### 3. AntGenome mutation silent failure (Finding 3 → inv_04: code smell, not active bug)

The manual two-loop replacement in `AntGenome.mutate` cannot fail in practice for well-formed trees. The `===` identity comparison is sound. However:
- The fallthrough at line 192 returns an unmodified genome silently (dead code that masks future bugs)
- The manual replacement duplicates `replace_subtree!` used in crossover — inconsistency
- Both mutation and crossover have the same latent deficiency: silent no-op on replacement failure

**Severity:** Low. Code quality issue, not a correctness bug.

## Not Investigated (Known/Philosophical)

### 4. GraphGenome doesn't support recurrent networks (Finding 2: KNOWN)

This is a deliberate design choice — feedforward NEAT only. The topological sort correctly rejects cycles. The paper does not claim recurrent network support.

## Additional Finding (from iteration 001 follow-up)

### 5. `GeneticProgramming.max_depth` is dead configuration

Confirmed: `max_depth` is declared in the struct, accepted by the constructor, but never read by any code in `src/`. It does NOT control tree depth in `_random_tree` (which has its own hardcoded depth logic) or in any mutation/crossover operator. Users who set `max_depth` expecting depth-limited trees are silently ignored.

**Severity:** Low-medium. Misleading API parameter.

## Follow-up Candidates

1. **Paper observations accuracy** — Research questions 3-5 from RESEARCH.md (references correct? descriptions match code? comments match?) have not been systematically checked yet.
2. **LLM operator correctness** — The `llm_operator.jl` file was not reviewed in either iteration.
3. **Distributed island model** — `distributed_island.jl` was not reviewed.
4. **The `_mutate_weights!` function iterates over `values(g.connections)` with Dict iteration order** — not deterministic. Uses `rand(rng)` per connection, so iteration order affects which connections get the same random draws. This may affect reproducibility for GraphGenome.
