# Adversarial Loop — Iteration Log

<!-- This file is the running summary of all iterations. Each iteration appends an entry. -->

## Iteration 001 — 2026-04-04

**Focus:** Core evolutionary loop, genetic operators, selection, and speciation (ExprGenome path)
**Gemini findings:** 2 actionable, 1 known/philosophical, 1 known
**Investigations run:**
- inv_01 (Operator rate validation): CONFIRMED — no validation on crossover_rate + mutation_rate sum; silent rate distortion when > 1.0
- inv_02 (Cross-island RNG contamination): CONFIRMED — migrant genomes carry source island's GenState.rng; mutations use dual RNG streams

**Confirmed findings:** (1) Missing constructor validation for operator rates, (2) Migrant ExprGenome RNG contamination
**Refuted findings:** None refuted; 2 findings classified as known/by-design (elitism vs sharing, migration vs speciation timing)
**Document changes:** None (per RESEARCH.md scope — no project files modified)
**Follow-up candidates:**
- `max_depth` field in GeneticProgramming appears unused — verify
- Crossover has no depth limit — check if controlled elsewhere
- The dual-RNG issue may extend beyond migrants to the general ExprGenome mutate API design
- TreeGenome, GraphGenome, AntGenome operator correctness (next iteration)
**So-what assessment:** Two real defects found. The rate validation gap is a straightforward missing check. The RNG contamination is more subtle — architecturally incoherent but practically harmless due to preserved determinism. Both are worth fixing but neither invalidates existing results.

## Iteration 002 — 2026-04-04

**Focus:** TreeGenome, GraphGenome (NEAT), AntGenome correctness + `max_depth` follow-up
**Gemini findings:** 3 actionable, 1 known, 1 overlap (merged with another)
**Investigations run:**
- inv_03 (GraphGenome crossover identical children): CONFIRMED — both children from same parent order; fully disjoint parents → identical children; less-fit parent's innovations systematically lost
- inv_04 (AntGenome mutation silent failure): NOT CONFIRMED as active bug — `===` identity comparison is sound; manual replacement duplicates `replace_subtree!` (code smell, not correctness issue)
- inv_05 (NEAT distance formula deviation): CONFIRMED (minor) — formula uses (c1+c2)*(E+D)/N instead of c1*E/N + c2*D/N; 2x scaling of structural distance with defaults

**Additional finding:** `GeneticProgramming.max_depth` is dead configuration — declared, accepted, never read.
**Additional finding:** `_mutate_weights!` iterates `values(g.connections)` (non-deterministic Dict order) with per-element `rand(rng)` calls — violates the project's deterministic-ordering convention.

**Confirmed findings:** (1) GraphGenome crossover produces identical children for disjoint parents, (2) NEAT distance 2x scaling, (3) `max_depth` dead code, (4) Dict iteration order in `_mutate_weights!`
**Refuted findings:** AntGenome mutation silent failure not an active bug
**Document changes:** None (per RESEARCH.md scope)
**Follow-up candidates:**
- Paper observations accuracy (RESEARCH.md questions 3-5)
- LLM operator correctness review
- Distributed island model review
- Full audit of Dict/Set iteration in GraphGenome for deterministic-ordering violations
**So-what assessment:** The GraphGenome crossover issue is the most significant finding — it's a real NEAT conformance deviation that reduces diversity. The `max_depth` dead code and Dict iteration order are straightforward fixes. The NEAT distance 2x scaling is a documentation gap.

## Iteration 003 — 2026-04-04

**Focus:** LLM mutation operator, AST sanitizer, paper observations accuracy, GraphGenome Dict iteration
**Gemini findings:** 4 actionable, 1 known, 1 refuted (Gemini was wrong about fitness sharing)
**Investigations run:**
- inv_06 (LLM operator exception handling): CONFIRMED (low probability) — `deserialize`/`sanitize` not wrapped in try/catch; StackOverflowError could crash loop; trivial fix
- inv_07 (GraphGenome Dict iteration audit): CONFIRMED — 6 mutation/crossover sites iterate Dict/Set with RNG in non-deterministic order; violates documented determinism convention
- inv_08 (JSON escaping completeness): CONFIRMED (minor) — unescape order bug (`\\\\` should be first), missing RFC 8259 control char escapes, missing `\uXXXX` unescaping

**Confirmed findings:** (1) GraphGenome non-deterministic Dict iteration (6 sites), (2) JSON unescape order bug, (3) LLM operator missing try/catch
**Refuted findings:** Gemini incorrectly challenged the fitness sharing inversion claim — the paper's explanation is correct
**Document changes:** None (per RESEARCH.md scope)
**Follow-up candidates:** Distributed island model review, TreeGenome `_replace_nth_node!` correctness, `defaults.jl` function sets
**So-what assessment:** The GraphGenome Dict iteration finding is the highest-impact result — it directly contradicts a documented project convention and breaks seed-deterministic reproducibility for NEAT. The JSON and LLM exception findings are real but low-impact. The paper observations are mostly accurate except for claim 1.6 (deterministic ordering) which is violated by GraphGenome.

---

## Batch Summary — Iterations 001-003 (2026-04-04)

### Cumulative Results

| # | Finding | Source | Severity | Type |
|---|---------|--------|----------|------|
| 1 | Missing operator rate validation (crossover_rate + mutation_rate > 1.0) | Iter 001 | Low-Medium | Missing validation |
| 2 | Migrant ExprGenome RNG contamination via GenState | Iter 001 | Low-Medium | Architectural |
| 3 | GraphGenome crossover produces identical children for disjoint parents | Iter 002 | Medium | NEAT conformance |
| 4 | NEAT distance formula 2x scaling vs standard | Iter 002 | Minor | Documentation gap |
| 5 | `GeneticProgramming.max_depth` dead configuration | Iter 002 | Low-Medium | Dead code |
| 6 | GraphGenome non-deterministic Dict iteration (6 sites) | Iter 003 | High | Reproducibility violation |
| 7 | JSON unescape order bug in LLM operator | Iter 003 | Low | Edge case |
| 8 | LLM operator missing try/catch around deserialize/sanitize | Iter 003 | Low | Robustness gap |

### Ranked by Impact

1. **GraphGenome Dict iteration non-determinism** (6 sites) — directly breaks documented reproducibility guarantee for NEAT
2. **GraphGenome crossover identical children** — reduces diversity; NEAT conformance deviation
3. **Missing operator rate validation** — silent misconfiguration; easy fix
4. **`max_depth` dead code** — misleading API; easy to fix or remove
5. **Migrant ExprGenome RNG contamination** — architecturally incoherent but preserves determinism
6. **NEAT distance 2x scaling** — documentation gap; internally consistent
7. **LLM operator missing try/catch** — formal gap in graceful degradation claim
8. **JSON unescape order** — edge case unlikely to manifest

### Confirmation Rate
- Gemini findings across 3 iterations: 14 total
- Investigated: 8
- Confirmed: 7 (including 2 minor/edge-case)
- Refuted: 1 (fitness sharing claim was correct)
- Known/Not investigated: 5

**Confirmation rate: 87.5%** (7/8 investigated findings confirmed)

### Stopping: 3 iterations completed — pausing for human review.

