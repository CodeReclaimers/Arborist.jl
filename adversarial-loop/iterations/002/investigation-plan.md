# Iteration 002 — Investigation Plan

## Gemini Finding Classification

| # | Title | Classification | Rationale |
|---|-------|---------------|-----------|
| 1 | GraphGenome crossover produces identical offspring | ACTIONABLE | Testable: check if both children are identical when parents have disjoint innovations. Also check the more general case with matching innovations — two calls with same RNG will produce different `rand(rng, Bool)` sequences. |
| 2 | GraphGenome doesn't support recurrent networks | KNOWN/PHILOSOPHICAL | The code deliberately uses topological sort and returns Inf for cycles. This is a design choice (feedforward NEAT), not a bug. The paper doesn't claim recurrent network support. |
| 3 | AntGenome mutation may fail silently | ACTIONABLE | The replacement logic is manual and complex. Worth verifying whether the nested search can miss targets. |
| 4 | NEAT distance formula deviates from standard | ACTIONABLE | Formal analysis — the formula combines disjoint+excess into one term. Worth checking if this matters for the documented XOR benchmark claim. |
| 5 | GraphGenome crossover flaw (same parent order twice) | ACTIONABLE (overlaps with Finding 1) | This is essentially the same observation as Finding 1 but with more emphasis on diversity loss. The key question: is NEAT crossover supposed to produce two children? Standard NEAT produces ONE child per crossover. The GP framework requires two children because `_breed_next_generation!` consumes them in pairs. |

## Investigations

### Investigation 3: GraphGenome crossover — identical children and NEAT conformance

**Hypothesis:** GraphGenome crossover produces two children by calling `_neat_crossover` twice with the same parent ordering. When parents have no matching innovations, both children are identical. Even with matching innovations, the two `rand(rng, Bool)` calls produce different sequences, so children differ only in the random matching-gene selection.

**Success/failure criteria:**
- CONFIRMED if: Two children are identical when no innovations match
- PARTIALLY CONFIRMED if: Children differ only via the stochastic matching-gene selection, but disjoint/excess genes are always from the same (fitter) parent

**Investigation mode:** Formal analysis + code experiment

**Estimated effort:** 10 minutes

### Investigation 4: AntGenome mutation silent failure rate

**Hypothesis:** The manual `===` identity-comparison replacement in AntGenome.mutate can fail to find the target node in some tree structures, returning an unchanged genome.

**Success/failure criteria:**
- CONFIRMED if: Mutation returns unchanged genomes at a non-trivial rate (>1%)
- REFUTED if: The replacement logic handles all reachable tree structures

**Investigation mode:** Formal analysis (trace the code logic for edge cases)

**Estimated effort:** 10 minutes

### Investigation 5: NEAT distance formula — disjoint vs excess

**Hypothesis:** The formula `(c1 + c2) * disjoint_excess / N` merges disjoint and excess genes into one term, deviating from standard NEAT which uses `c1 * E / N + c2 * D / N`.

**Success/failure criteria:**
- CONFIRMED if: The code uses a single combined count where NEAT specifies two separate counts
- Assessment: Is this a meaningful deviation or a common simplification?

**Investigation mode:** Formal analysis

**Estimated effort:** 5 minutes
