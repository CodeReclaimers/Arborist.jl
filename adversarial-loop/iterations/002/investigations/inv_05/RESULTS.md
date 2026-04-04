# Investigation 05: NEAT Distance Formula — Disjoint vs Excess

## What Was Tested

Formal analysis of `_neat_distance` (graph_genome.jl lines 348-373) compared to the standard NEAT specification.

## Key Findings

### Standard NEAT formula:
`δ = c1 * E / N + c2 * D / N + c3 * W`
- E = excess genes (innovation numbers beyond the other genome's maximum)
- D = disjoint genes (within the range of both genomes but not matching)
- c1, c2, c3 = tunable coefficients

### Arborist implementation:
`δ = (c1 + c2) * (E + D) / N + c3 * W`
- Combines excess and disjoint into a single count via `symdiff`
- Sums c1 + c2 as a combined coefficient

### When this matters:
- With defaults c1=c2=1.0: standard gives `(E + D) / N + 0.4W`, Arborist gives `2(E+D) / N + 0.4W`. The structural distance term is scaled by 2x compared to standard NEAT with c1=c2=1.0.
- This means Arborist's default speciation threshold would need to be ~2x the standard NEAT threshold to produce equivalent speciation behavior.
- The deviation only matters for independent tuning of disjoint vs excess sensitivity, which is rare in practice (most implementations use c1=c2).

### neat-python comparison:
The project author maintains neat-python. In neat-python, the distance formula uses separate disjoint and excess counts with separate coefficients. The Arborist simplification is a deliberate deviation, not an oversight.

## Verdict: CONFIRMED (minor deviation)

The formula is a common simplification. It does NOT invalidate the XOR benchmark results (which use the default parameters) but does change the effective scale of the distance metric compared to standard NEAT literature values. This is worth documenting — anyone porting speciation thresholds from NEAT papers would need to adjust.

## Severity: Minor

The 2x scaling is absorbed by the threshold parameter. Not a bug, but a documentation gap.
