# Iteration 001 — Document Updates

Per RESEARCH.md scope: **No project files were modified.** All findings are documented in the adversarial-loop artifacts only.

## Findings that should be recorded when the human approves changes:

1. **Operator rate validation** — Add to CLAUDE.md "Known Limitations" or fix in code:
   - `GeneticProgramming` constructor does not validate `crossover_rate + mutation_rate <= 1.0`
   - Silent rate distortion when sum exceeds 1.0

2. **RNG contamination** — Add to CLAUDE.md "Known Limitations" or fix in code:
   - Migrant ExprGenomes carry source island's GenState.rng; mutations use dual RNG streams
   - The `rng` parameter to `mutate()` is partially decorative for ExprGenome
   - Reproducibility is preserved; architectural cleanliness is not

3. **`max_depth` appears unused** — Follow-up candidate for next iteration; if confirmed, either remove the field or wire it into tree generation.
