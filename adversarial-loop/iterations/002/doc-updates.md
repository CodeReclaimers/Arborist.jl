# Iteration 002 — Document Updates

Per RESEARCH.md scope: **No project files were modified.**

## Findings to record when approved:

1. **GraphGenome crossover produces identical children for disjoint parents** — The `crossover` function calls `_neat_crossover` twice with same parent order. Both children get fitter parent's topology exclusively. Less-fit parent's unique innovations never survive crossover.

2. **NEAT distance formula 2x scaling** — The combined (c1+c2) coefficient means speciation thresholds are 2x the standard NEAT scale. Document this for users porting parameters from NEAT literature.

3. **`GeneticProgramming.max_depth` is dead code** — Declared, accepted, never read. Either wire it into tree generation or remove it.

4. **AntGenome mutation duplication** — The manual two-loop replacement in `mutate` duplicates `replace_subtree!` from evolution.jl used in crossover. Functionally equivalent but inconsistent.

5. **GraphGenome `_mutate_weights!` uses non-deterministic Dict iteration** — Iterating `values(g.connections)` with `rand(rng)` per element means iteration order affects RNG consumption. This violates the "deterministic sorted ordering" convention documented in CLAUDE.md for Dict/Set sampling.
