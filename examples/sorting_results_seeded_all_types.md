# Sorting Algorithm Evolution Results — Seeded (All Template Types)

**Date:** 2026-03-25
**Seed:** 42

## Parameters
- pop_size=300, generations=500
- mutation_rate=0.4, crossover_rate=0.3
- elitism=5, tournament_size=5
- bloat_penalty=0.0005
- curriculum: 3 → 20
- upgrade_threshold=0.05
- comparison_alpha_base=0.0
- max_seed_type=4 (all seed templates: single-pass, bubble sort, partition, selection sort)

## Results
- **Best fitness:** 0.028
- **Final curriculum length:** 20
- **Achievement:** Tier 2 — 100% correct sorting on all lengths 3–24
- **Generalization:** Yes — evolved program generalizes to array lengths well beyond training curriculum
- **Converged:** within 5 generations at this population size

## Interpretation

With all four seed template types (including the selection sort and bubble sort
skeletons), evolution rapidly preserves and slightly simplifies the selection
sort structure. The nested-loop architecture is provided by the seed; evolution
refines index arithmetic and eliminates dead code.

This result demonstrates that the framework handles non-trivial control flow
(nested loops, conditional swaps, index arithmetic) when seeded with
appropriate structural templates. See `sorting_results_ablation_type1.md`
for the ablation showing that evolution cannot discover nested-loop structure
from single-pass seeds alone at this scale.
