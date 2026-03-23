# Speciation

Arborist.jl provides three speciation strategies that control diversity pressure during evolution.

## NoSpeciation

All individuals belong to a single species. This is the default. Use when fast convergence is desired and diversity maintenance is not a concern.

## ThresholdSpeciation (genomic distance)

NEAT-style speciation using the genome's `distance()` function. Individuals are assigned to the first species whose representative has `distance(g1, g2) <= threshold`.

```julia
speciation = ThresholdSpeciation(
    threshold = 10.0,          # compatibility distance cutoff
    min_species_size = 2,      # minimum size to avoid culling
    stagnation_limit = 15,     # generations without improvement before culling
    sharing_formula = :log2,   # fitness sharing strength
)
```

Best suited for NEAT-style topology evolution (GraphGenome) where structural innovation needs protection. For ExprGenome, syntactic distances tend to be large even for small changes, which can create too many species.

## BehavioralSpeciation (behavioral distance)

Speciation based on what programs *do*, not how they're structured. Two programs are in the same species if they make similar decisions on a fixed set of probe inputs, regardless of AST structure.

```julia
speciation = BehavioralSpeciation(
    fingerprint_fn = g -> compute_fingerprint(g, probe),  # genome -> fingerprint
    distance_fn = hamming_distance,                        # (fp, fp) -> Float64
    threshold = 0.15,          # behavioral distance cutoff
    min_species_size = 2,
    stagnation_limit = 15,
    sharing_formula = :sqrt,   # fitness sharing strength
)
```

The user provides two functions:
- `fingerprint_fn(genome)` — computes a behavioral fingerprint (any type)
- `distance_fn(a, b)` — distance between two fingerprints (0.0 = identical behavior)

Best suited for behavioral optimization problems (bin packing, game playing, controller synthesis) where multiple syntactically distinct programs implement the same strategy.

## Fitness sharing

All speciation strategies that track species support configurable fitness sharing via the `sharing_formula` parameter. For minimization problems (lower fitness = better), sharing penalizes members of large species to prevent monocultures:

| Formula | Penalty factor | Use case |
|---------|---------------|----------|
| `:none` | 1.0 | No sharing (species assignment only) |
| `:log2` | 1 + log2(\|S\|) | Gentle; singletons unpenalized (default for ThresholdSpeciation) |
| `:sqrt` | sqrt(\|S\|) | Moderate (default for BehavioralSpeciation) |
| `:linear` | \|S\| | Strong; NEAT default for topology protection |

The `apply_sharing(raw_fitness, species_size, formula)` function is available for custom speciation implementations.

## Choosing a strategy

- **Simple regression/optimization**: `NoSpeciation()` — fast convergence, no overhead
- **NEAT topology evolution**: `ThresholdSpeciation(sharing_formula=:linear)` — protects novel topologies
- **Behavioral optimization** (e.g., bin packing): `BehavioralSpeciation(sharing_formula=:sqrt)` — collapses syntactic variants, maintains strategic diversity
- **ExprGenome with diversity pressure**: `ThresholdSpeciation(threshold=25.0, sharing_formula=:sqrt)` — use a high threshold to avoid singleton explosion

## Species lifecycle

For both `ThresholdSpeciation` and `BehavioralSpeciation`:

1. Each generation, individuals are assigned to species by distance to the species representative
2. If no species matches, a new species is created
3. Stagnation is tracked per species (generations without fitness improvement)
4. Species are culled when stagnant AND below `min_species_size`, except the species containing the global best individual
5. Orphaned individuals are redistributed to the nearest remaining species
