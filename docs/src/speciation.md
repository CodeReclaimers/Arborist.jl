# Speciation

## NoSpeciation

All individuals belong to a single species. This is the default.

## ThresholdSpeciation

NEAT-style speciation with fitness sharing, adapted for minimization (lower fitness = better).

```julia
speciation = ThresholdSpeciation(
    threshold = 10.0,        # compatibility distance cutoff
    min_species_size = 2,    # minimum size to avoid culling
    stagnation_limit = 15,   # generations without improvement before culling
)
```

Fitness sharing multiplies each individual's raw fitness by its species size, making members of large species appear worse for selection. This protects structural innovations in small species from being overwhelmed by the dominant population — the core mechanism that makes NEAT work.

Species are culled when they exceed `stagnation_limit` generations without improvement AND are below `min_species_size`. The species containing the global best individual is never culled.
