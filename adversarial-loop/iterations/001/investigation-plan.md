# Iteration 001 — Investigation Plan

## Gemini Finding Classification

| # | Title | Classification | Rationale |
|---|-------|---------------|-----------|
| 1 | Operator selection probabilities lack validation | ACTIONABLE | Testable: check if rates > 1.0 silently changes semantics |
| 2 | Island migration creates cross-island RNG dependencies | ACTIONABLE | Testable: check if migrant genomes carry source island RNG into mutations on destination |
| 3 | Elitism based on raw fitness contradicts fitness sharing | KNOWN/PHILOSOPHICAL | This is a deliberate design choice. Elitism preserving the best raw fitness individual is standard GP behavior. Fitness sharing is only applied to the selection step for breeding, not to elitism. This is how NEAT and most speciated GP implementations work. |
| 4 | Migration replaces individuals without re-evaluating speciation | KNOWN | Standard behavior in island models. Migrants are integrated on the next generation's speciation pass. This is by design — re-running speciation mid-generation would be unusual. |

## Investigations

### Investigation 1: Operator rate validation and semantics

**Hypothesis:** When `crossover_rate + mutation_rate > 1.0`, the mutation rate is silently clamped and reproduction becomes unreachable.

**Success/failure criteria:**
- CONFIRMED if: No validation prevents rates summing > 1.0, AND the effective mutation rate differs from configured
- REFUTED if: Constructor validates, OR the behavior is documented

**Investigation mode:** Code experiment + formal analysis

**Estimated effort:** 10 minutes

### Investigation 2: Cross-island RNG contamination via migrant genomes

**Hypothesis:** When an ExprGenome migrates between islands, its GenState.rng is the source island's RNG. Mutations on the destination island that call functions through g.state (e.g., `create_random_statement(g.state)`) will use the source island's RNG rather than the destination island's RNG.

**Success/failure criteria:**
- CONFIRMED if: SubtreeMutation on a migrant genome draws random numbers from the source island's RNG
- REFUTED if: The RNG is somehow replaced or all random draws go through the explicit `rng` parameter

**Investigation mode:** Formal analysis (trace the code paths) + code experiment

**Estimated effort:** 15 minutes
