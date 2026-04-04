# Iteration 001 — Synthesis

## Confirmed Findings

### 1. Operator rate validation missing (Finding 1 → inv_01: CONFIRMED)

The `GeneticProgramming` constructor does not validate that `crossover_rate + mutation_rate <= 1.0`. When the sum exceeds 1.0:
- The effective mutation rate becomes `1.0 - crossover_rate`, not the configured value
- The reproduction (copy) operator is silently eliminated
- No warning or error is produced

The same pattern exists in the legacy `evolve!` function. No existing tests cover this edge case — all tests use rate sums <= 0.7.

**Severity:** Low-medium. Default rates are safe (0.3 + 0.3 = 0.6). The failure mode is silent misconfiguration.

**Fix (not applied, per RESEARCH.md scope):** Add `ArgumentError` in the constructor when `crossover_rate + mutation_rate > 1.0`.

### 2. Cross-island RNG contamination via migrant GenState (Finding 2 → inv_02: CONFIRMED)

When an ExprGenome migrates between islands, `deepcopy` copies the entire GenState including the source island's RNG. Subsequent mutations on the destination island use two RNG streams: the destination island's RNG for node selection, and the copied source island's RNG for content generation (via `g.state.rng` in `create_random_statement`, `mutate!`, `wrap_rvalue`, etc.).

**What is preserved:** Full determinism/reproducibility from a fixed seed. No shared-state corruption.

**What is wrong:** The explicit-RNG-everywhere convention is violated in spirit. The `rng` parameter to `mutate()` is partially decorative for ExprGenome — it controls node selection but not content generation. Migrant genomes propagate foreign RNG lineage into the destination island's population through subsequent breeding.

**Severity:** Low-medium. Pseudo-random quality is unaffected, but the architectural incoherence is a latent source of confusion.

**Fix (not applied):** Re-bind migrant genomes' `state` to the destination island's GenState after migration.

## Refuted / Not Investigated

### 3. Elitism contradicts fitness sharing (Finding 3: KNOWN/PHILOSOPHICAL)

Elitism based on raw fitness is standard behavior in speciated GP (including NEAT). Fitness sharing is applied only to the selection step for breeding, not to elitism. This is a deliberate design choice, not a bug. The paper observations document does not claim otherwise.

### 4. Migration doesn't re-evaluate speciation (Finding 4: KNOWN)

Standard island model behavior. Migrants are integrated into speciation on the next generation's `_apply_speciation!` call. Re-running speciation mid-generation would be non-standard.

## Follow-up Candidates

1. **The dual-RNG issue (inv_02) extends to all ExprGenome mutations, not just migrants.** The `mutate(op, g, rng)` API passes `rng` explicitly, but the mutation implementations also use `g.state.rng` for content generation. This means the explicit `rng` parameter is only partially effective. Worth checking whether this causes any practical issues in the non-island-model solve path.

2. **The `max_depth` field in GeneticProgramming is never used.** It's declared in the struct (algorithm.jl line 28) and accepted by the constructor, but no code reads `alg.max_depth`. Worth verifying — this could be dead configuration.

3. **The crossover implementation in evolution.jl has a potential depth-explosion issue.** The `crossover` function can swap subtrees of any depth, with no depth limit. Combined with no bloat penalty by default, this could produce arbitrarily deep trees. Worth checking if this is controlled elsewhere.

4. **Review TreeGenome, GraphGenome, AntGenome** — their operator implementations and solve paths are separate from ExprGenome and were not covered in this iteration.
