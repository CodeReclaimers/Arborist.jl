# Investigation 02: ExprGenome GenState.rng Contamination During Island Migration

## What Was Tested

Traced the RNG usage path when an ExprGenome migrates between islands in the sequential IslandModel to determine whether the migrant's GenState.rng contaminates the destination island's RNG stream.

## Code Path Trace

### 1. Island Initialization (solve.jl lines 284-291)

Each island gets an independent RNG seeded from the master RNG:

```julia
island_rng = Random.MersenneTwister(island_rng_seed)   # line 286
pop = _initialize_population(problem, alg, island_rng)  # line 287
```

`_initialize_population` (solve.jl lines 33-49) creates a **single shared GenState** per island (line 40), then gives that same `state` object to every genome on that island (line 45). All genomes on island I share `island_states[I].rng`.

### 2. Migration (solve.jl lines 407-437)

Line 417: `deepcopy(island_genomes[i][order[j]])` creates the emigrant. This deep-copies the ExprGenome, including its `state::GenState` field, including `state.rng`.

**Critical detail:** Before deepcopy, the genome's `state` field points to the shared island GenState. After deepcopy, the emigrant has its own **independent copy** of that GenState (including a copy of the RNG at its current state). The emigrant is then inserted into the destination island's genome array (line 431).

### 3. Breeding on the Destination Island (solve.jl lines 326-337)

The island loop calls `_breed_next_generation!` with `state.rng` (line 337), which is the **destination island's** GenState RNG. Inside `_breed_next_generation!` (lines 78-108):

- Tournament selection uses the passed `rng` (destination island's) -- line 98
- Mutation operator selection uses the passed `rng` -- line 99
- `mutate(op, genomes[p_idx], rng)` is called with the destination `rng` -- line 100

### 4. The Split: Two Different RNGs Used in One Mutation

**SubtreeMutation.mutate** (mutation.jl line 44-51):
- Line 47: `rand(rng, 1:length(new_body))` -- uses the **destination island's RNG** (passed parameter) to pick which statement to replace
- Line 48: `create_random_statement(g.state)` -- uses **g.state.rng** to generate the replacement subtree

If `g` is a migrant genome, `g.state` is the deep-copied GenState from the source island. So `g.state.rng` is a snapshot of the source island's RNG at migration time. All random number generation inside `create_random_statement` (codegen.jl line 394: `rand(s.rng, s.statement_types)`, line 399: `rand(s.rng, ...)`, and every downstream function) uses this copied RNG.

**PointMutation.mutate** (mutation.jl line 61-79):
- Line 69: `rand(rng, all_nodes)` -- uses **destination island's RNG** to pick target node
- Line 71: `mutate!(g.state, target)` -- uses **g.state.rng** for the actual mutation content

`mutate!` (codegen.jl line 414) dispatches to functions like `mutate_assignment!` (line 242) which uses `rand(s.rng, Bool)`, `rand(s.rng, get_lvalues_of_type(...))`, etc.

**ExpansionMutation.mutate** (mutation.jl line 144-176):
- Line 160: `rand(rng, leaf_positions)` -- uses **destination island's RNG**
- Line 165: `wrap_rvalue(g.state, leaf_value)` -- uses **g.state.rng** (codegen.jl line 222: `rand(s.rng, available_funcs)`)

**HoistMutation.mutate** (mutation.jl line 90-134):
- Lines 110, 114: `rand(rng, ...)` -- uses **destination island's RNG** for node selection
- No call to g.state.rng (purely structural operation)

### 5. Non-Migrant Genomes Also Affected

This is not limited to migrants. Even on a single island, the genome's `state` field points to the shared island GenState. When `_breed_next_generation!` calls `mutate(op, genomes[p_idx], rng)`, the `rng` parameter and `genomes[p_idx].state.rng` are the **same object** (both are `island_states[isle].rng`). So for non-migrant genomes, the two RNG paths happen to use the same RNG instance -- no split occurs.

The split only manifests for migrant genomes, where `g.state.rng` is a deep-copied RNG from the source island.

## Verdict: CONFIRMED

The RNG contamination path exists exactly as described.

## Impact Assessment

### What is NOT broken:

1. **Determinism/reproducibility is preserved.** `deepcopy` creates a deterministic snapshot of the source island's RNG state. Given the same master seed, the same migration happens at the same generation, the same RNG state is copied, and the same mutation outcomes occur. The run is fully reproducible.

2. **No shared-state corruption.** The deep-copied RNG is independent of the source island's actual RNG. Advancing the copy does not affect the source island.

3. **No crash or type error.** The copied GenState has all the same function sets, variable tables, etc. as the source island (they are constructed from the same problem specification).

### What IS wrong:

1. **Dual-RNG mutation.** When a migrant genome is mutated, the node-selection step uses the destination island's RNG and the content-generation step uses the copied source island's RNG. This means two independent RNG streams contribute to a single mutation operation, which is architecturally incoherent even if deterministic.

2. **Destination island's RNG stream is partially bypassed.** The content of mutations applied to migrant genomes is determined by an RNG that is not derived from the destination island's seed. If you changed only the destination island's seed, the mutation content for migrant genomes would not change (only node selection would change).

3. **Cross-island coupling of RNG sequences.** The destination island's mutation outcomes for migrant genomes depend on the source island's internal RNG state at migration time. This coupling is invisible to anyone reasoning about island independence.

4. **Accumulation over generations.** A migrant genome that is selected for mutation multiple times will keep advancing its private RNG copy. If that genome is later selected as a parent for crossover, the child inherits the same GenState (via `ExprGenome(new_body, g.state)` on line 50 of mutation.jl), propagating the foreign RNG further into the destination island's population.

### Severity: Low-to-Medium

- **Low** for practical impact: The mutation outcomes are still pseudo-random and deterministic. The quality of evolution is unlikely to be materially affected -- a "wrong" RNG stream produces random numbers that are just as good for GP purposes.
- **Medium** for architectural cleanliness: The explicit-RNG-everywhere convention (documented in CLAUDE.md) is violated in spirit. The `rng` parameter to `mutate()` is partially decorative for ExprGenome -- it controls node selection but not content generation. This is a latent source of confusion for anyone trying to reason about RNG flow.

### Suggested Fix (not implemented)

After migration, re-bind migrant genomes' `state` to the destination island's shared GenState:

```julia
# In _migrate!, after inserting emigrant into destination:
island_genomes[dest][worst_idx] = ExprGenome(genome.body, island_states[dest])
```

This would ensure all genomes on an island share the island's GenState (and its RNG), maintaining the single-RNG-stream invariant. The `island_states` array would need to be passed to `_migrate!`.
