# Investigation inv_07: Dict Iteration Non-Determinism in GraphGenome

**File audited:** `/home/alan/GenProg.jl/src/genome/graph_genome.jl`

**Convention under test:** "All Dict/Set sampling uses `_sorted_*` helpers" (from CLAUDE.md)

## Summary

Seven sites iterate over Dict keys/values/pairs with RNG consumption during iteration. **Six of the seven are non-deterministic.** One function (`_topological_sort`) correctly applies sorting. The `_sorted_*` helper pattern used elsewhere in the codebase (e.g., `codegen.jl`) is not used anywhere in `graph_genome.jl`.

---

## Sites Found

### 1. `_mutate_weights!` -- Line 220

- **What is iterated:** `values(g.connections)` (Dict values)
- **How RNG is used:** `rand(rng)` and `randn(rng)` called per connection (line 221-222)
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **YES.** Each connection is independently perturbed with probability 0.9. Because `rand(rng)` is consumed once per connection, the order of iteration determines which random draw is applied to which connection. Different Dict iteration orders produce different weight perturbations even with the same RNG state.

### 2. `_mutate_weight_replace!` -- Line 228

- **What is iterated:** `collect(values(g.connections))` (Dict values collected to a Vector)
- **How RNG is used:** `rand(rng, conns)` selects one connection (line 230)
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **YES.** `collect(values(...))` produces a Vector in non-deterministic Dict iteration order. `rand(rng, conns)` picks an index into that Vector, so the same RNG draw selects different connections depending on iteration order.

### 3. `_mutate_add_connection!` -- Line 235

- **What is iterated:** `collect(keys(g.nodes))` (Dict keys collected to a Vector)
- **How RNG is used:** `rand(rng, node_ids)` called up to 20 times in a loop (line 240-241), plus `randn(rng)` for weight (line 257)
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **YES.** Same reasoning as site 2. `rand(rng, node_ids)` indexes into a Vector whose order depends on Dict iteration order.

### 4. `_mutate_add_node!` -- Line 263

- **What is iterated:** `values(g.connections)` via list comprehension (line 263)
- **How RNG is used:** `rand(rng, enabled_conns)` selects a connection (line 266), `rand(rng, [:sigmoid, :tanh, :relu])` selects activation (line 271)
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **YES.** The comprehension `[c for c in values(g.connections) if c.enabled]` produces a Vector in non-deterministic order. `rand(rng, enabled_conns)` then selects from that Vector by index.

### 5. `_mutate_toggle_connection!` -- Line 284

- **What is iterated:** `collect(values(g.connections))` (Dict values collected to a Vector)
- **How RNG is used:** `rand(rng, conns)` selects one connection (line 286)
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **YES.** Same pattern as sites 2 and 4.

### 6. `_neat_crossover` -- Lines 300-302

- **What is iterated:** `union(keys(fitter.connections), keys(other.connections))` -- a **Set** (line 300), iterated on line 302
- **How RNG is used:** `rand(rng, Bool)` called for each matching gene (line 308)
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **YES.** The `for inn in all_innovations` loop iterates over a Set in non-deterministic order. For each matching innovation, `rand(rng, Bool)` is consumed. The consumption order of RNG draws is therefore non-deterministic, meaning different matching genes get different random choices depending on iteration order.

### 7. `complexity` -- Line 193

- **What is iterated:** `values(g.connections)` (Dict values)
- **How RNG is used:** None
- **Sorted before iteration:** No
- **Non-deterministic outcome:** **No.** This only counts enabled connections -- it's a pure reduction with no RNG involvement. Non-deterministic iteration order does not affect the result.

---

## Additional Checks

### `_topological_sort` (line 472)

- **Initialization:** Iterates `keys(g.nodes)` twice (lines 475, 480) to build `in_degree` and `adj` dicts. These are pure assignments (no RNG), so non-deterministic order does not affect the resulting data structures.
- **Kahn's algorithm:** The initial queue is built from `in_degree` pairs (line 493, `[n for (n, d) in in_degree if d == 0]`) and then **sorted on line 494** (`sort!(queue)`). Adjacency lists are also **sorted on line 500** (`sort!(adj[n])`).
- **Verdict:** **Correctly deterministic.** Sorting is applied at both queue initialization and neighbor expansion. This is the only function in the file that applies deterministic ordering.

### `evaluate_genome` (line 410)

- **Input/output ID collection (lines 421-422):** `[n.id for n in values(g.nodes) if n.type == :input]` iterates `values(g.nodes)` in non-deterministic order, but the result is immediately passed to `sort!()`.
- **Bias ID collection (line 423):** `[n.id for n in values(g.nodes) if n.type == :bias]` iterates non-deterministically and is **not sorted**. However, bias nodes all get the same value (1.0), so the order is irrelevant to correctness.
- **Inner loop (line 443):** `for c in values(g.connections)` iterates connections in non-deterministic order, but the loop body is a pure summation (`total += ...`), which is commutative. No RNG is involved.
- **Verdict:** **Functionally deterministic** despite unsorted iteration. No RNG is used in this function. The summation is commutative (modulo floating-point associativity, which is a negligible concern here).

### `serialize` (line 196)

- Correctly sorts: `sort!(collect(g.nodes), by=first)` and `sort!(collect(g.connections), by=first)`.

### `_copy_graph` (line 512)

- Iterates `g.nodes` and `g.connections` but only copies entries into a new Dict. No RNG, no ordering dependency.

### `_neat_distance` (line 348)

- Iterates `matching` (a Set from `intersect`) on line 363. No RNG; the loop body is a commutative summation. Functionally deterministic.

---

## Impact Assessment

The six non-deterministic sites (1-6) mean that **GraphGenome evolution is not reproducible from a given RNG seed**, violating the project's explicit determinism convention. The `_topological_sort` function and `evaluate_genome` are correctly handled, but every mutation operator and the crossover function are affected.

### Recommended Fixes

Each site needs one of:
- Sort the collected Vector by a stable key (innovation number for connections, node ID for nodes) before indexing with `rand(rng, ...)`
- Sort the Set before iterating when RNG is consumed per-element

Concrete patterns:
- `values(g.connections)` with RNG --> `sort!(collect(values(g.connections)), by=c -> c.innovation)` or `[g.connections[k] for k in sort!(collect(keys(g.connections)))]`
- `keys(g.nodes)` with RNG --> `sort!(collect(keys(g.nodes)))`
- `union(keys(...), keys(...))` with RNG --> `sort!(collect(union(...)))`

### Severity

**High for reproducibility.** Any test or benchmark relying on seed-deterministic GraphGenome results will be fragile. The `solve()` function for GraphGenome calls `mutate` and `crossover` every generation, so the non-determinism compounds across the entire evolutionary run.
