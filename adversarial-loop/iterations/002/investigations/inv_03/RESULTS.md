# Investigation 03: GraphGenome Crossover — Identical Children and NEAT Conformance

## What Was Tested

Formal analysis of the `crossover` function for GraphGenome (`src/genome/graph_genome.jl` lines 171-182) and its interaction with `_neat_crossover` (lines 294-342).

## Key Findings

### 1. Two children CAN differ — but only via matching-gene selection

The two calls to `_neat_crossover` share the same `rng` sequentially. Inside `_neat_crossover`, matching genes (innovations present in both parents) are assigned by `rand(rng, Bool)`. Since the RNG advances between the two calls, the two children receive DIFFERENT random selections for matching genes.

**When parents have matching innovations:** Children differ in which parent's weight/enabled-status they inherit for each matching gene. This produces genuine variation.

**When parents have NO matching innovations (fully disjoint):** No `rand` calls are made. The function is fully deterministic — both children are identical copies of the fitter parent's topology. The crossover consumes two population slots but produces only one unique genome. This reduces effective population size.

### 2. Standard NEAT produces ONE child, not two

The core issue is architectural. Standard NEAT crossover produces a single offspring. The GP framework's `_breed_next_generation!` requires two children (it advances `idx` by 2 on crossover). The adaptation of calling `_neat_crossover` twice with the same parent order is a reasonable but suboptimal solution:

- A better approach would be to swap parent order for the second child: `child2 = _neat_crossover(other, fitter, rng)`. This gives the second child the less-fit parent's disjoint/excess genes, preserving more genetic material.
- Alternatively: produce one child and fill the second slot with a mutated copy.

### 3. Disjoint/excess genes from the less-fit parent are ALWAYS lost

In `_neat_crossover`, the code only includes disjoint/excess genes from the `fitter` parent (`elseif has_f`). Genes unique to the `other` parent are silently discarded. This is correct per the NEAT specification for unequal fitness — disjoint/excess genes from the fitter parent are preferred. But since both children use the same parent order, there is NO pathway for the less-fit parent's unique structural innovations to survive crossover. They can only survive via reproduction (copy without crossover).

## Verdict: CONFIRMED

- Identical children when parents have fully disjoint innovations
- Systematic loss of less-fit parent's unique genes in both children
- Architectural mismatch: NEAT produces 1 child, framework needs 2

## Severity: Medium

The XOR NEAT benchmark (4/5 seeds converge) suggests this isn't catastrophic — the algorithm works despite this limitation. But it does reduce diversity pressure and means structural innovations from less-fit parents can only survive via the reproduction operator, not crossover.
