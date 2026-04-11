# Arborist.jl — Online Bin Packing Example

You are implementing a built-in example for Arborist.jl (formerly
GenProg.jl) that demonstrates evolving a bin packing heuristic using
the simulator pattern with ExprGenome. Read the existing src/,
ext/, and test/ directories in full before writing anything, with
particular attention to:
- src/genome/ant_genome.jl (the simulator pattern to follow)
- src/genome/codegen.jl (FunctionSet, GenState, FunctionDetails)
- src/genome/evolution.jl (evaluate_individual!, add_loop_checks)
- test/benchmarks/ant_trail.jl (how benchmarks are structured)

This example must work as both a standalone runnable script and as
a registered example in examples/bin_packing.jl.

---

## Background

In 2023, DeepMind's FunSearch paper (Romera-Paredes et al., Nature)
demonstrated that combining LLMs with evolutionary search could
discover novel bin packing heuristics that outperform classic
algorithms. The specific problem was online 1D bin packing:

- Items arrive one at a time with sizes drawn from a distribution
- Bins have fixed capacity C = 1.0
- Each item must be placed immediately (no lookahead)
- Goal: minimize total bins used (equivalently, maximize bin utilization)

Classic heuristics for comparison:
- First Fit (FF): place in first bin that fits
- Best Fit (BF): place in the bin with least remaining space that
  still fits (minimizes wasted space per placement)
- First Fit Decreasing (FFD): sort items descending first, then FF
  (not applicable to online setting without lookahead)

FunSearch's target: beat Best Fit on the standard benchmark
distribution (item sizes uniform in [0, 1], 1000 items per episode).

---

## Problem specification

### Bin packing simulator

Implement in examples/bin_packing.jl as module-level state following
the AntGenome pattern:

```julia
mutable struct BinPackingState
    bins::Vector{Float32}      # remaining capacity of each open bin
    n_bins::Int32              # number of bins opened so far
    items_placed::Int32        # items successfully placed
    total_waste::Float32       # accumulated wasted space
    capacity::Float32          # bin capacity (always 1.0f0)
end

const _bp_state = Ref{BinPackingState}(
    BinPackingState(Float32[], Int32(0), Int32(0), 0.0f0, 1.0f0)
)
```

Expose these scalar primitives to the function set:

```julia
# Query primitives (read-only, safe to call multiple times)
bp_n_bins()::Int32          # number of currently open bins
bp_bin_remaining(i::Int32)::Float32  # remaining capacity of bin i (1-indexed, clamped)
bp_item_size()::Float32     # size of the current item being placed
bp_capacity()::Float32      # bin capacity (always 1.0f0)

# Action primitive (side-effectful, should be called once)
bp_place_in_bin(i::Int32)::Bool  # place current item in bin i
                                  # returns true if successful
                                  # returns false if item doesn't fit
                                  # opens a new bin if i > n_bins

# Arithmetic helpers (these go in the function set, not as primitives)
# Standard +, -, *, / on Float32 and Int32
```

The evolved program receives the current item and must call
`bp_place_in_bin(i)` for some bin index `i`. If the program fails
to place the item (doesn't call bp_place_in_bin, or calls it with
an invalid index), the evaluator falls back to opening a new bin.

### Evolved program structure

The evolved program is an `ExprGenome` with:
- Input variables: none (state accessed via primitives)
- Output variables: `result::Bool` (whether placement succeeded)
- Temp variables: 4-6 Int32 and Float32 variables for loop counters
  and score accumulators
- Function set: bp_n_bins, bp_bin_remaining, bp_item_size,
  bp_capacity, bp_place_in_bin, plus standard arithmetic and
  comparison operators

The key insight: a good evolved program will discover a loop like:
```julia
best_bin = Int32(0)
best_score = Float32(-1.0)
i = Int32(1)
while i <= bp_n_bins()
    remaining = bp_bin_remaining(i)
    if remaining >= bp_item_size()
        score = remaining - bp_item_size()  # Best Fit score
        if score < best_score || best_bin == 0
            best_bin = i
            best_score = score
        end
    end
    i = i + Int32(1)
end
bp_place_in_bin(best_bin)
```

### Fitness function

Evaluate each genome over multiple episodes to get stable fitness:

```julia
struct BinPackingEvaluator <: AbstractEvaluator
    n_episodes::Int        # number of independent runs (default 20)
    n_items::Int           # items per episode (default 200)
    capacity::Float32      # bin capacity (default 1.0f0)
    item_dist::Symbol      # :uniform or :bimodal (default :uniform)
    rng_seed::Int          # for reproducible evaluation (default 42)
end
```

Fitness = mean bins used across episodes, normalized by optimal lower
bound (sum of item sizes / capacity). Perfect packing = 1.0.
Lower is better. Best Fit typically achieves ~1.02-1.05 on uniform
items. First Fit typically achieves ~1.05-1.10.

The lower bound normalization makes fitness comparable across episodes
with different item distributions.

Evaluation procedure per episode:
1. Generate n_items from the specified distribution using a seeded RNG
   (seed = rng_seed + episode_index for reproducibility)
2. Reset _bp_state[] to empty
3. For each item: set current item size, run evolved program,
   check if placement succeeded, fall back to new bin if not
4. Record bins_used / lower_bound as episode fitness
5. Return mean across all episodes

### Comparison baselines

Implement these as Julia functions for comparison:

```julia
function first_fit(items::Vector{Float32}, capacity::Float32)::Int
function best_fit(items::Vector{Float32}, capacity::Float32)::Int
function worst_fit(items::Vector{Float32}, capacity::Float32)::Int
```

These run the classic heuristics and return total bins used.
Use them in the fitness evaluator to compute relative performance:

```julia
# Optional: report evolved fitness relative to Best Fit
bf_fitness = best_fit_normalized(items, capacity)
evolved_fitness = run_evolved(genome, items, capacity)
improvement = (bf_fitness - evolved_fitness) / bf_fitness * 100
```

---

## Evolution parameters

Start with these parameters and adjust based on results:

```julia
algorithm = GeneticProgramming(
    pop_size        = 200,
    generations     = 500,
    mutation_rate   = 0.4,
    crossover_rate  = 0.3,
    elitism         = 3,
    tournament_size = 5,
    max_depth       = 10,
    bloat_penalty   = 0.001,
    mutation_ops    = [SubtreeMutation(), PointMutation(),
                       HoistMutation(), ExpansionMutation()],
)

evaluator = BinPackingEvaluator(
    n_episodes = 20,
    n_items    = 200,
    capacity   = 1.0f0,
    item_dist  = :uniform,
    rng_seed   = 42
)
```

If convergence is slow after 200 generations, increase n_episodes
to 50 and n_items to 500 for a more stable fitness signal.

---

## Success criteria

The example is considered successful (ready to ship) if:

1. The evolved program beats First Fit on the test set (different
   seeds from training). This is a low bar that proves the framework
   works on this problem class.

2. Ideally: the evolved program matches or beats Best Fit on the
   test set. This replicates FunSearch's baseline result using
   classical GP operators alone (no LLM).

3. The evolved program is human-readable — print the best genome's
   body as Julia source code so users can see what was discovered.

If criterion 2 is not met within 500 generations, document the
result honestly and note that the LLM operator is expected to help
(this motivates the LLM operator section of the paper).

---

## Output flushing

This is a long-running experiment. Add flush(stdout) after every
generation's progress output:

```julia
println("Gen $gen | best=$(round(best, digits=4)) | " *
        "mean=$(round(mean_fit, digits=4)) | " *
        "vs BF=$(round(vs_bestfit, digits=4)) | " *
        "elapsed=$(round(elapsed, digits=1))s")
flush(stdout)
```

---

## File layout

```
examples/
  bin_packing.jl          # main example script, runnable standalone
  bin_packing_results.md  # auto-generated results summary (create after run)
```

The example script should:
1. Define all types and primitives at the top
2. Have a `run_bin_packing(; kwargs...)` function with keyword args
   for all parameters so users can easily adjust
3. Have a `main()` function that calls run_bin_packing with defaults
   and prints a summary
4. Be runnable as: `julia --project examples/bin_packing.jl`
5. Print the best evolved program's source code at the end
6. Save results to bin_packing_results.md

---

## After the run

Whether or not Best Fit is beaten, create bin_packing_results.md
documenting:
- Best fitness achieved
- Comparison to First Fit and Best Fit baselines
- The actual evolved program (pretty-printed Julia source)
- Number of generations run and wall clock time
- Any interesting intermediate programs found during evolution
- Notes on what the evolved program appears to be doing
  (e.g., "discovered a variant of Best Fit with a bias toward
  half-full bins")

This file becomes the basis for the examples/ section of the docs.

---

## Important: loop safety

The bin packing problem is particularly prone to infinite loops
because the evolved program naturally wants to iterate over bins.
Ensure add_loop_checks is applied before evaluation. The default
loop limit (10_000 iterations) is appropriate here since a program
iterating over 200 bins should never exceed a few hundred iterations.

Also guard bp_bin_remaining against out-of-bounds access — clamp
the index to [1, n_bins] and return 0.0f0 for empty or invalid bins.
