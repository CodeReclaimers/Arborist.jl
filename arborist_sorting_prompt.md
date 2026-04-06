# Arborist.jl — List Sorting Example

You are implementing a built-in example for Arborist.jl that
demonstrates evolving a sorting algorithm using the simulator
pattern with ExprGenome. Read the existing src/, ext/, and test/
directories in full before writing anything, with particular
attention to:
- src/genome/ant_genome.jl (the simulator pattern to follow)
- src/genome/codegen.jl (FunctionSet, GenState, FunctionDetails)
- src/genome/evolution.jl (evaluate_individual!, add_loop_checks)
- test/benchmarks/ant_trail.jl (how benchmarks are structured)

This example must work as both a standalone runnable script and as
a registered example in examples/sorting.jl.

---

## Background

Sorting algorithm synthesis is a classic control-flow GP problem.
The goal is to evolve a program that correctly sorts an array of
integers using only scalar array-access primitives — no built-in
sort function. A successful evolved program will independently
discover something resembling bubble sort, insertion sort, or
selection sort.

This is a compelling demonstration because:
1. The problem requires genuine nested control flow (loops within
   loops, conditional swaps)
2. The evolved program is human-readable and universally understood
3. It has a long history in GP research (Koza 1992, section on
   sorting networks)
4. Success is unambiguous — either the array is sorted or it isn't

---

## Problem specification

### Array simulator

Implement in examples/sorting.jl as module-level state following
the AntGenome pattern. The array being sorted is hidden in a
module-level Ref; the evolved program accesses it only through
scalar primitives:

```julia
mutable struct SortState
    arr::Vector{Int32}     # the array being sorted
    n::Int32               # array length
    comparisons::Int32     # number of comparisons made (for analysis)
    swaps::Int32           # number of swaps made (for analysis)
    op_count::Int32        # total operations (for bloat analysis)
end

const _sort_state = Ref{SortState}(
    SortState(Int32[], Int32(0), Int32(0), Int32(0), Int32(0))
)
```

Expose these scalar primitives to the function set:

```julia
# Query primitives
sort_get(i::Int32)::Int32      # get element at index i (1-indexed, clamped)
sort_n()::Int32                # array length
sort_less(i::Int32, j::Int32)::Bool  # arr[i] < arr[j] (clamped indices)

# Mutation primitives
sort_swap!(i::Int32, j::Int32)::Bool  # swap elements at i and j
                                       # returns true if i != j and both valid
                                       # increments swaps counter

# Arithmetic (in function set)
# Standard +, -, on Int32 for index arithmetic
```

All index operations must clamp to [1, n] — never throw on
out-of-bounds access. The evolved program should never crash
due to an invalid index.

### Key design decision: no set! primitive

Notice there is no `sort_set!` primitive. This forces the evolved
program to use swaps rather than arbitrary writes, which:
1. Preserves the multiset property (no elements created or destroyed)
2. Dramatically reduces the search space
3. Makes the correctness check simple (verify sorted order)
4. Matches how real comparison-based sorting algorithms work

### Evolved program structure

The evolved program is an `ExprGenome` with:
- Input variables: none (array accessed via primitives)
- Output variables: `done::Bool` (unused but required by harness)
- Temp variables: 6-8 Int32 variables for loop counters and
  temporary indices (i, j, k, min_idx, etc.)
- Function set: sort_get, sort_n, sort_less, sort_swap!, plus
  standard +, -, Int32 arithmetic and Bool comparisons

A well-evolved program will discover something like:

```julia
# Bubble sort variant:
i = Int32(1)
while i < sort_n()
    j = Int32(1)
    while j <= sort_n() - i
        if sort_less(j + Int32(1), j)
            sort_swap!(j, j + Int32(1))
        end
        j = j + Int32(1)
    end
    i = i + Int32(1)
end

# Or selection sort variant:
i = Int32(1)
while i < sort_n()
    min_idx = i
    j = i + Int32(1)
    while j <= sort_n()
        if sort_less(j, min_idx)
            min_idx = j
        end
        j = j + Int32(1)
    end
    sort_swap!(i, min_idx)
    i = i + Int32(1)
end
```

### Fitness function

```julia
struct SortingEvaluator <: AbstractEvaluator
    n_episodes::Int        # number of test arrays per evaluation (default 30)
    array_length::Int      # length of each test array (default 8)
    value_range::Int       # values in [-value_range, value_range] (default 100)
    rng_seed::Int          # for reproducible test arrays (default 42)
    partial_credit::Bool   # whether to give partial credit (default true)
end
```

**Start small: array_length = 8.** This is critical. Sorting 8
elements requires O(n²) comparisons at most 64 operations per pass,
which is tractable for GP. Do not start with larger arrays — the
fitness landscape is much harder and evolution will struggle to find
any gradient.

**Fitness calculation** (lower is better):

Primary metric: fraction of element pairs that are out of order
(inversion count normalized by maximum possible inversions):

```julia
function inversion_count(arr::Vector{Int32})::Float64
    n = length(arr)
    n == 0 && return 0.0
    count = 0
    for i in 1:n-1
        for j in i+1:n
            arr[i] > arr[j] && (count += 1)
        end
    end
    return count / (n * (n-1) / 2)
end
```

Fitness = mean inversion count across all episodes.
Perfect sort = 0.0. Random permutation ≈ 0.5.

If partial_credit=false: fitness = fraction of episodes where
array is NOT perfectly sorted (binary, no gradient).
Use partial_credit=true — the inversion count provides gradient
signal that helps evolution find intermediate solutions.

**Optional secondary metric**: penalize excessive operations
(bloat penalty on op_count / (n * log2(n))).

### Curriculum learning — IMPORTANT

Sorting is hard to evolve from scratch on length-8 arrays. Use a
curriculum: start with very short arrays and increase length as
the population improves.

```julia
mutable struct CurriculumSortingEvaluator <: AbstractEvaluator
    current_length::Int    # starts at 3, increases to target_length
    target_length::Int     # final array length (default 8)
    n_episodes::Int        # episodes per evaluation
    rng_seed::Int
    upgrade_threshold::Float64  # fitness level to trigger curriculum advance
                                # (default 0.05 — nearly perfect on current size)
end
```

Curriculum schedule:
- Length 3: trivial (3 comparisons maximum), used to bootstrap
- Length 4: 6 comparisons maximum
- Length 5: 10 comparisons maximum
- Length 6, 7, 8: progressively harder

Advance the curriculum when the best individual's fitness drops
below upgrade_threshold on the current length. Log the advancement:

```julia
println("Curriculum advance: length $old_length -> $new_length at gen $gen")
flush(stdout)
```

This is the single most important implementation decision for this
example. Without curriculum learning, evolution on length-8 arrays
typically fails to find any gradient for hundreds of generations.

---

## Evolution parameters

```julia
algorithm = GeneticProgramming(
    pop_size        = 300,
    generations     = 1000,
    mutation_rate   = 0.4,
    crossover_rate  = 0.3,
    elitism         = 5,
    tournament_size = 5,
    max_depth       = 12,      # deeper trees needed for nested loops
    bloat_penalty   = 0.0005,
    mutation_ops    = [SubtreeMutation(), PointMutation(),
                       HoistMutation(), ExpansionMutation()],
)
```

The higher pop_size and generation count reflect that sorting is
harder than bin packing. 1000 generations may not be enough for
length-8 without curriculum learning; with curriculum learning,
500 generations is often sufficient.

---

## Success criteria

The example is considered successful (ready to ship) if:

**Tier 1 (minimum):** Evolves a program that correctly sorts at
least 90% of random length-8 arrays. This demonstrates that the
framework can evolve non-trivial control flow.

**Tier 2 (target):** Evolves a program that correctly sorts 100%
of random length-8 arrays. Print the evolved program and annotate
what sorting algorithm it appears to have discovered.

**Tier 3 (stretch):** After achieving Tier 2, test the evolved
program (without re-evolution) on length-10 and length-12 arrays
to see if it generalizes beyond its training distribution. A
program that genuinely discovered bubble sort will generalize;
one that overfit to length-8 will not.

Document whatever tier is achieved honestly. Even Tier 1 is a
meaningful result for the paper.

---

## Output flushing

This is a long-running experiment. Flush after every generation:

```julia
println("Gen $gen/$total | " *
        "best=$(round(best, digits=4)) | " *
        "mean=$(round(mean_fit, digits=4)) | " *
        "curriculum_len=$current_length | " *
        "elapsed=$(round(elapsed, digits=1))s")
flush(stdout)
```

Also flush when printing the final evolved program.

---

## Verification

After evolution, verify the best evolved program rigorously:

```julia
function verify_sorter(genome, n_tests=1000, max_length=8;
                       rng=Random.MersenneTwister(999))
    # Test on arrays the program was NOT trained on
    # (different seed from training)
    results = Dict{Int, Float64}()
    for len in 3:max_length
        correct = 0
        for _ in 1:n_tests
            arr = rand(rng, Int32(-100):Int32(100), len)
            sorted_arr = run_genome_on_array(genome, arr)
            issorted(sorted_arr) && (correct += 1)
        end
        results[len] = correct / n_tests
    end
    return results
end
```

Print the verification table:
```
Verification results (1000 random arrays per length, unseen seeds):
  Length 3: 100.0% correct
  Length 4: 100.0% correct
  Length 5:  98.7% correct
  Length 6:  94.3% correct
  Length 7:  87.1% correct
  Length 8:  79.4% correct
```

This table tells the story of whether the program genuinely learned
to sort or just overfit to training conditions.

---

## File layout

```
examples/
  sorting.jl              # main example script, runnable standalone
  sorting_results.md      # auto-generated results summary (create after run)
```

The example script should:
1. Define all types and primitives at the top
2. Have a `run_sorting(; kwargs...)` function with keyword args
   for all parameters
3. Have a `main()` function that runs with defaults and prints summary
4. Be runnable as: `julia --project examples/sorting.jl`
5. Print the best evolved program's source code at the end
6. Print the verification table
7. Save everything to sorting_results.md

---

## After the run

Create sorting_results.md documenting:
- Best fitness achieved at each curriculum stage
- The actual evolved program (pretty-printed Julia source)
- The verification table (accuracy by array length)
- Wall clock time and generations run
- Interpretation: what algorithm did the program discover?
  Does it generalize? Does it terminate correctly?
- Comparison to known GP results on sorting (Koza 1992 evolved
  sorting networks using a different representation — note the
  architectural difference)

---

## Important: loop safety

Sorting programs are the canonical example of evolved infinite
loops — a buggy loop counter that never increments will loop
forever. Ensure add_loop_checks is applied before evaluation.

The appropriate loop limit for sorting: 10 * n² where n is the
array length. For length-8 arrays this is 640 iterations, which
is generous for any O(n²) sort and still catches true infinite loops.

Pass the loop limit as a parameter so it scales with curriculum
advancement:

```julia
loop_limit = 10 * current_length^2
```

---

## Note on generalization

If the evolved program achieves Tier 2 (100% on length-8) but
fails to generalize to longer arrays, that is an interesting
result in itself — it suggests the program overfit to the specific
length rather than discovering a general sorting principle. This
is worth documenting and connects to the broader GP generalization
literature. Don't treat non-generalization as a failure; treat it
as a finding.
