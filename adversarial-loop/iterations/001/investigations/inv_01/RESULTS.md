# Investigation inv_01: Operator Rate Validation in GeneticProgramming

## What Was Tested

Whether the `GeneticProgramming` constructor validates that `crossover_rate + mutation_rate <= 1.0`, and what happens when the sum exceeds 1.0.

## Key Findings

### 1. No validation in the constructor

`src/algorithm.jl` lines 43-66: The `GeneticProgramming` keyword constructor accepts `mutation_rate` and `crossover_rate` without any bounds checking. No `ArgumentError` is thrown if their sum exceeds 1.0. Compare with `IslandModel`, which does validate `async` vs `distributed` (line 112-114) -- so the pattern of constructor validation exists in the codebase but was not applied here.

### 2. Silent effective-rate distortion in `_breed_next_generation!`

`src/solve.jl` lines 87-107: The breeding logic is:

```julia
r = rand(rng)                                    # r in [0, 1)
if r < alg.crossover_rate && idx + 1 <= pop_size # crossover
elseif r < alg.crossover_rate + alg.mutation_rate # mutation
else                                              # reproduction (copy)
end
```

With `crossover_rate=0.8, mutation_rate=0.5`:
- Crossover fires when `r < 0.8` (80% of draws)
- Mutation fires when `0.8 <= r < 1.3`, but since `r < 1.0` always, this is effectively `0.8 <= r < 1.0` (20% of draws)
- Reproduction never fires (the `else` branch is unreachable because `crossover_rate + mutation_rate = 1.3 > 1.0`)

The user-specified `mutation_rate=0.5` silently becomes an effective rate of `1.0 - 0.8 = 0.2`. The reproduction operator is completely eliminated without warning.

### 3. Legacy `evolve!` has the identical pattern

`src/genome/evolution.jl` lines 348-368: Same three-branch structure with `r < crossover_rate`, `r < crossover_rate + mutation_rate`, `else`. Same silent distortion when rates sum to more than 1.0.

### 4. No tests cover this edge case

Searched all test files for rate validation tests. Every existing test uses rates that sum to less than 1.0 (e.g., 0.4+0.2=0.6, 0.3+0.3=0.6, 0.5+0.2=0.7). No test verifies behavior when rates sum above 1.0, and no test checks for a validation error on invalid rates.

## Verdict

**CONFIRMED** -- This is a real defect. The finding is correct on all points:

1. No constructor validation exists.
2. The effective mutation rate becomes `min(mutation_rate, 1.0 - crossover_rate)` when the sum exceeds 1.0, silently ignoring the user's configuration.
3. The reproduction (copy) operator is silently eliminated when rates sum to >= 1.0.

## Severity Assessment

**Low-medium.** The default rates (0.3 + 0.3 = 0.6) are safe, and all existing tests and examples use valid combinations. However, the failure mode is silent misconfiguration with no diagnostic -- a user who sets `crossover_rate=0.7, mutation_rate=0.5` expecting 70/50 split with no reproduction would actually get 70/30/0 and never know. The fix is straightforward: add a validation check in the `GeneticProgramming` constructor (and optionally in `evolve!`) that throws `ArgumentError` when `crossover_rate + mutation_rate > 1.0`.

## Files Examined

- `/home/alan/GenProg.jl/src/algorithm.jl` -- constructor, no validation
- `/home/alan/GenProg.jl/src/solve.jl` lines 78-108 -- `_breed_next_generation!`
- `/home/alan/GenProg.jl/src/genome/evolution.jl` lines 311-369 -- legacy `evolve!`
- `/home/alan/GenProg.jl/test/` -- all test files searched, no rate-sum tests found
