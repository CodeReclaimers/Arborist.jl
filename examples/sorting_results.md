# Sorting Algorithm Evolution Results

**Date:** 2026-03-24T23:22:29.427
**Seed:** 42

## Parameters
- pop_size=300, generations=500
- mutation_rate=0.4, crossover_rate=0.3
- elitism=5, tournament_size=5
- bloat_penalty=0.0005
- curriculum: 3 → 20
- upgrade_threshold=0.05
- comparison_alpha_base=0.05
- (adaptive α = 0.05 × max(0, length − 6))

## Results
- **Best fitness:** 0.027667
- **Wall time:** 2357.9s
- **Converged:** false
- **Final curriculum length:** 20
- **Achievement:** Below Tier 1: 0.3% correct on length-8
- **Generalization:** No — performance degrades on longer arrays (possible overfitting to length)

## Evolved Program
```julia
function evolved_sorter()
    done = false
    __temp_1 = 0
    __temp_2 = 0
    __temp_3 = 0
    __temp_4 = 0
    __temp_5 = 0
    __temp_6 = 0
    __temp_7 = 0
    __temp_8 = 0

    __temp_1 - -3
    while true
        __temp_3 = __temp_1 - -1
        __temp_2 = __temp_1 + Int32(1)
        while __temp_2 <= sort_n()
            if sort_less(__temp_2, __temp_3)
                __temp_3 = __temp_2
            end
            __temp_2 = __temp_2 + Int32(1)
        end
        sort_swap!(__temp_1, __temp_3)
        __temp_1 = __temp_1 + Int32(1)
    end

    return done
end
```

## Fitness History (curriculum advances)
- Gen 1: 0.013
- Gen 5: 0.065431
- Gen 6: 0.088579
- Gen 9: 0.072227
- Gen 11: 0.049752
- Gen 500: 0.027667

## Verification
```
Verification results (1000 random arrays per length, unseen seeds):
  Length | Correct |  Comps  |  Swaps  | Comps/n² | Comps/(n·lg n)
  ------+---------+---------+---------+----------+--------------
      3 |    0.6% |     6.0 |     2.7 |    0.667 |        1.262
      4 |    0.2% |    10.0 |     3.7 |    0.625 |         1.25
      5 |    0.5% |    15.0 |     4.8 |      0.6 |        1.292
      6 |    0.3% |    21.0 |     5.8 |    0.583 |        1.354
      7 |    0.2% |    28.0 |     6.8 |    0.571 |        1.425
      8 |    0.3% |    36.0 |     7.9 |    0.562 |          1.5
      9 |    0.1% |    45.0 |     8.9 |    0.556 |        1.577
     10 |    0.1% |    55.0 |     9.9 |     0.55 |        1.656
     11 |    0.2% |    66.0 |    10.9 |    0.545 |        1.734
     12 |    0.6% |    78.0 |    11.9 |    0.542 |        1.813
     13 |    0.4% |    91.0 |    12.9 |    0.538 |        1.892
     14 |    0.3% |   105.0 |    13.9 |    0.536 |         1.97
     15 |    0.5% |   120.0 |    14.9 |    0.533 |        2.048
     16 |    0.1% |   136.0 |    15.9 |    0.531 |        2.125
     17 |    0.4% |   153.0 |    16.9 |    0.529 |        2.202
     18 |    0.4% |   171.0 |    18.0 |    0.528 |        2.278
     19 |    0.6% |   190.0 |    18.9 |    0.526 |        2.354
     20 |    0.3% |   210.0 |    19.9 |    0.525 |        2.429
     21 |    0.3% |   231.0 |    20.9 |    0.524 |        2.504
     22 |    0.2% |   253.0 |    22.0 |    0.523 |        2.579
     23 |    0.2% |   276.0 |    23.0 |    0.522 |        2.653
     24 |    0.5% |   300.0 |    24.0 |    0.521 |        2.726
```

## Notes
This example evolves a sorting algorithm from scratch using genetic
programming with curriculum learning and an adaptive comparison count
penalty. The evolved program accesses the array only through scalar
primitives (sort_get, sort_n, sort_less, sort_swap!) — no built-in
sort function is available.

The comparison penalty α = 0.05 × max(0, n − 6) is zero for
the early curriculum stages (lengths 3–6, correctness only) and
increases linearly as arrays get longer, creating pressure to reduce
comparison count. At length 24, the O(n²) vs O(n log n) difference
is ~5×, which provides meaningful fitness signal.

Sorting is a classic GP benchmark (Koza 1992). Unlike Koza's sorting
networks which used a fixed comparator representation, this example
evolves imperative programs with loops, conditionals, and variable
assignments — making the search space much larger but the evolved
programs more interpretable as conventional sorting algorithms.
