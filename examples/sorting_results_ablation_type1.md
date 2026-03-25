# Sorting Algorithm Evolution Results

**Date:** 2026-03-25T07:49:18.889
**Seed:** 123

## Parameters
- pop_size=300, generations=500
- mutation_rate=0.4, crossover_rate=0.3
- elitism=5, tournament_size=5
- bloat_penalty=0.0005
- curriculum: 3 → 8
- upgrade_threshold=0.05
- comparison_alpha_base=0.0
- (adaptive α = 0.0 × max(0, length − 6))

## Results
- **Best fitness:** 0.112556
- **Wall time:** 3202.2s
- **Converged:** false
- **Final curriculum length:** 4
- **Achievement:** Below Tier 1: 0.8% correct on length-8
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

    sort_swap!(sort_get(78), 72)
    while true
        if sort_less(__temp_1 + Int32(1), __temp_1)
            sort_swap!(__temp_1, __temp_1 + Int32(1))
        end
        __temp_1 = __temp_1 + Int32(1)
    end

    return done
end
```

## Fitness History (curriculum advances)
- Gen 1: 0.142333
- Gen 29: 0.1065
- Gen 291: 0.085278
- Gen 299: 0.062556
- Gen 300: 0.041333
- Gen 301: 0.124667
- Gen 305: 0.113056
- Gen 500: 0.112556

## Verification
```
Verification results (1000 random arrays per length, unseen seeds):
  Length | Correct |  Comps  |  Swaps  | Comps/n² | Comps/(n·lg n)
  ------+---------+---------+---------+----------+--------------
      3 |   84.1% |  5760.0 |     1.4 |    640.0 |     1211.385
      4 |   49.5% |  5760.0 |     2.1 |    360.0 |        720.0
      5 |   20.9% |  5760.0 |     2.9 |    230.4 |      496.139
      6 |    7.1% |  5760.0 |     3.7 |    160.0 |      371.379
      7 |    2.5% |  5760.0 |     4.6 |  117.551 |      293.108
      8 |    0.8% |  5760.0 |     5.5 |     90.0 |        240.0
      9 |    0.2% |  5760.0 |     6.3 |   71.111 |      201.898
     10 |    0.0% |  5760.0 |     7.2 |     57.6 |      173.393
     11 |    0.0% |  5760.0 |     8.2 |   47.603 |      151.365
     12 |    0.0% |  5760.0 |     9.0 |     40.0 |      133.893
     13 |    0.0% |  5760.0 |    10.0 |   34.083 |      119.736
     14 |    0.0% |  5760.0 |    10.9 |   29.388 |      108.062
     15 |    0.0% |  5760.0 |    11.9 |     25.6 |       98.288
     16 |    0.0% |  5760.0 |    12.9 |     22.5 |         90.0
     17 |    0.0% |  5760.0 |    13.8 |   19.931 |       82.893
     18 |    0.0% |  5760.0 |    14.7 |   17.778 |        76.74
     19 |    0.0% |  5760.0 |    15.7 |   15.956 |       71.366
     20 |    0.0% |  5760.0 |    16.6 |     14.4 |       66.637
     21 |    0.0% |  5760.0 |    17.5 |   13.061 |       62.447
     22 |    0.0% |  5760.0 |    18.5 |   11.901 |       58.711
     23 |    0.0% |  5760.0 |    19.4 |   10.888 |       55.362
     24 |    0.0% |  5760.0 |    20.4 |     10.0 |       52.345
```

## Notes
This example evolves a sorting algorithm from scratch using genetic
programming with curriculum learning and an adaptive comparison count
penalty. The evolved program accesses the array only through scalar
primitives (sort_get, sort_n, sort_less, sort_swap!) — no built-in
sort function is available.

The comparison penalty α = 0.0 × max(0, n − 6) is zero for
the early curriculum stages (lengths 3–6, correctness only) and
increases linearly as arrays get longer, creating pressure to reduce
comparison count. At length 24, the O(n²) vs O(n log n) difference
is ~5×, which provides meaningful fitness signal.

Sorting is a classic GP benchmark (Koza 1992). Unlike Koza's sorting
networks which used a fixed comparator representation, this example
evolves imperative programs with loops, conditionals, and variable
assignments — making the search space much larger but the evolved
programs more interpretable as conventional sorting algorithms.
