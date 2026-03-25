# Sorting Algorithm Evolution Results

**Date:** 2026-03-24T20:52:34.277
**Seed:** 123

## Parameters
- pop_size=50, generations=30
- mutation_rate=0.4, crossover_rate=0.3
- elitism=5, tournament_size=5
- bloat_penalty=0.0005
- curriculum: 3 → 6
- upgrade_threshold=0.05

## Results
- **Best fitness:** 0.0115
- **Wall time:** 16.8s
- **Converged:** false
- **Final curriculum length:** 6
- **Achievement:** Tier 2 (target): 100% correct on length-8
- **Generalization:** Yes — program generalizes beyond training distribution

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

    __temp_1 = __temp_4
    while __temp_1 < sort_n()
        __temp_3 = __temp_1
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
- Gen 30: 0.0115

## Verification
```
Verification results (1000 random arrays per length, unseen seeds):
  Length 3: 100.0% correct
  Length 4: 100.0% correct
  Length 5: 100.0% correct
  Length 6: 100.0% correct
  Length 7: 100.0% correct
  Length 8: 100.0% correct
  Length 9: 100.0% correct
  Length 10: 100.0% correct
  Length 11: 100.0% correct
  Length 12: 100.0% correct
```

## Notes
This example evolves a sorting algorithm from scratch using genetic
programming with curriculum learning. The evolved program accesses the
array only through scalar primitives (sort_get, sort_n, sort_less,
sort_swap!) — no built-in sort function is available.

Sorting is a classic GP benchmark (Koza 1992). Unlike Koza's sorting
networks which used a fixed comparator representation, this example
evolves imperative programs with loops, conditionals, and variable
assignments — making the search space much larger but the evolved
programs more interpretable as conventional sorting algorithms.
