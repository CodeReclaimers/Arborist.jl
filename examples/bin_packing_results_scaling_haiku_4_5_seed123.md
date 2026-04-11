# Bin Packing Evolution Results

Generated: 2026-04-10T08:45:14.338

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2181.9s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.0768** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.071 |
| **Evolved** | **1.0823** |

## Best Evolved Program
```julia
function evolved_heuristic()
    result = false
    __temp_1 = 0
    __temp_2 = 0
    __temp_3 = 0
    __temp_4 = 0.0
    __temp_5 = 0.0
    __temp_6 = 0.0

    __temp_1 = Int32(1)
    bp_place_in_bin(30 + 24)
    __temp_5 = Float32(0.1437102375672318)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5
                __temp_2 = __temp_1
                __temp_5 = __temp_6
            end
        end
        __temp_1 = __temp_1 + Int32(1)
    end
    bp_place_in_bin(__temp_2)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.0934 |
| 6 | 1.0909 |
| 11 | 1.0879 |
| 16 | 1.0844 |
| 21 | 1.0839 |
| 26 | 1.0834 |
| 31 | 1.0799 |
| 36 | 1.0799 |
| 41 | 1.0799 |
| 46 | 1.0773 |
| 51 | 1.0768 |
| 56 | 1.0768 |
| 61 | 1.0768 |
| 66 | 1.0768 |
| 71 | 1.0768 |
| 76 | 1.0768 |
| 81 | 1.0768 |
| 86 | 1.0768 |
| 91 | 1.0768 |
| 96 | 1.0768 |
| 100 | 1.0768 |
