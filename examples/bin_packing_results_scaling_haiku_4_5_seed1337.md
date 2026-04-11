# Bin Packing Evolution Results

Generated: 2026-04-10T10:30:03.926

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2161.0s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.0705** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.0518** |

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

    bp_place_in_bin(90)
    __temp_6 * 1.3663062f0
    __temp_5 = 376.89642f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5
                __temp_2 = __temp_1
                __temp_5 = __temp_6 * 1.3663062f0
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
| 1 | 1.099 |
| 6 | 1.0939 |
| 11 | 1.0889 |
| 16 | 1.074 |
| 21 | 1.0735 |
| 26 | 1.072 |
| 31 | 1.071 |
| 36 | 1.071 |
| 41 | 1.0705 |
| 46 | 1.0705 |
| 51 | 1.0705 |
| 56 | 1.0705 |
| 61 | 1.0705 |
| 66 | 1.0705 |
| 71 | 1.0705 |
| 76 | 1.0705 |
| 81 | 1.0705 |
| 86 | 1.0705 |
| 91 | 1.0705 |
| 96 | 1.0705 |
| 100 | 1.0705 |
