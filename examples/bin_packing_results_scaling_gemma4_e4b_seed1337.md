# Bin Packing Evolution Results

Generated: 2026-04-10T13:48:48.517

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2589.9s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.071** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.0508** |

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
    bp_place_in_bin(88)
    __temp_5 = 34.85626f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5
                __temp_2 = __temp_1
                __temp_5 = __temp_6 / 1.054398f0
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
| 6 | 1.094 |
| 11 | 1.0894 |
| 16 | 1.0889 |
| 21 | 1.0884 |
| 26 | 1.0874 |
| 31 | 1.0874 |
| 36 | 1.0743 |
| 41 | 1.0716 |
| 46 | 1.0716 |
| 51 | 1.0716 |
| 56 | 1.071 |
| 61 | 1.071 |
| 66 | 1.071 |
| 71 | 1.071 |
| 76 | 1.071 |
| 81 | 1.071 |
| 86 | 1.071 |
| 91 | 1.071 |
| 96 | 1.071 |
| 100 | 1.071 |
