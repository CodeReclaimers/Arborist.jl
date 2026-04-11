# Bin Packing Evolution Results

Generated: 2026-04-10T09:59:51.947

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 4351.3s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.0868** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.071 |
| **Evolved** | **1.0724** |

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
    Int32(0)
    __temp_5 = Float32(2.0)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5 / 2.2177606f0
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
| 6 | 1.0884 |
| 11 | 1.0879 |
| 16 | 1.0869 |
| 21 | 1.0868 |
| 26 | 1.0868 |
| 31 | 1.0868 |
| 36 | 1.0868 |
| 41 | 1.0868 |
| 46 | 1.0868 |
| 51 | 1.0868 |
| 56 | 1.0868 |
| 61 | 1.0868 |
| 66 | 1.0868 |
| 71 | 1.0868 |
| 76 | 1.0868 |
| 81 | 1.0868 |
| 86 | 1.0868 |
| 91 | 1.0868 |
| 96 | 1.0868 |
| 100 | 1.0868 |
