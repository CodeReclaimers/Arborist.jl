# Bin Packing Evolution Results

Generated: 2026-04-11T09:41:52.545

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2257.0s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.0719** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.071 |
| **Evolved** | **1.0631** |

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
    bp_place_in_bin(__temp_1 - -81)
    __temp_5 = 66.81357f0
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
| 6 | 1.0904 |
| 11 | 1.0874 |
| 16 | 1.0869 |
| 21 | 1.0869 |
| 26 | 1.0868 |
| 31 | 1.075 |
| 36 | 1.074 |
| 41 | 1.0735 |
| 46 | 1.0724 |
| 51 | 1.0719 |
| 56 | 1.0719 |
| 61 | 1.0719 |
| 66 | 1.0719 |
| 71 | 1.0719 |
| 76 | 1.0719 |
| 81 | 1.0719 |
| 86 | 1.0719 |
| 91 | 1.0719 |
| 96 | 1.0719 |
| 100 | 1.0719 |
