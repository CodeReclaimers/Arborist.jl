# Bin Packing Evolution Results

Generated: 2026-04-10T17:44:26.310

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2885.6s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.0907** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.0671** |

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
    88 * __temp_1
    __temp_5 = 1.128975f0 / 1.151722f0 - 0.81092536f0
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
| 1 | 1.099 |
| 6 | 1.0955 |
| 11 | 1.094 |
| 16 | 1.093 |
| 21 | 1.0925 |
| 26 | 1.0925 |
| 31 | 1.0925 |
| 36 | 1.0925 |
| 41 | 1.0925 |
| 46 | 1.092 |
| 51 | 1.092 |
| 56 | 1.092 |
| 61 | 1.0919 |
| 66 | 1.0907 |
| 71 | 1.0907 |
| 76 | 1.0907 |
| 81 | 1.0907 |
| 86 | 1.0907 |
| 91 | 1.0907 |
| 96 | 1.0907 |
| 100 | 1.0907 |
