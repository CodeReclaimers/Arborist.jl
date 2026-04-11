# Bin Packing Evolution Results

Generated: 2026-04-10T08:08:52.396

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 1978.4s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0841 |
| Best Fit  | 1.0687 |
| Worst Fit | 1.1715 |
| **Evolved** | **1.0752** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0924 |
| Best Fit  | 1.0787 |
| **Evolved** | **1.0653** |

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

    __temp_5 = 22.714495f0
    bp_place_in_bin(71)
    Float32(2.0)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 / (0.5519032f0 - (-0.6282779f0 - __temp_6)) < __temp_5
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
| 1 | 1.0887 |
| 6 | 1.0847 |
| 11 | 1.0832 |
| 16 | 1.0827 |
| 21 | 1.0822 |
| 26 | 1.0821 |
| 31 | 1.0821 |
| 36 | 1.0821 |
| 41 | 1.0821 |
| 46 | 1.0821 |
| 51 | 1.0821 |
| 56 | 1.0821 |
| 61 | 1.0821 |
| 66 | 1.0821 |
| 71 | 1.0797 |
| 76 | 1.0797 |
| 81 | 1.0792 |
| 86 | 1.0782 |
| 91 | 1.0767 |
| 96 | 1.0752 |
| 100 | 1.0752 |
