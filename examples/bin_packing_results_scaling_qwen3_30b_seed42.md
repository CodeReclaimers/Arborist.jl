# Bin Packing Evolution Results

Generated: 2026-04-11T09:04:15.425

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 1830.6s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0841 |
| Best Fit  | 1.0687 |
| Worst Fit | 1.1715 |
| **Evolved** | **1.0714** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0924 |
| Best Fit  | 1.0787 |
| **Evolved** | **1.0648** |

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
    bp_place_in_bin(82)
    __temp_5 = Float32(0.31723693961141425)
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
| 1 | 1.0887 |
| 6 | 1.0857 |
| 11 | 1.0852 |
| 16 | 1.0827 |
| 21 | 1.0822 |
| 26 | 1.0822 |
| 31 | 1.0822 |
| 36 | 1.0822 |
| 41 | 1.0822 |
| 46 | 1.0822 |
| 51 | 1.0822 |
| 56 | 1.0822 |
| 61 | 1.0817 |
| 66 | 1.0817 |
| 71 | 1.0817 |
| 76 | 1.0817 |
| 81 | 1.0768 |
| 86 | 1.0753 |
| 91 | 1.0743 |
| 96 | 1.0728 |
| 100 | 1.0714 |
