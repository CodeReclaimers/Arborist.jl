# Bin Packing Evolution Results

Generated: 2026-04-10T16:56:20.677

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2901.1s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0883 |
| Best Fit  | 1.0764 |
| Worst Fit | 1.1739 |
| **Evolved** | **1.0774** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.0711 |
| **Evolved** | **1.0668** |

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
    bp_place_in_bin(76 + 7)
    __temp_5 = Float32(147.37446793823815)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 * __temp_6 + __temp_6 < __temp_5
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
| 1 | 1.0964 |
| 6 | 1.0939 |
| 11 | 1.0914 |
| 16 | 1.0904 |
| 21 | 1.0899 |
| 26 | 1.0899 |
| 31 | 1.0899 |
| 36 | 1.0855 |
| 41 | 1.082 |
| 46 | 1.0815 |
| 51 | 1.0805 |
| 56 | 1.0805 |
| 61 | 1.0805 |
| 66 | 1.0805 |
| 71 | 1.0805 |
| 76 | 1.0805 |
| 81 | 1.0784 |
| 86 | 1.0784 |
| 91 | 1.0774 |
| 96 | 1.0774 |
| 100 | 1.0774 |
