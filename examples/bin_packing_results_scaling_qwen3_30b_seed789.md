# Bin Packing Evolution Results

Generated: 2026-04-11T10:49:08.822

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2057.5s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0883 |
| Best Fit  | 1.0764 |
| Worst Fit | 1.1739 |
| **Evolved** | **1.0804** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.0711 |
| **Evolved** | **1.0657** |

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
    bp_place_in_bin(__temp_2 - -78)
    __temp_5 = Float32(2.0)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 * 1.4572573f0 < __temp_5
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
| 6 | 1.0914 |
| 11 | 1.0909 |
| 16 | 1.0899 |
| 21 | 1.0899 |
| 26 | 1.0899 |
| 31 | 1.0899 |
| 36 | 1.0899 |
| 41 | 1.0899 |
| 46 | 1.0899 |
| 51 | 1.0899 |
| 56 | 1.0899 |
| 61 | 1.0869 |
| 66 | 1.0804 |
| 71 | 1.0804 |
| 76 | 1.0804 |
| 81 | 1.0804 |
| 86 | 1.0804 |
| 91 | 1.0804 |
| 96 | 1.0804 |
| 100 | 1.0804 |
