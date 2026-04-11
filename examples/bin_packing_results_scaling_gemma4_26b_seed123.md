# Bin Packing Evolution Results

Generated: 2026-04-10T15:21:25.960

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2725.4s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.0815** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.071 |
| **Evolved** | **1.0851** |

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
    __temp_6 = -1.5268667f0
    __temp_5 = 212.09372f0 / 837.4724f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 + __temp_6 < __temp_5
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
| 11 | 1.0874 |
| 16 | 1.0869 |
| 21 | 1.0864 |
| 26 | 1.0864 |
| 31 | 1.0864 |
| 36 | 1.0864 |
| 41 | 1.0863 |
| 46 | 1.0863 |
| 51 | 1.0863 |
| 56 | 1.0863 |
| 61 | 1.0863 |
| 66 | 1.0863 |
| 71 | 1.0863 |
| 76 | 1.0815 |
| 81 | 1.0815 |
| 86 | 1.0815 |
| 91 | 1.0815 |
| 96 | 1.0815 |
| 100 | 1.0815 |
