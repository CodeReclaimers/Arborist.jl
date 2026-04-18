# Bin Packing Evolution Results

Generated: 2026-04-11T10:14:51.260

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 1978.7s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.084** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.0714** |

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
    1.903506f0 / __temp_4
    __temp_5 = Float32(2.0)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5 - __temp_6
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
| 1 | 1.0906 |
| 6 | 1.0851 |
| 11 | 1.0841 |
| 16 | 1.0841 |
| 21 | 1.0841 |
| 26 | 1.0841 |
| 31 | 1.0841 |
| 36 | 1.0841 |
| 41 | 1.0841 |
| 46 | 1.0841 |
| 51 | 1.0841 |
| 56 | 1.0841 |
| 61 | 1.084 |
| 66 | 1.084 |
| 71 | 1.084 |
| 76 | 1.084 |
| 81 | 1.084 |
| 86 | 1.084 |
| 91 | 1.084 |
| 96 | 1.084 |
| 100 | 1.084 |
