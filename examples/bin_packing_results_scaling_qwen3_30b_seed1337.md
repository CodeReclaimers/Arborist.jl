# Bin Packing Evolution Results

Generated: 2026-04-11T11:21:00.681

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 1911.8s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.0801** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.0564** |

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

    __temp_3 = 51
    bp_place_in_bin(77)
    __temp_5 = Float32(2.0)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 <= __temp_5
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
| 6 | 1.094 |
| 11 | 1.0935 |
| 16 | 1.0925 |
| 21 | 1.0925 |
| 26 | 1.092 |
| 31 | 1.092 |
| 36 | 1.092 |
| 41 | 1.092 |
| 46 | 1.092 |
| 51 | 1.092 |
| 56 | 1.092 |
| 61 | 1.0875 |
| 66 | 1.0875 |
| 71 | 1.0875 |
| 76 | 1.0875 |
| 81 | 1.0875 |
| 86 | 1.0875 |
| 91 | 1.0875 |
| 96 | 1.0875 |
| 100 | 1.0801 |
