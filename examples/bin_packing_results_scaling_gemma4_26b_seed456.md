# Bin Packing Evolution Results

Generated: 2026-04-10T16:07:59.558

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2793.6s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.0714** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.0628** |

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

    __temp_1 = __temp_2
    bp_place_in_bin(84 - -5)
    __temp_5 = Float32(2.0)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = (__temp_4 - 0.0022032154f0) - bp_item_size()
            if __temp_6 < __temp_5
                __temp_2 = __temp_1
                __temp_5 = __temp_6 * 1.7157867f0
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
| 6 | 1.0856 |
| 11 | 1.0841 |
| 16 | 1.0841 |
| 21 | 1.0841 |
| 26 | 1.081 |
| 31 | 1.081 |
| 36 | 1.081 |
| 41 | 1.0745 |
| 46 | 1.0745 |
| 51 | 1.0745 |
| 56 | 1.0735 |
| 61 | 1.072 |
| 66 | 1.072 |
| 71 | 1.072 |
| 76 | 1.072 |
| 81 | 1.0714 |
| 86 | 1.0714 |
| 91 | 1.0714 |
| 96 | 1.0714 |
| 100 | 1.0714 |
