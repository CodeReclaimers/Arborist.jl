# Bin Packing Evolution Results

Generated: 2026-04-10T10:09:47.487

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 1861.8s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.0736** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.0569** |

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

    bp_place_in_bin(90)
    __temp_3 = -15
    __temp_5 = 1.63951f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 * (__temp_4 + (__temp_6 + 0.06809918f0)) < __temp_5
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
| 6 | 1.0939 |
| 11 | 1.0889 |
| 16 | 1.0824 |
| 21 | 1.0809 |
| 26 | 1.0794 |
| 31 | 1.0794 |
| 36 | 1.0755 |
| 41 | 1.0755 |
| 46 | 1.0755 |
| 51 | 1.0755 |
| 56 | 1.0755 |
| 61 | 1.0741 |
| 66 | 1.0741 |
| 71 | 1.0741 |
| 76 | 1.0741 |
| 81 | 1.0736 |
| 86 | 1.0736 |
| 91 | 1.0736 |
| 96 | 1.0736 |
| 100 | 1.0736 |
