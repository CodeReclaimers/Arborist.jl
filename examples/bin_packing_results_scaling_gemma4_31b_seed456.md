# Bin Packing Evolution Results

Generated: 2026-04-11T02:36:33.848

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 10657.2s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.0745** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.0607** |

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
    bp_place_in_bin(68 + __temp_1 * 18)
    __temp_5 = 3.132227f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5
                __temp_2 = __temp_1
                __temp_5 = (__temp_6 + __temp_6) / 1.7652832f0
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
| 6 | 1.0871 |
| 11 | 1.0846 |
| 16 | 1.0835 |
| 21 | 1.0835 |
| 26 | 1.0835 |
| 31 | 1.0835 |
| 36 | 1.0835 |
| 41 | 1.0835 |
| 46 | 1.0835 |
| 51 | 1.079 |
| 56 | 1.079 |
| 61 | 1.079 |
| 66 | 1.0785 |
| 71 | 1.0785 |
| 76 | 1.0785 |
| 81 | 1.078 |
| 86 | 1.078 |
| 91 | 1.076 |
| 96 | 1.075 |
| 100 | 1.0745 |
