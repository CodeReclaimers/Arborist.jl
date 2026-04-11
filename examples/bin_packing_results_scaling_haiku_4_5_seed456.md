# Bin Packing Evolution Results

Generated: 2026-04-10T09:20:03.969

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2089.6s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.08** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.0761** |

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

    __temp_2 = Int32(1)
    bp_place_in_bin(45)
    __temp_5 = 0.41448992f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 - 0.00079379824f0 < __temp_5 * 0.4277123f0
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
| 21 | 1.0836 |
| 26 | 1.0836 |
| 31 | 1.0836 |
| 36 | 1.083 |
| 41 | 1.083 |
| 46 | 1.083 |
| 51 | 1.0815 |
| 56 | 1.0815 |
| 61 | 1.0815 |
| 66 | 1.0815 |
| 71 | 1.0815 |
| 76 | 1.0815 |
| 81 | 1.0815 |
| 86 | 1.081 |
| 91 | 1.081 |
| 96 | 1.081 |
| 100 | 1.08 |
