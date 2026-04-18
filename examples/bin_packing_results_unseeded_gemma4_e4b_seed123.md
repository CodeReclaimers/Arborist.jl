# Bin Packing Evolution Results

Generated: 2026-04-12T01:05:59.521

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 5041.7s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.1973** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.071 |
| **Evolved** | **1.1877** |

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

    __temp_1 = bp_n_bins()
    bp_place_in_bin(__temp_2 - ((((2 - __temp_1) + (17 + -16)) + (17 + -16)) + (17 + -16)))
    bp_place_in_bin(2 - (((2 - __temp_1) + (17 + -16)) + (17 + -16)))
    bp_place_in_bin(__temp_1)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9423 |
| 11 | 1.3318 |
| 21 | 1.3275 |
| 31 | 1.3275 |
| 41 | 1.2575 |
| 51 | 1.2466 |
| 61 | 1.2466 |
| 71 | 1.2368 |
| 81 | 1.2368 |
| 91 | 1.2368 |
| 101 | 1.2368 |
| 111 | 1.2368 |
| 121 | 1.2368 |
| 131 | 1.2368 |
| 141 | 1.2368 |
| 151 | 1.2368 |
| 161 | 1.2368 |
| 171 | 1.2368 |
| 181 | 1.2368 |
| 191 | 1.2027 |
| 200 | 1.1973 |
