# Bin Packing Evolution Results

Generated: 2026-04-11T21:17:34.999

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 3195.9s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0883 |
| Best Fit  | 1.0764 |
| Worst Fit | 1.1739 |
| **Evolved** | **1.2348** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.0711 |
| **Evolved** | **1.2408** |

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

    bp_place_in_bin(__temp_2 + 2)
    __temp_2 = bp_n_bins()
    bp_place_in_bin(__temp_2 - (2 + 1))
    bp_place_in_bin(__temp_2 - 2)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.3416 |
| 11 | 1.3376 |
| 21 | 1.3376 |
| 31 | 1.3376 |
| 41 | 1.3376 |
| 51 | 1.3326 |
| 61 | 1.3326 |
| 71 | 1.3311 |
| 81 | 1.3237 |
| 91 | 1.2446 |
| 101 | 1.2354 |
| 111 | 1.2353 |
| 121 | 1.2353 |
| 131 | 1.2348 |
| 141 | 1.2348 |
| 151 | 1.2348 |
| 161 | 1.2348 |
| 171 | 1.2348 |
| 181 | 1.2348 |
| 191 | 1.2348 |
| 200 | 1.2348 |
