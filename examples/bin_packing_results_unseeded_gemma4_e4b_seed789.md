# Bin Packing Evolution Results

Generated: 2026-04-12T03:49:37.660

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 4851.7s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0883 |
| Best Fit  | 1.0764 |
| Worst Fit | 1.1739 |
| **Evolved** | **1.2359** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.0711 |
| **Evolved** | **1.2437** |

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

    67 * 6
    __temp_2 = bp_n_bins()
    bp_place_in_bin(__temp_2 - 6)
    bp_place_in_bin(__temp_2)

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
| 51 | 1.3376 |
| 61 | 1.2404 |
| 71 | 1.2404 |
| 81 | 1.2404 |
| 91 | 1.2404 |
| 101 | 1.2404 |
| 111 | 1.2404 |
| 121 | 1.2404 |
| 131 | 1.2404 |
| 141 | 1.2404 |
| 151 | 1.2388 |
| 161 | 1.2364 |
| 171 | 1.2359 |
| 181 | 1.2359 |
| 191 | 1.2359 |
| 200 | 1.2359 |
