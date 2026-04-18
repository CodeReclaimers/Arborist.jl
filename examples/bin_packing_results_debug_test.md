# Bin Packing Evolution Results

Generated: 2026-04-12T06:31:30.879

## Configuration
- Population: 50
- Generations: 10
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 96.4s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0841 |
| Best Fit  | 1.0687 |
| Worst Fit | 1.1715 |
| **Evolved** | **1.9413** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0924 |
| Best Fit  | 1.0787 |
| **Evolved** | **1.9251** |

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

    __temp_4 = 304.8653f0
    __temp_3 = 98
    bp_place_in_bin(43)
    __temp_4 = __temp_6

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9493 |
| 2 | 1.9483 |
| 3 | 1.9413 |
| 4 | 1.9413 |
| 5 | 1.9413 |
| 6 | 1.9413 |
| 7 | 1.9413 |
| 8 | 1.9413 |
| 9 | 1.9413 |
| 10 | 1.9413 |
| 10 | 1.9413 |
