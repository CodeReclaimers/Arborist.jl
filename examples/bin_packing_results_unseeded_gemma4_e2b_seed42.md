# Bin Packing Evolution Results

Generated: 2026-04-11T18:32:44.037

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 3441.2s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0841 |
| Best Fit  | 1.0687 |
| Worst Fit | 1.1715 |
| **Evolved** | **1.7791** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0924 |
| Best Fit  | 1.0787 |
| **Evolved** | **1.7877** |

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

    bp_place_in_bin(3)
    bp_place_in_bin(34)
    bp_place_in_bin(83)
    bp_place_in_bin(__temp_3 + 44)
    bp_place_in_bin(83 + (9 + 22))
    bp_place_in_bin((83 + (9 + 29)) + 9)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9475 |
| 11 | 1.8755 |
| 21 | 1.8652 |
| 31 | 1.8554 |
| 41 | 1.8396 |
| 51 | 1.8363 |
| 61 | 1.8343 |
| 71 | 1.8103 |
| 81 | 1.8033 |
| 91 | 1.8033 |
| 101 | 1.7837 |
| 111 | 1.7837 |
| 121 | 1.7837 |
| 131 | 1.7791 |
| 141 | 1.7791 |
| 151 | 1.7791 |
| 161 | 1.7791 |
| 171 | 1.7791 |
| 181 | 1.7791 |
| 191 | 1.7791 |
| 200 | 1.7791 |
