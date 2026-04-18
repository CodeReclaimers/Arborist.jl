# Bin Packing Evolution Results

Generated: 2026-04-12T05:12:13.912

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 4956.2s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.8344** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.874** |

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

    __temp_1 = 8
    bp_place_in_bin(3 + __temp_3)
    bp_place_in_bin(__temp_1 + 22)
    bp_place_in_bin(57)
    bp_place_in_bin(86 + (3 - __temp_3))

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9629 |
| 11 | 1.9544 |
| 21 | 1.9534 |
| 31 | 1.9026 |
| 41 | 1.8816 |
| 51 | 1.8676 |
| 61 | 1.8676 |
| 71 | 1.8676 |
| 81 | 1.8676 |
| 91 | 1.8676 |
| 101 | 1.8676 |
| 111 | 1.8676 |
| 121 | 1.8676 |
| 131 | 1.8676 |
| 141 | 1.8671 |
| 151 | 1.8671 |
| 161 | 1.8671 |
| 171 | 1.8671 |
| 181 | 1.8671 |
| 191 | 1.84 |
| 200 | 1.8344 |
