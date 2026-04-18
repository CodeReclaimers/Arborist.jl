# Bin Packing Evolution Results

Generated: 2026-04-11T20:24:19.081

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 3353.4s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.8317** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.8382** |

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

    bp_place_in_bin(6)
    bp_place_in_bin(44)
    bp_place_in_bin(82 + 10)
    bp_place_in_bin(82 - -23)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9547 |
| 11 | 1.9061 |
| 21 | 1.9057 |
| 31 | 1.8702 |
| 41 | 1.8626 |
| 51 | 1.8626 |
| 61 | 1.8626 |
| 71 | 1.8423 |
| 81 | 1.8342 |
| 91 | 1.8342 |
| 101 | 1.8322 |
| 111 | 1.8317 |
| 121 | 1.8317 |
| 131 | 1.8317 |
| 141 | 1.8317 |
| 151 | 1.8317 |
| 161 | 1.8317 |
| 171 | 1.8317 |
| 181 | 1.8317 |
| 191 | 1.8317 |
| 200 | 1.8317 |
