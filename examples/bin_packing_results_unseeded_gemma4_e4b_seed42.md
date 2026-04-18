# Bin Packing Evolution Results

Generated: 2026-04-11T23:41:57.737

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 5181.0s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0841 |
| Best Fit  | 1.0687 |
| Worst Fit | 1.1715 |
| **Evolved** | **1.7773** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0924 |
| Best Fit  | 1.0787 |
| **Evolved** | **1.7819** |

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

    bp_place_in_bin(8)
    bp_place_in_bin(16)
    bp_place_in_bin(83)
    bp_place_in_bin(72 - -31)
    bp_place_in_bin(80 - -31)
    bp_place_in_bin(68 + 74)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9475 |
| 11 | 1.87 |
| 21 | 1.8256 |
| 31 | 1.8106 |
| 41 | 1.8101 |
| 51 | 1.8101 |
| 61 | 1.8063 |
| 71 | 1.8063 |
| 81 | 1.8063 |
| 91 | 1.8062 |
| 101 | 1.786 |
| 111 | 1.7773 |
| 121 | 1.7773 |
| 131 | 1.7773 |
| 141 | 1.7773 |
| 151 | 1.7773 |
| 161 | 1.7773 |
| 171 | 1.7773 |
| 181 | 1.7773 |
| 191 | 1.7773 |
| 200 | 1.7773 |
