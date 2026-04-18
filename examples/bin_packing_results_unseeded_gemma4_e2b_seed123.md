# Bin Packing Evolution Results

Generated: 2026-04-11T19:28:25.653

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 3341.6s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.1918** |

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
    bp_place_in_bin((__temp_1 + -2) + -3)
    bp_place_in_bin(__temp_1 + -2)
    bp_place_in_bin(__temp_1)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9423 |
| 11 | 1.3318 |
| 21 | 1.3216 |
| 31 | 1.2923 |
| 41 | 1.2079 |
| 51 | 1.2015 |
| 61 | 1.2015 |
| 71 | 1.2015 |
| 81 | 1.2015 |
| 91 | 1.2015 |
| 101 | 1.2015 |
| 111 | 1.2015 |
| 121 | 1.1918 |
| 131 | 1.1918 |
| 141 | 1.1918 |
| 151 | 1.1918 |
| 161 | 1.1918 |
| 171 | 1.1918 |
| 181 | 1.1918 |
| 191 | 1.1918 |
| 200 | 1.1918 |
