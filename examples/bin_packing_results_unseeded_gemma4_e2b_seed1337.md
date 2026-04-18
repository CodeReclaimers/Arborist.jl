# Bin Packing Evolution Results

Generated: 2026-04-11T22:15:27.610

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 3472.6s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0895 |
| Best Fit  | 1.079 |
| Worst Fit | 1.1829 |
| **Evolved** | **1.8186** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0781 |
| Best Fit  | 1.0655 |
| **Evolved** | **1.8473** |

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

    bp_place_in_bin(22)
    bp_place_in_bin(__temp_1 + 47)
    begin
        bp_place_in_bin((47 + 25) + -42)
        bp_place_in_bin(77)
    end
    __temp_1 = ((((47 + 25) - -44) + -42) - -44) + -42
    bp_place_in_bin(__temp_1 + 47)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9629 |
| 11 | 1.9544 |
| 21 | 1.9534 |
| 31 | 1.9171 |
| 41 | 1.9061 |
| 51 | 1.8529 |
| 61 | 1.8523 |
| 71 | 1.8478 |
| 81 | 1.8478 |
| 91 | 1.8478 |
| 101 | 1.8478 |
| 111 | 1.8458 |
| 121 | 1.8458 |
| 131 | 1.8458 |
| 141 | 1.8452 |
| 151 | 1.8399 |
| 161 | 1.8399 |
| 171 | 1.8399 |
| 181 | 1.8399 |
| 191 | 1.8399 |
| 200 | 1.8283 |
