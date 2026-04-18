# Bin Packing Evolution Results

Generated: 2026-04-12T02:28:45.894

## Configuration
- Population: 200
- Generations: 200
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 4966.4s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.8266** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.8442** |

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

    bp_place_in_bin(4)
    bp_place_in_bin(13)
    bp_place_in_bin(32 + __temp_1)
    bp_place_in_bin((55 - 7) + 32)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.9547 |
| 11 | 1.9465 |
| 21 | 1.8993 |
| 31 | 1.8984 |
| 41 | 1.8984 |
| 51 | 1.864 |
| 61 | 1.8279 |
| 71 | 1.8279 |
| 81 | 1.8279 |
| 91 | 1.8279 |
| 101 | 1.8279 |
| 111 | 1.8279 |
| 121 | 1.8279 |
| 131 | 1.8279 |
| 141 | 1.8271 |
| 151 | 1.8271 |
| 161 | 1.8271 |
| 171 | 1.8271 |
| 181 | 1.8271 |
| 191 | 1.8266 |
| 200 | 1.8266 |
