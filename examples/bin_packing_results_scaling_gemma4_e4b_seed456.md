# Bin Packing Evolution Results

Generated: 2026-04-10T12:22:36.641

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2846.0s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.081 |
| Best Fit  | 1.0706 |
| Worst Fit | 1.1666 |
| **Evolved** | **1.085** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0852 |
| Best Fit  | 1.0704 |
| **Evolved** | **1.0852** |

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

    bp_place_in_bin(__temp_1)
    while true
        bp_place_in_bin(__temp_1)
        __temp_1 = __temp_1 + Int32(1)
    end
    bp_place_in_bin(__temp_1)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.0906 |
| 6 | 1.089 |
| 11 | 1.0865 |
| 16 | 1.0855 |
| 21 | 1.085 |
| 26 | 1.085 |
| 31 | 1.085 |
| 36 | 1.085 |
| 41 | 1.085 |
| 46 | 1.085 |
| 51 | 1.085 |
| 56 | 1.085 |
| 61 | 1.085 |
| 66 | 1.085 |
| 71 | 1.085 |
| 76 | 1.085 |
| 81 | 1.085 |
| 86 | 1.085 |
| 91 | 1.085 |
| 96 | 1.085 |
| 100 | 1.085 |
