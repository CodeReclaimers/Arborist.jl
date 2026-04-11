# Bin Packing Evolution Results

Generated: 2026-04-10T11:35:10.584

## Configuration
- Population: 200
- Generations: 100
- Items per episode: 200
- Episodes: 20
- Distribution: uniform
- Wall time: 2562.4s

## Training Set Results
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0862 |
| Best Fit  | 1.0734 |
| Worst Fit | 1.1731 |
| **Evolved** | **1.0808** |

## Test Set Results (different seeds)
| Heuristic | Normalized bins |
|-----------|----------------|
| First Fit | 1.0854 |
| Best Fit  | 1.071 |
| **Evolved** | **1.081** |

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

    __temp_1 = Int32(1)
    __temp_6 = __temp_5
    __temp_5 = Float32(0.15320625127538925)
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5
                __temp_2 = __temp_1
                __temp_5 = __temp_6
            end
        end
        __temp_1 = __temp_1 + Int32(1)
    end
    bp_place_in_bin(__temp_2)

    return result
end

```

## Fitness History (sampled)
| Generation | Best Fitness |
|-----------|-------------|
| 1 | 1.0934 |
| 6 | 1.0879 |
| 11 | 1.0838 |
| 16 | 1.0818 |
| 21 | 1.0808 |
| 26 | 1.0808 |
| 31 | 1.0808 |
| 36 | 1.0808 |
| 41 | 1.0808 |
| 46 | 1.0808 |
| 51 | 1.0808 |
| 56 | 1.0808 |
| 61 | 1.0808 |
| 66 | 1.0808 |
| 71 | 1.0808 |
| 76 | 1.0808 |
| 81 | 1.0808 |
| 86 | 1.0808 |
| 91 | 1.0808 |
| 96 | 1.0808 |
| 100 | 1.0808 |
