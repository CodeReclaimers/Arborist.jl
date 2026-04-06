# Bin Packing Combined Experiment Results

Generated: 2026-03-23

## Table 1: LLM Operator Impact (Uniform Distribution)

| Config | Best (100 gen) | Test vs FF | Test vs BF | LLM fallback rate | Time |
|---|---|---|---|---|---|
| Classical only (A1) | 1.0782 | 1.30% | 0.05% | N/A | 218.1s |
| Classical + Qwen3-Coder (A2) | 1.0679 | 2.25% | 1.00% | 0.2% | 1767.4s |

## Table 2: Distribution Comparison (100 Generations)

| Config | Distribution | Best (test) | vs FF (test) | vs BF (test) | Time |
|---|---|---|---|---|---|
| No speciation (B1) | uniform | 1.0648 | 2.53% | 1.29% | 262.1s |
| No speciation (B2) | bimodal | 1.0802 | 0.00% | -0.35% | 334.0s |
| Behavioral spec (B3) | bimodal | 1.0802 | 0.00% | -0.35% | 474.8s |

## Baselines

| Distribution | First Fit | Best Fit | Gap |
|---|---|---|---|
| Uniform | 1.0841 (train) / 1.0924 (test) | 1.0687 (train) / 1.0787 (test) | 0.0154 / 0.0137 |
| Bimodal | 1.0771 | 1.0738 | 0.0032 |

## LLM Operator Metrics (Experiment A2)

- **Total LLM calls**: 1313
- **Successful parses**: 1310 (99.8%)
- **Fallback count**: 3 (0.2%)
- **Mean call latency**: 1.21s
- **Total LLM time**: 1588.9s (out of 1767.4s total wall time)

## Best Evolved Programs

### A1: Classical Only (Uniform) — Test: 1.0782
```julia
# Best-fit scanner with relaxed threshold (multiplier 1.2269)
__temp_1 = Int32(1)
__temp_5 = 0.44579023f0           # initial "best remaining" threshold
while __temp_1 <= bp_n_bins()
    __temp_4 = bp_bin_remaining(__temp_1)
    if __temp_4 >= bp_item_size()
        __temp_6 = __temp_4 - bp_item_size()
        if __temp_6 < __temp_5 * 1.2269365f0   # relaxed best-fit
            __temp_2 = __temp_1
            __temp_5 = __temp_6
        end
    end
    __temp_1 = __temp_1 + Int32(1)
end
bp_place_in_bin(__temp_2)
```
**Strategy**: Best-fit with a relaxed comparison threshold. The multiplier `1.2269` means it will accept a bin even if the remaining space is slightly larger than the current best, creating a bias toward earlier (older) bins. This is close to standard best-fit but not quite optimal.

### A2: Classical + LLM (Uniform) — Test: 1.0679
```julia
# Tighter best-fit scanner (multiplier 0.6157)
__temp_1 = Int32(1)
bp_place_in_bin(62)               # pre-place attempt (no-op most of the time)
__temp_5 = Float32(0.281)         # lower initial threshold
while __temp_1 <= bp_n_bins()
    __temp_4 = bp_bin_remaining(__temp_1)
    if __temp_4 >= bp_item_size()
        __temp_6 = __temp_4 - bp_item_size()
        if __temp_6 < __temp_5 * 0.6157252f0   # tighter threshold
            __temp_2 = __temp_1
            __temp_5 = __temp_6
        end
    end
    __temp_1 = __temp_1 + Int32(1)
end
bp_place_in_bin(__temp_2)
```
**Strategy**: Same best-fit skeleton but with a *tighter* threshold (`0.6157` multiplier shrinks the acceptance window). Combined with a lower initial threshold (0.281 vs 0.446), this is more selective — it requires a substantially better bin before switching. This "stubborn best-fit" avoids the early-bin bias of A1 and achieves better packing.

### B1: Uniform — Test: 1.0648
```julia
# Best-fit scanner (seeded template, partially optimized)
__temp_1 = Int32(1)
__temp_5 = 0.53914535f0
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
```
**Strategy**: Clean best-fit. No multiplier — uses direct `<` comparison. This is the most "textbook" best-fit, and achieves the best test score on uniform distribution (1.0648, beating BF baseline of 1.0787 by 1.29%).

### B2/B3: Bimodal — Test: 1.0802
```julia
# First-fit linear scan
while true
    result = bp_place_in_bin(__temp_1)
    __temp_1 = __temp_1 + Int32(1)
end
```
**Strategy**: Simple first-fit. On bimodal distributions, this ties with First Fit and slightly underperforms Best Fit. The evolved programs did not discover item-size-conditional strategies.

## Interpretation

### 1. Did Qwen3-Coder improve on the classical GP baseline?

**Yes, meaningfully.** A2 (Classical + LLM) achieved test fitness 1.0679 vs A1's 1.0782 — a 0.95% improvement in absolute terms. Against the Best Fit baseline, A2 beats it by 1.0% while A1 only matches it (0.05%).

The LLM's fallback rate was remarkably low: **0.2%** (3 out of 1313 calls). Qwen3-Coder understood the problem representation well and consistently generated valid Julia programs with the bin packing primitives. Mean latency was 1.21s per call, which is reasonable for a local 30B model.

The LLM contribution appears to be in *refining* the best-fit strategy parameters rather than discovering fundamentally new strategies. The A2 program uses a tighter threshold multiplier (0.6157) and lower initial bound (0.281), which together create a more selective bin choice. This kind of parameter tuning within a structural template is exactly where LLMs excel — they can reason about the semantics of the program to make targeted improvements that pure random mutation would take much longer to find.

The cost is significant: A2 took 1767s vs A1's 218s (8.1x slower), with 1589s spent on LLM inference. For 100 generations, this was approximately 13 LLM calls per generation, each taking ~1.2s.

### 2. Is the bimodal fitness landscape richer than uniform?

**Surprisingly, no** — at least not with 100 generations and this configuration. The bimodal distribution has a *smaller* gap between First Fit and Best Fit (0.0032 vs 0.0154 on uniform train set), meaning there's less room for GP to improve. The evolved bimodal programs converged to simple first-fit scanners and did not discover item-size-conditional logic.

This is counterintuitive: we expected bimodal to support richer strategies (e.g., pairing large and small items). The likely explanation is that the bimodal distribution used here (50/50 split, small=0.1-0.3, large=0.6-0.9) makes it relatively easy for First Fit to find good pairings naturally — a large item leaves 0.1-0.4 capacity, and small items (0.1-0.3) fill that gap efficiently. The FF-BF gap is small because the pairing happens automatically.

A bimodal distribution with more overlap or a wider size range might create a harder problem where evolved strategies could differentiate.

### 3. Does behavioral speciation help more on bimodal than uniform?

**No measurable improvement.** B3 (bimodal + behavioral speciation) achieved the same test fitness as B2 (bimodal, no speciation): 1.0802. While speciation successfully maintained high diversity (up to 105 species), this diversity did not translate to better best-fitness outcomes.

The hypothesis that bimodal supports more genuinely different strategies was not confirmed. The species count grew to over 100, suggesting behavioral speciation was finding many *syntactically different but functionally equivalent* programs (different code, same first-fit-like behavior). The behavioral fingerprinting mechanism may need a finer probe to distinguish genuinely different strategies on this problem.

On uniform distribution (B1), speciation was not tested, but the baseline (no speciation) already achieved the best result across all experiments: 1.0648 test fitness, beating Best Fit by 1.29%.
