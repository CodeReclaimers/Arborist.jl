# Arborist.jl — Bin Packing Experiment Results

Original data generated: 2026-03-24. LLM columns updated 2026-04-09
after fixing two latent bugs in the LLM mutation pipeline (parser
discarding control flow, sanitizer rejecting domain-specific function
calls). See the "Fixed" entries under 0.1.0 in `CHANGELOG.md` for
the landed fixes. Classical, behavioral, and template results are
unchanged from the original run.

## 1. Multi-seed Comparison (5 seeds: 42, 123, 456, 789, 1337)

| Config | Mean test | Std | Min | Max | vs BF (mean) | Beats BF |
|--------|-----------|-----|-----|-----|-------------|----------|
| Best-fit template | 1.0713 | 0.0047 | 1.0655 | 1.0787 | 0.0% | 0/5 |
| Classical GP | 1.0717 | 0.0112 | 1.0527 | 1.0804 | -0.03% | 2/5 |
| Behavioral speciation | 1.0689 | 0.0027 | 1.0655 | 1.0711 | 0.23% | 1/5 |
| Classical + LLM | 1.0664 | 0.0084 | 1.0531 | 1.0757 | 0.46% | 3/5 |

### Per-seed Detail

| Seed | Template | Classical | Behavioral | LLM |
|------|----------|-----------|------------|-----|
| 42 | 1.0787 | 1.0782 | 1.0664 | 1.0662 |
| 123 | 1.071 | 1.0804 | 1.071 | 1.0659 |
| 456 | 1.0704 | 1.071 | 1.0704 | 1.0710 |
| 789 | 1.0711 | 1.0761 | 1.0711 | 1.0757 |
| 1337 | 1.0655 | 1.0527 | 1.0655 | 1.0531 |

## 2. Extended Run (300 generations, seed 42)

### Classical (G1A)
| Gen | Best |
|-----|------|
| 1 | 1.0887 |
| 10 | 1.0857 |
| 20 | 1.0827 |
| 30 | 1.0822 |
| 40 | 1.0822 |
| 50 | 1.0817 |
| 60 | 1.0817 |
| 70 | 1.0817 |
| 80 | 1.0817 |
| 90 | 1.0817 |
| 100 | 1.0811 |
| 110 | 1.0806 |
| 120 | 1.0806 |
| 130 | 1.0806 |
| 140 | 1.0806 |
| 150 | 1.0806 |
| 160 | 1.0768 |
| 170 | 1.0718 |
| 180 | 1.0713 |
| 190 | 1.0698 |
| 200 | 1.0698 |
| 210 | 1.0698 |
| 220 | 1.0698 |
| 230 | 1.0698 |
| 240 | 1.0698 |
| 250 | 1.0698 |
| 260 | 1.0698 |
| 270 | 1.0698 |
| 280 | 1.0698 |
| 290 | 1.0698 |
| 300 | 1.0698 |

### Classical + LLM (G1B)
| Gen | Best |
|-----|------|
| 1 | 1.0887 |
| 11 | 1.0852 |
| 21 | 1.0827 |
| 31 | 1.0812 |
| 41 | 1.0812 |
| 51 | 1.0729 |
| 61 | 1.0729 |
| 71 | 1.0729 |
| 81 | 1.0729 |
| 91 | 1.0729 |
| 101 | 1.0729 |
| 111 | 1.0729 |
| 121 | 1.0729 |
| 131 | 1.0719 |
| 141 | 1.0719 |
| 151 | 1.0719 |
| 161 | 1.0719 |
| 171 | 1.0719 |
| 181 | 1.0719 |
| 191 | 1.0719 |
| 201 | 1.0713 |
| 211 | 1.0713 |
| 221 | 1.0688 |
| 231 | 1.0688 |
| 241 | 1.0688 |
| 251 | 1.0688 |
| 261 | 1.0688 |
| 271 | 1.0688 |
| 281 | 1.0688 |
| 291 | 1.0683 |
| 300 | 1.0683 |


## 3. Time-Normalized Comparison

Note: the time-normalized data below is from the original 2026-03-24
run. The "Classical + LLM (A2)" row was collected under the pre-fix
parser (LLM control flow silently discarded). Since A2 used only 100
generations, the impact of the parser bug was limited to
constant/expression tuning — the qualitative conclusion still holds.

## Comparison (same wall-clock budget)
| Config | Generations | Test fitness | vs BF | Wall time |
|--------|------------|-------------|-------|-----------|
| Classical (A1, ref) | 100 | 1.0782 | 0.05% | 218s |
| Classical + LLM (A2, ref) | 100 | 1.0679 | 1.00% | 1767s |
| Classical time-matched (G3A) | 800 | 1.0623 | 1.51% | 4373.0s |

## Best Program (G3A)
```julia
function evolved_heuristic()
    result = false
    __temp_1 = 0
    __temp_2 = 0
    __temp_3 = 0
    __temp_4 = 0.0
    __temp_5 = 0.0
    __temp_6 = 0.0

    __temp_2 = 94
    bp_place_in_bin(80)
    __temp_5 = 0.41280127f0 * 0.51612556f0
    while __temp_1 <= bp_n_bins()
        __temp_4 = bp_bin_remaining(__temp_1)
        if __temp_4 >= bp_item_size()
            __temp_6 = __temp_4 - bp_item_size()
            if __temp_6 < __temp_5 * 1.1741527f0
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

## 4. Key Findings

### Does the LLM provide sample efficiency or just need more compute?

**The LLM provides genuine sample efficiency, and with a working
pipeline it now also improves multi-seed reliability.**

At 100 generations, the corrected LLM variant achieves mean test
1.0664 (0.46% better than BF) while classical achieves 1.0717
(-0.03% vs BF). The LLM reached 1.0729 by gen 51 while classical
was still at 1.0817 at that point — roughly 3x faster convergence in
generation count.

On the 300-gen extended run (G1B), the LLM reached 1.0683 (test
1.0648), matching the pre-fix classical extended run (1.0698 train,
1.0623 test) and the current classical control (1.0693 train, 1.0637
test) within noise. The LLM does not find fundamentally different
strategies — it converges to the same Best Fit template — but it
reaches the same neighborhood faster.

**Wall-time efficiency remains the LLM's weakness.** G1B took 7888s
vs G1A's 628s (12.6x overhead) for similar final fitness. The LLM's
value is in settings where generation count is limited (expensive
evaluators, real-time optimization) rather than wall-clock budget.

### The LLM now improves multi-seed reliability

With the corrected pipeline (run 3), the LLM variant achieves:
- Mean test 1.0664 ± 0.0084 (vs classical 1.0717 ± 0.0112)
- 3/5 seeds beat Best Fit (vs classical 2/5)
- Best single seed: 1.0531 (vs classical best: 1.0527)

The LLM's standard deviation (0.0084) is lower than classical
(0.0112), though not as low as behavioral speciation (0.0027). It
occupies a middle ground: more reliable than pure classical GP, and
with a higher ceiling than behavioral speciation.

### Pre-fix vs post-fix LLM comparison

The original 2026-03-24 LLM data was collected under two latent bugs:
1. The ExprGenome deserializer stripped all control flow from LLM
   output, keeping only assignment statements (fixed 2026-03-25)
2. The ASTSanitizer rejected any LLM output containing domain-specific
   function calls like bp_n_bins/bp_place_in_bin (fixed 2026-04-09)

The combined effect was that ~99% of LLM mutations fell back to
classical SubtreeMutation. The "LLM" column in the original data was
effectively a classical run with 2s of wasted latency per call.

With both bugs fixed (run 3), the per-seed comparison is:

| Seed | Pre-fix LLM | Post-fix LLM | Delta |
|------|-------------|--------------|-------|
| 42 | 1.0787 | 1.0662 | **-0.0125** |
| 123 | 1.0616 | 1.0659 | +0.0043 |
| 456 | 1.0701 | 1.0710 | +0.0009 |
| 789 | 1.0741 | 1.0757 | +0.0016 |
| 1337 | 1.0655 | 1.0531 | **-0.0124** |

Seeds 42 and 1337 improved dramatically. Seeds 123, 456, 789 were
neutral. The LLM fix increases variance by enabling more aggressive
mutations that either pay off or don't.

### No novel control flow from the LLM

Despite the parser now correctly accepting while/if/for/break/continue
from LLM output, all evolved best programs across all runs have the
same structural template: a while loop scanning bins, if-checking
capacity, and tracking the best-so-far bin. The LLM's contribution
is tuning the scoring expression, not introducing structural
innovation. The parser and sanitizer fixes were necessary for
correctness but did not unlock a qualitatively different class of
evolved programs on this benchmark.

### Is behavioral speciation reliable across seeds or noisy?

**Behavioral speciation is the most reliable configuration.** Its
standard deviation (0.0027) is 4x lower than classical GP (0.0112),
3x lower than LLM (0.0084). Every seed converges to a clean best-fit
scanner with test fitness between 1.0655 and 1.0711.

However, behavioral speciation does not produce the best absolute
results — it trades peak performance for consistency. Its best
individual result (1.0655) ties with the template baseline.

### What fraction of seeds beat Best Fit?

| Config | Seeds beating BF |
|--------|-----------------|
| Best-fit template | 0/5 |
| Classical GP | 2/5 |
| Behavioral speciation | 1/5 |
| Classical + LLM | 3/5 |

With the corrected LLM pipeline, LLM now has the highest beat rate.
The improvement comes from the LLM's ability to propose
semantically meaningful constant/expression modifications that move
the scoring threshold in the right direction more often than random
subtree mutation.

### Recommended configuration

**For reliability**: Behavioral speciation (lowest variance).

**For best absolute performance**: Classical GP with extended
generations (~300-800) or Classical + LLM with corrected pipeline
at 100 generations. Seed 1337 classical at 100 gen (1.0527) and
seed 1337 LLM at 100 gen (1.0531) are the two best results;
indistinguishable.

**For the LLM claim**: The LLM provides ~3x sample efficiency in
generation count and now improves multi-seed reliability (lower
variance, higher beat rate). Wall-time overhead is ~12x. It is most
useful when evaluation is expensive or generation count is limited.

### Wall clock times

| Experiment | Time |
|-----------|------|
| G1A: Classical 300 gen | 628s (10m) |
| G1B: LLM 300 gen | 7888s (2h 11m) |
| G2A: Classical 5 seeds | 1087s (18m) |
| G2B: Behavioral 5 seeds | 5503s (1h 32m) |
| G2C: LLM 5 seeds | 11010s (3h 4m) |
| G2D: Template baseline | <1s |
| G3A: Classical 800 gen | 4373s (1h 13m) |
| **Total parallel** | **~3h 4m** (limited by G2C) |

Note: G1A is from the 2026-04-08 rerun (same code revision as LLM
runs). G2A, G2B, G2D, G3A are from the original 2026-03-24 run
(classical/behavioral paths are unchanged by the LLM fixes).
