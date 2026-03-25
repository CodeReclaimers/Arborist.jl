# Arborist.jl — Paper Observations and Findings

*Reference document compiled from development and experimental work,
March 2026. Covers the Arborist.jl framework design, bin packing
experiments, and connections to related work. Intended as a
structured memory aid for writing the paper.*

---

## 1. Framework Design and Architecture

### 1.1 The Wallace.jl lesson

Wallace.jl (2014–2015) was the most ambitious Julia evolutionary
computation framework to date, covering Koza GP, strictly-typed GP,
grammar-guided GP, PushGP, Cartesian GP, NSGA-II/III, SPEA2, and
co-evolution. It died at Julia 0.3 for a single architectural reason:
it used Julia's internal reflection APIs to generate new concrete
struct types at runtime, one per problem configuration. When Julia
tightened its compiler internals between 0.3 and 0.4, those hooks
changed or disappeared and the entire framework became non-functional
overnight.

Arborist.jl uses parametric types (`GPProblem{G<:AbstractGenome,
E<:AbstractEvaluator}`) instead of runtime type generation, achieving
the same problem-specific specialization via Julia's normal compile-time
dispatch. The framework has no dependencies on undocumented compiler
internals and cannot be broken by Julia compiler changes.

**Paper claim**: Modern Julia's multiple dispatch achieves what Wallace
needed runtime reflection to accomplish, without any stability risk.

### 1.2 The Problem/Algorithm/Solve pattern

Arborist.jl adopts the Problem/Algorithm/Solve pattern established by
SciML's DifferentialEquations.jl and Optimization.jl ecosystems. The
entire public API fits in three lines:

```julia
problem  = GPProblem(evaluator, ExprGenome; function_set=fset)
algorithm = GeneticProgramming(pop_size=100, generations=200)
result   = solve(problem, algorithm)
```

This is the Julia ecosystem's lingua franca for scientific computing.
Users of DifferentialEquations.jl or Optimization.jl will find the
API immediately familiar.

### 1.3 Dual genome types

The `@eval` approach for ExprGenome and DynamicExpressions.jl for
TreeGenome represent a genuine architectural tradeoff, not a redundancy:

- **TreeGenome**: 8.4x faster evaluation than ExprGenome on the Koza
  symbolic regression suite. Achieves this by eliminating `@eval`
  compilation overhead via vectorized expression tree evaluation over
  data matrices. Appropriate for pure function approximation.

- **ExprGenome**: Supports arbitrary Julia control flow (loops,
  conditionals, mutable state, side effects). Required for programs
  like bin packing heuristics, sorting algorithms, and the Santa Fe
  Ant Trail. Cannot be replaced by TreeGenome for these use cases.

The 4-bit boolean parity benchmark (originally infeasible with
ExprGenome due to ~11 minutes per seed from `@eval` overhead) becomes
tractable with TreeGenome, demonstrating the practical significance
of the distinction.

### 1.4 The simulator pattern for side-effectful programs

ExprGenome's type system assumes pure functions with typed inputs and
outputs. Programs requiring side effects (bin packing, ant trail,
sorting) are handled via the simulator pattern: hide mutable state in
module-level `Ref` objects, expose scalar primitives to the function
set. This was developed for AntGenome and generalized to bin packing
and sorting examples.

The pattern enables Turing-complete program synthesis while keeping
the type-consistency machinery intact. The critical implementation
detail is thread-local simulator state (one instance per thread) to
support parallel evaluation via `Threads.@threads`.

### 1.5 LLM mutation operator architecture

`LLMMutationOperator` implements the FunSearch/AlphaEvolve pattern
(LLM as semantic mutation operator in an evolutionary loop) as a
first-class `AbstractMutationOperator`. Key design decisions:

- Graceful degradation: all LLM failures (API error, timeout, parse
  failure, type-consistency failure) log a `@warn` and fall back to
  a classical operator. The evolutionary loop is robust to 100% LLM
  failure rate.
- Partial recovery in `deserialize`: invalid lines in LLM output are
  skipped rather than rejecting the entire response. Critical for
  real LLM outputs that mix valid code with explanatory text.
- Supports Anthropic, OpenAI-compatible, and local Ollama endpoints
  via endpoint URL detection.
- The ASTSanitizer provides defense-in-depth: evolved programs
  calling filesystem or network primitives are rejected before
  `@eval` execution. Documented as defense-in-depth, not a sandbox.

**Contrast with DTSE**: Both LLMMutationOperator and DTSE's shadow
gradients are examples of "informed search" that replace blind
stochastic operators with operators using richer signal. DTSE routes
gradient information through an ODE solver adjoint; LLMMutationOperator
routes semantic knowledge from pre-training. Both are composable with
classical operators in the same evolutionary loop.

### 1.6 Reproducibility design

All stochastic operations thread an explicit `rng::AbstractRNG`
parameter derived from a user-specified seed. Collection sampling
uses deterministic sorted ordering to counter Julia's non-deterministic
Dict/Set iteration. Temp variable names use `__temp_$i` counters
rather than `gensym()` to avoid session-dependent naming.

Fixed seed + `parallel=false` gives fully reproducible runs.
Fixed seed + `parallel=true` gives reproducible population
initialization but non-deterministic evaluation order due to thread
scheduling.

---

## 2. Online 1D Bin Packing Experiments

### 2.1 Problem setup

Online 1D bin packing: items arrive with sizes drawn from a
distribution, bins have fixed capacity C=1.0, each item must be
placed immediately. Fitness = mean(bins_used / L2_lower_bound)
across N episodes. Lower is better; 1.0 = optimal.

Baselines on uniform item distribution, test set:
- First Fit: 1.0924
- Best Fit: 1.0787
- Gap: 0.0137

The FunSearch paper (Romera-Paredes et al., Nature 2023) used this
problem on OR-Library benchmarks, discovering heuristics that beat
Best Fit by avoiding small gaps rather than always placing in the
tightest bin.

### 2.2 Key results summary

| Configuration | Test fitness | vs Best Fit | Std (5 seeds) |
|---|---|---|---|
| First Fit baseline | 1.0924 | — | — |
| Best Fit baseline | 1.0787 | — | — |
| Best-fit template (no evolution) | 1.0713 | 0.0% | 0.0047 |
| Classical GP, 100 gen | 1.0717 | -0.03% | 0.0112 |
| Behavioral speciation, 100 gen | 1.0689 | +0.23% | 0.0027 |
| Classical + Qwen3-Coder, 100 gen | 1.0700 | +0.13% | 0.0067 |
| Classical GP, 300 gen | 1.0698 | +0.13% | — |
| Classical + Qwen3-Coder, 300 gen | 1.0683 | +0.28% | — |
| Classical GP, 800 gen (time-matched) | 1.0623 | +1.51% | — |
| Best single result (seed 1337) | 1.0527 | +2.41% | — |

### 2.3 What evolved programs discovered

All successful evolved programs independently discovered the
Best Fit skeleton: a bounded loop over open bins, computing
remaining capacity minus item size, tracking the minimum-waste
bin, and placing in that bin after the scan. Structural variation
appears in threshold multipliers and initial values.

Representative evolved Best Fit variant (B1, seed 42):
```julia
__temp_1 = Int32(1)
__temp_5 = 0.53914535f0      # initial best-score threshold
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

LLM-refined variant (A2, seed 42) uses a tighter threshold:
```julia
__temp_5 = Float32(0.281)     # lower initial threshold
...
if __temp_6 < __temp_5 * 0.6157252f0  # shrinks acceptance window
```

The LLM tuned the threshold parameters toward a more selective
strategy that avoids early-bin bias. This parameter tuning within
a structural template is where LLMs excel.

### 2.4 LLM operator findings

**Qwen3-Coder-30B (local Ollama, 3.3B active parameters):**
- Fallback rate: 0.2% (3/1313 calls)
- Mean call latency: 1.21s
- Understood the problem representation and generated valid Julia
  with bin packing primitives on 99.8% of calls

**Sample efficiency finding**: The LLM reaches fitness 1.0699 at
generation 91; classical GP doesn't reach that level until generation
~190. The LLM provides genuine 2x sample efficiency in *generation
count*. However, in *wall time*, classical GP given equal budget
(800 generations, ~4400s) achieves 1.0623, surpassing the 100-gen
LLM result (1.0679) and the 300-gen LLM result (1.0683).

**Conclusion**: LLM operators provide sample efficiency when generation
count is the bottleneck (expensive evaluators, limited time). Given
equal wall-clock budget, classical GP with more generations matches
and surpasses LLM-assisted GP at 4x less wall time. The LLM's value
is clearest in settings where the function set is complex enough
that semantic guidance provides advantages beyond threshold tuning.

### 2.5 Speciation findings

**Genomic speciation (ThresholdSpeciation, symmetric-difference
distance) is harmful for this problem.** With threshold=10, the
population splits into ~100-122 species for 200 individuals because
syntactically trivial differences (one literal changed, one operator
swapped) produce large structural distances. Most species are
singletons or pairs, making fitness sharing irrelevant. The species
assignment algorithm itself enforces diversity, and the overhead
prevents good programs from reproducing freely.

Root cause of the harm: the sharing formula `shared = raw * species_size`
(correct for NEAT-style topology protection) inverts selection
pressure when most species are singletons. A species of 5 at fitness
1.10 has shared fitness 5.5, which loses to a singleton at 2.0.
Weaker formulas (`raw * sqrt(species_size)`, `raw * log2(species_size+1)`)
reduce but do not eliminate the problem.

**Behavioral speciation** (BehavioralSpeciation with Hamming distance
on placement decisions over a fixed probe sequence) correctly collapses
syntactic variants of the same algorithm into one species. Reduces
species count from ~100 to 53-60 with only 3% computational overhead
vs no speciation. Achieves the lowest variance across seeds (std=0.0027
vs 0.0112 for classical GP), trading peak performance for reliability.

**Key insight**: Speciation strategy is problem-class dependent.
For structural innovation problems (NEAT, topology search),
`ThresholdSpeciation` with `:linear` sharing is correct — small
novel species need protection. For behavioral optimization problems
where convergence to a good basin is desirable, behavioral speciation
with `:sqrt` or `:log2` sharing is more appropriate. Arborist.jl
exposes the sharing formula as a parameter to support both use cases.

### 2.6 Seeding findings

The best-fit template (the clean Best Fit skeleton without evolution)
achieves mean test fitness 1.0713 across 5 seeds with std=0.0047.
This is nearly identical to 100-gen classical GP (mean 1.0717, std=0.0112).

**Interpretation**: The seeding strategy matters as much as the
evolutionary algorithm for this problem. Population initialization
with semantically meaningful templates (not just random programs)
provides a strong starting point that 100 generations of classical
GP barely improves upon. The real improvement comes from extended
runs (300-800 generations) that have enough time to refine beyond
the template.

**Practical recommendation**: For expensive real-world problems,
invest time in designing good seed templates rather than running
longer evolutionary searches with random initialization.

### 2.7 Bimodal distribution findings

The bimodal distribution (50% small items 0.1-0.3, 50% large items
0.6-0.9) has a much smaller First Fit / Best Fit gap (0.0032) than
uniform (0.0137). This counterintuitive result arises because the
bimodal distribution makes First Fit nearly optimal: large items
leave 0.1-0.4 capacity, and small items fill that gap efficiently
by construction. The pairing happens automatically without any
sophisticated heuristic.

Evolved bimodal programs converged to first-fit scanners rather
than discovering item-size-conditional strategies. The hypothesis
that bimodal supports richer evolved strategies was not confirmed
for this specific parameterization.

**Implication**: Richer fitness landscapes (larger FF-BF gaps) support
more interesting evolved strategies. The uniform distribution is a
better benchmark for demonstrating GP's ability to discover non-trivial
heuristics.

### 2.8 Comparison to FunSearch

FunSearch (Romera-Paredes et al., 2023) used Gemini 1.5 as the
LLM operator on OR-Library benchmarks, discovering heuristics that
beat Best Fit by avoiding bins where gaps are unlikely to be filled.
The key insight: only place in the tightest bin if the fit is very
tight; otherwise use a bin with more remaining space.

Arborist.jl's evolved programs discovered Best Fit variants
(tighter threshold parameters) rather than FunSearch's insight
about avoiding small gaps. This likely reflects the limited
function set (no operation to measure how "useful" a gap is) and
the modest generation budgets in these experiments. The FunSearch
strategy is expressible in ExprGenome's primitive set and may
emerge with longer runs.

The critical architectural difference: FunSearch treats the
evolutionary loop as bespoke infrastructure around the LLM.
Arborist.jl treats the LLM as one operator among many in a
composable framework. The evolved programs are identical in
representation; the infrastructure is reusable.

---

## 3. Speciation Design Observations

### 3.1 The sharing formula tradeoff

Three sharing formulas are now implemented in Arborist.jl:
- `:linear` — `raw * species_size`: strong penalty, appropriate for
  NEAT-style topology protection of small novel species
- `:sqrt` — `raw * sqrt(species_size)`: moderate penalty, good for
  behavioral optimization
- `:log2` — `raw * log2(species_size+1)`: gentle penalty, good when
  species sizes are expected to be large

The bin packing experiments demonstrate empirically that `:linear`
is harmful for behavioral optimization (created ~100 species, hurt
convergence) while `:sqrt` provides diversity pressure without
convergence harm.

### 3.2 Behavioral vs genomic distance

Genomic distance (symmetric-difference of AST node sets after
type-normalization) measures syntactic novelty. For behavioral
optimization, this is the wrong thing to measure: two programs
implementing First Fit with different AST structure are
genomically distant but behaviorally identical.

Behavioral distance (Hamming distance on placement decisions over
a fixed probe sequence) correctly identifies this equivalence.
The threshold of 0.15 (programs that agree on ≥85% of placement
decisions are same-species) successfully collapses syntactic variants
while distinguishing genuinely different strategies.

The probe sequence design matters: 50 items from a fixed seed,
with 20 pre-opened bins, provides enough resolution to distinguish
First Fit from Best Fit variants while being cheap to compute
(3% overhead vs no speciation).

### 3.3 XOR NEAT benchmark

The fitness sharing sign error (division instead of multiplication)
in the initial GraphGenome implementation produced networks that
learned nothing — shared fitness 0.22 for a 5-member species vs
raw 2.0 for a singleton, inverting selection. After fixing to
multiplication, XOR converged 4/5 seeds to fitness < 0.01 within
150 generations, matching the original NEAT paper benchmark.
Networks grew organically from 4 to 7-11 nodes via structural mutation.

---

## 4. Related Work Connections

### 4.1 neat-python

Arborist.jl's ExprGenome is the program synthesis analog of neat-python's
neural network genome. Both use typed representations, both expose
a simple user-facing API (implement your fitness function, run),
and both were designed with zero mandatory dependencies for maximum
accessibility. The citation count correlation between neat-python's
accessibility and its adoption motivates Arborist.jl's same design
philosophy.

### 4.2 FunSearch and AlphaEvolve

FunSearch (Romera-Paredes et al., Nature 2023): LLM as semantic
mutation operator in an evolutionary loop, discovered best-known
results for cap set problem and beat Best Fit on bin packing.
Uses Gemini 1.5, island-based population database, bespoke
infrastructure per problem.

AlphaEvolve (DeepMind, 2025): Generalizes FunSearch from evolving
single functions to entire codebases. Uses ensemble of Gemini
Flash + Pro, MAP-Elites-inspired program database.

Arborist.jl's position: the LLM operator is one pluggable mutation
operator in a framework that also supports classical GP, NEAT-style
neural evolution, and side-effectful program synthesis. The
framework is reusable across problem domains without bespoke
infrastructure.

### 4.3 DTSE (shadow gradients)

DTSE (Differentiable Topology Search via Shadow Gradients) uses
adjoint-based gradient information routed through ODE solvers to
inform neural network topology search — the same "informed search"
principle as the LLM operator, applied to continuous optimization
rather than discrete program synthesis. Both are composable with
classical operators. The DTSE connection motivates a unified view
of "informed search operators" as a design pattern.

### 4.4 DynamicExpressions.jl

TreeGenome wraps DynamicExpressions.jl's `Node{T}` type, which
is the backbone of SymbolicRegression.jl and PySR. The 8.4x
evaluation speedup over ExprGenome on the Koza suite comes from
vectorized SIMD evaluation over data matrices without `@eval`
compilation overhead. This makes TreeGenome appropriate for the
symbolic regression use case where DynamicExpressions.jl was
designed for, while ExprGenome handles the program synthesis
use cases that DynamicExpressions.jl cannot represent.

---

## 5. Open Questions for Future Work

### 5.1 The gap FunSearch closes

FunSearch discovered a qualitatively different bin packing strategy
(avoid small gaps) rather than just tuning Best Fit parameters.
Arborist.jl's classical GP found threshold-tuned Best Fit variants.
The questions:
1. Does a longer Arborist.jl run (2000+ generations) discover the
   gap-avoidance strategy independently?
2. Does a smarter system prompt that explicitly describes the
   gap-avoidance concept help the LLM operator guide the search there?
3. Is the function set rich enough? Adding a `bp_gap_score(i)` primitive
   that measures how useful a bin's remaining space is might accelerate
   discovery.

### 5.2 Bimodal with harder parameterization

The 50/50 bimodal with small=0.1-0.3 and large=0.6-0.9 has a small
FF-BF gap because First Fit naturally pairs items well. A harder
bimodal parameterization (small=0.2-0.4, large=0.5-0.8, with
significant overlap) would create a larger gap and richer fitness
landscape where item-size-conditional strategies have a real advantage.
Testing whether evolved programs discover `bp_item_size() > threshold`
conditionals on this harder distribution is a natural follow-on.

### 5.3 OR-Library benchmark comparison

FunSearch used OR1-OR4 benchmark instances from the OR-Library
(Falkenauer, 1994). Running Arborist.jl on the same instances
would allow direct quantitative comparison with Table 1 of the
FunSearch paper, which is the clearest way to position the work
relative to the state of the art.

### 5.4 The LLM operator with a better prompt

The current system prompt explains the primitive API but does not
provide a "skeleton" in the FunSearch sense — a pre-defined program
structure with only the scoring function to evolve. FunSearch's
key insight was that fixing the skeleton and evolving only the
priority function significantly improved results. An Arborist.jl
system prompt that includes the Best Fit skeleton and asks the LLM
to improve only the scoring condition would be a direct FunSearch
analog.

### 5.5 3D bin packing via ContainerLoading.jl

ContainerLoading.jl provides a working 3D packing engine with MCTS,
the maximal empty space decomposition, and realistic physics
constraints (drop-from-above, corner support). Arborist.jl can
evolve the spatial strategy (the function that scores candidate
placement spaces) using the simulator pattern. This extends FunSearch's
1D result to 3D, which is directly relevant to the Liberty Robotics
box packing use case.

### 5.6 Sorting algorithm synthesis

Implemented and tested. Uses curriculum learning (length 3 → 20),
normalized inversion count fitness, and the swap-only primitive set
(no `set!`, only `swap!`) to ensure multiset preservation.

**Baseline result**: With seeded initial population (4 template types
including selection sort and bubble sort skeletons), evolution preserves
and slightly simplifies the selection sort template. Achieves 100%
correct sorting on all lengths 3–24 within 5 generations at pop=20.
This is Tier 2 (target) with full generalization — a clean result
demonstrating that the framework handles non-trivial control flow
(nested loops, conditional swaps, index arithmetic).

**Comparison count penalty experiments**: Three attempts to push
evolution beyond O(n²) sorting via an additive comparison penalty
`α × comparisons / (n · log₂ n)` with adaptive α = base × max(0, length − 6).
All three failed, revealing a fundamental fitness design finding:

1. **Ungated additive penalty** (α applied to all episodes): Evolution
   immediately broke correctness by skipping array elements (inner loop
   step of 5 instead of 1), because saving comparisons was worth more
   fitness than the inversion cost of missing pairs. At α=0.3, saving
   51 comparisons (worth 0.36 fitness) exceeded the ~0.05 inversion
   penalty from skipping elements.

2. **Gated penalty** (α applied only to correctly-sorted episodes):
   Created a subtler perverse incentive. Programs that sort almost
   nothing pay almost no comparison cost, while correct sorters pay
   the full penalty. An incorrect program with 5% inversions (fitness
   ~0.05) outranked a correct selection sort (fitness 0 + 0.12
   comparison cost) by total fitness. Two bugs surfaced:
   - Curriculum advancement checked only genomes[1] (best by total
     fitness), which wasn't the correctness-best individual
   - Even after checking top-20 individuals, the fitness landscape
     still had a lower-fitness basin for nearly-correct programs than
     for fully-correct ones at α=0.7

**Key finding**: Additive multi-objective fitness creates perverse
selection pressure whenever the secondary objective (comparison count)
has a steeper gradient than the primary objective (correctness). This
is a general GP fitness design principle: secondary objectives that
can be cheaply "gamed" by degrading the primary objective must use
lexicographic prioritization (sort by correctness first, comparison
count second), not additive weighting. The additive approach fails
regardless of gating strategy because the fitness landscape always
has a basin where trading small amounts of correctness for large
comparison savings is locally optimal.

**Untried alternatives**: Lexicographic fitness (any correct program
beats any incorrect one), threshold gating (comparison penalty only
when ALL episodes are correct), or very small α (~0.01) as a strict
tiebreaker. The lexicographic approach is the cleanest but requires
modifying the fitness comparison throughout the solve loop.

### 5.7 Adversarial LLM strategy (dual-LLM GP)

Current LLM operator uses a single model as proposer. A proposer-judge
architecture (one LLM proposes mutations, a second lightweight LLM
screens for semantic plausibility before fitness evaluation) would
reduce the effective fallback rate by filtering obviously degenerate
proposals before the expensive `@eval` step. This is the "constitutional
mutation" pattern — not yet implemented.

### 5.8 Parallelism completions

Phase 5 implemented `Threads.@threads` for population evaluation
but deferred island model parallelism and concurrent LLM calls.
Island parallelism maps naturally to `Threads.@spawn` and would
make the LLM operator economical at larger scales (LLM mutations
run concurrently with classical evaluation on other islands,
amortizing the latency).

---

## 6. Empirical Numbers for Paper Tables

### Koza symbolic regression suite (TreeGenome)
- Koza-1 (x⁴+x³+x²+x): 5/5 seeds converge, ~1.5s per benchmark
- Koza-2 (x⁵-2x³+x): 5/5 seeds converge, ~1.5s
- Koza-3 (x⁶-2x⁴+x²): 5/5 seeds converge, ~1.5s

### Boolean 4-bit parity (TreeGenome)
- 1/5 seeds converge to fitness ≤ 0.1 within 500 generations
- Demonstrates that 4-bit parity is genuinely hard for tree-based GP
- Previously infeasible with ExprGenome (~11 minutes per seed)

### XOR NEAT (GraphGenome)
- 4/5 seeds converge to fitness < 0.01 within 150 generations
- Matches original NEAT paper benchmark
- Networks grow from 4 to 7-11 nodes via structural mutation

### TreeGenome vs ExprGenome speedup
- 8.4x faster evaluation on 1000-point Koza-1
- Expected to be proportionally larger on larger datasets

### Bin packing (online 1D, uniform distribution)
- First Fit: 1.0924 (test)
- Best Fit: 1.0787 (test)
- Best single evolved result: 1.0527 (seed 1337, classical GP,
  3.41% better than Best Fit)
- Best mean across 5 seeds: 1.0689 (behavioral speciation)
- LLM fallback rate (Qwen3-Coder-30B local): 0.2%
- LLM mean call latency: 1.21s
- Sample efficiency: LLM reaches target fitness ~2x faster in
  generation count; classical GP matches in wall time given 4x
  more generations

### Sorting algorithm synthesis (ExprGenome, curriculum learning)
- Seeded selection sort: 100% correct on lengths 3–24, achieved in
  5 generations at pop=20 (selection sort template preserved intact)
- Comparison count at length 12: 66 (= n(n-1)/2, pure selection sort)
- Comparison count at length 24: 276 (= n(n-1)/2, pure selection sort)
- Comps/(n·lg n) ratio: ~1.07 at length 12, ~2.01 at length 24
  (confirming quadratic scaling vs the O(n log n) ideal)
- Comparison penalty experiments: 3 runs, 0 produced correct sub-quadratic
  sorting — additive penalty creates perverse incentive regardless of
  gating strategy

### Test suite
- 1172 tests total (Phases 1-6 + behavioral diversity)
- Fast tier (unit + integration): ~17 seconds
- Full benchmark tier: ~80 seconds (GENPROG_RUN_BENCHMARKS=true)
- Package precompile time: ~364ms (GenProg core)

---

## 7. Naming and Positioning Notes

**Name**: Arborist.jl (formerly GenProg.jl, renamed to avoid collision
with Weimer et al. 2009 automated software repair framework which is
well-known in software engineering research).

**The Wallace.jl revival framing**: Wallace.jl was the right idea,
killed by the wrong mechanism, at the wrong time. Arborist.jl is
what Wallace.jl looks like built correctly in modern Julia.

**Positioning relative to existing Julia EC packages**:
- Evolutionary.jl: last release Jan 2022, GP pipeline broken in
  current version, no active maintenance
- Metaheuristics.jl: numerical optimization only, vector-based
  search spaces, not applicable to program synthesis
- EvoLP.jl: academic playground, 27 stars, not production-ready
- Cambrian.jl: abstract EC framework used in academic EC courses,
  requires `genes::<:AbstractArray` which limits non-vector genomes

None of the existing packages support non-vector genome types with
an extensible interface. Arborist.jl fills this gap.

**Relationship to neat-python**: Arborist.jl is by the neat-python
maintainer. neat-python covers NEAT neural topology evolution in
Python. Arborist.jl covers generic GP in Julia with NEAT as one
of several supported genome types. The packages are complementary.
