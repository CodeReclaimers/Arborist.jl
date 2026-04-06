# Wallace.jl Feature Gap Analysis & Recommendations for Arborist.jl

## Background

Wallace.jl (Timperley & Stepney, ECAL 2015) was a Julia 0.3-era evolutionary computation
framework. Archived since 2017, 21 GitHub stars. Many features claimed in the README were
aspirational -- source code shows partial implementations for several advertised capabilities
(Strongly Typed GP, PushGP, Cartesian GP, NSGA-II, coevolution all claimed but not found in source).

This document compares Wallace's feature set with Arborist.jl's current implementation and
the broader Julia ecosystem, then recommends which missing features are worth adding before
package registration.

## Features Arborist already covers (equal or better than Wallace)

| Feature | Wallace | Arborist |
|---------|---------|----------|
| Koza-style tree GP | Yes | ExprGenome, TreeGenome |
| Graph/neuroevolution | Basic | Full NEAT (GraphGenome) |
| Tournament selection | Yes (size 7) | Yes (configurable) |
| Subtree crossover/mutation | Yes | Yes + point, hoist, expansion |
| Fitness sharing | Partial stub | Full (4 formulas: none/linear/sqrt/log2) |
| Island model | Claimed, partial | Full (sequential, sync distributed, async distributed) |
| Ant trail evaluator | Yes | Yes (AntGenome + AntEvaluator) |
| Multi-threading | Basic | Threaded eval + Distributed.jl workers |
| Imperative programs | No | AntGenome (side-effectful) |
| LLM-guided mutation | N/A (predates LLMs) | Full (FunSearch/AlphaEvolve pattern) |
| Speciation | No | Threshold + Behavioral |
| AST security | No | Whitelist-based sanitizer |

Arborist is substantially more complete than Wallace in every shared area.

## Wallace features Arborist lacks -- ranked by value

### 1. Multi-Objective GP (NSGA-II) -- HIGH priority, pre-registration

**What it is.** Optimizing multiple objectives simultaneously (e.g., prediction accuracy vs.
program complexity). NSGA-II (non-dominated sorting + crowding distance) is the standard
algorithm.

**Wallace status.** Had Pareto fitness infrastructure (Goldberg ranking, MOGA ranking,
lexicographic, weighted sum scalarization). Full NSGA-II was not completed.

**Julia ecosystem gap.** Evolutionary.jl has NSGA-II but is dormant (no commits since 2023-05,
31 open issues). Its TreeGP + NSGA-II integration is minimal. Metaheuristics.jl has NSGA-II/III,
SPEA2, and SMS-EMOA but has no GP representations at all.

**Why it matters.** "Accuracy vs. complexity" is the central tension in GP. Parsimony pressure
via bloat penalty is a crude proxy for what Pareto optimization handles properly. Multi-objective
GP is well-established in the literature and expected by GP researchers. The dormancy of
Evolutionary.jl creates a clear opening.

**Implementation scope.** Moderate. Requires vector-valued fitness, non-dominated sorting,
crowding distance, modified selection. The solve loop needs `Vector{Float64}` fitness instead
of `Float64`. Could be a new algorithm type or an extension of `GeneticProgramming`.

**Reception likelihood.** High. Reviewers familiar with GP will notice its absence. The
combination of multi-objective + island model + LLM mutation would be unique across all
languages, not just Julia.

### 2. Strongly Typed GP -- HIGH priority, pre-registration

**What it is.** Type annotations on function and terminal sets; tree construction enforces
argument/return type constraints at every node. Enables mixing booleans, integers, floats,
and domain-specific types in a single evolved program.

**Wallace status.** Claimed in README, not found in source code.

**Julia ecosystem gap.** No Julia implementation anywhere. DEAP (Python) has
`PrimitiveSetTyped`, which is the reference implementation most researchers use.

**Why it matters.** Required for program synthesis and any domain mixing numeric + boolean +
categorical operations. Without it, researchers doing typed GP in Julia must write from scratch
or fall back to Python/DEAP. This is a "table stakes" feature -- its absence is conspicuous
to anyone with GP experience.

**Implementation scope.** Moderate. Extends ExprGenome/TreeGenome function sets with type
signatures; modifies tree builders (grow, full, half-and-half) and crossover to respect type
constraints at each node. The type system can be Julia's own type hierarchy.

**Reception likelihood.** High. This is a "we've come to expect it" feature. Every serious
GP framework (ECJ, DEAP, HeuristicLab) supports typed GP.

### 3. Grammatical Evolution -- HIGH priority, pre-registration

**What it is.** An integer-vector genome mapped through a BNF grammar to produce programs in
an arbitrary target language. The grammar defines the search space; evolution operates on the
integer vector via standard GA operators.

**Wallace status.** Actually implemented -- grammar parser, codon wrapping, derivation to
target language.

**Julia ecosystem gap.** Complete gap. No Julia package offers grammatical evolution.
Researchers doing GE in Julia must write from scratch. The canonical Python implementation
is PonyGE2.

**Why it matters.** GE is actively researched (O'Neill's group continues publishing). It is
the standard approach when evolving programs in an arbitrary target language or when complex
syntactic constraints must be enforced. A clean Julia GE implementation would attract the
PonyGE2 user community.

**Implementation scope.** Self-contained. New genome type (`GrammarGenome`), BNF parser,
codon-to-derivation mapping. Does not require changes to existing genome types or the solve
loop. Standard integer-vector crossover and mutation apply to the codon representation.

**Reception likelihood.** High. Fills a complete ecosystem gap. Previous GE research in Julia
was blocked by the lack of a reference implementation.

### 4. Cooperative/Competitive Coevolution -- MODERATE priority, post-registration

**What it is.** Multiple populations evolving interdependently. Cooperative: subpopulations
contribute components of a composite solution. Competitive: populations provide test cases
for each other (predator-prey, adversarial testing).

**Wallace status.** Architectural support via multi-species populations, but no coevolution
algorithm was implemented.

**Julia ecosystem gap.** Complete gap. No Julia package implements general coevolution.
Metaheuristics.jl has CCMO (constrained multi-objective coevolution) but that is a specific
algorithm, not a general framework.

**Why it matters.** Used for adversarial testing, game-playing AI, modular problem
decomposition. Active in reinforcement learning and game AI research. Arborist's island model
provides ~80% of the needed infrastructure -- the missing piece is cross-population fitness
coupling (an individual's fitness depends on opponents/collaborators from other populations).

**Implementation scope.** Moderate. Requires a fitness evaluation protocol where individuals
from different populations interact, plus a coordination mechanism for population updates.

**Reception likelihood.** Moderate. Smaller audience than multi-objective or typed GP, but
the complete ecosystem gap means any implementation would be the only option in Julia.

### 5. Cartesian GP -- LOW-MODERATE priority, post-registration

**What it is.** Fixed-length integer vector encoding a directed acyclic graph. Each gene
specifies a node's function and input connections. Popular for digital circuit design, image
processing, and neural architecture search.

**Wallace status.** Claimed in README, not implemented in source.

**Julia ecosystem gap.** CartesianGeneticProgramming.jl exists but is built on dormant
Cambrian.jl (last updated 2021-12). Fragile foundation.

**Why it matters.** Active research area, particularly in hardware design and image processing.
However, it overlaps significantly with GraphGenome's niche (both encode computation graphs).

**Implementation scope.** New genome type with fixed-length integer vector, node-level
mutation (no crossover in standard CGP). Moderate effort.

**Reception likelihood.** Moderate. The existing (fragile) Julia implementation reduces
urgency. Would be more compelling if Arborist's version integrated with the rest of the
framework (island model, LLM mutation, speciation).

### 6. PushGP -- LOW priority, post-registration

**What it is.** Stack-based GP with typed stacks (integer, float, boolean, exec, code).
Very expressive for general program synthesis -- avoids the closure problem of tree-based GP.

**Wallace status.** Claimed in README, not implemented.

**Julia ecosystem gap.** No Julia implementation. Canonical implementation is Clojush (Clojure).

**Why it matters.** Active research niche (Spector, Helmuth, et al.). Used extensively in
program synthesis benchmarks. A Julia PushGP would be notable but the user base is small
and specialized.

**Implementation scope.** Large. Entirely new execution model (stack machine), instruction
set design, type system. Self-contained but significant effort.

**Reception likelihood.** Moderate within the program synthesis community, low in the broader
GP community. Better as a post-1.0 feature or contributed module.

### 7. Additional Selection Operators -- LOW priority

Roulette wheel, rank-based, stochastic universal sampling, truncation selection.

Wallace had only tournament + random. Evolutionary.jl has the full set. For GP specifically,
tournament selection dominates the literature -- other operators are rarely used in GP
contexts. Easy to add but low impact on adoption.

## Features to skip

| Feature | Reason |
|---------|--------|
| Bit/float/int vector GA | Evolutionary.jl covers this well. Would dilute GP focus. |
| Estimation of Distribution Algorithms | Different paradigm from GP. Poor architectural fit. Better as separate package. |
| CMA-ES / Differential Evolution | Numerical optimization, not GP. Metaheuristics.jl and Evolutionary.jl cover these. |
| Particle Swarm Optimization | Same as above. |

## Pre-registration recommendation

Add these three before v0.1.0:

1. **Multi-Objective GP (NSGA-II)** -- highest immediate impact, expected by GP researchers
2. **Strongly Typed GP** -- table stakes for program synthesis, conspicuous absence
3. **Grammatical Evolution** -- complete Julia ecosystem gap, self-contained implementation

These three features, combined with the existing LLM mutation, island model, and speciation
support, would make Arborist the most feature-complete GP framework in any language's
ecosystem -- not just Julia's.

## Post-registration roadmap (v0.2.0+)

4. Coevolution (builds naturally on island model)
5. Cartesian GP (new genome type)
6. Additional selection operators (easy but low priority)
7. PushGP (significant scope, niche audience)
