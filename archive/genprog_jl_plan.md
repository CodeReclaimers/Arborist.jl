# GenProg.jl — Design and Development Plan

*A generic, extensible genetic programming framework for Julia,
built on the Problem/Algorithm/Solve pattern.*

---

## 1. Motivation and Positioning

The Julia ecosystem has no generic evolutionary computation framework
that survives past major language versions, supports non-vector genome
types, or exposes the kind of composable Problem/Algorithm/Solve
interface that the SciML ecosystem normalized. Wallace.jl was the most
ambitious attempt (2014–2015) and died at Julia 0.3 because it used
runtime type generation via computational reflection — hooks that
changed when Julia's compiler matured. Every subsequent package either
narrowly targets numerical optimization (Metaheuristics.jl,
Evolutionary.jl) or is unmaintained.

This framework targets a different position: **genetic programming
first, generic evolutionary computation second**, with explicit
first-class support for LLM-as-operator, and an architecture that
cannot be broken by Julia compiler changes because it never touches
compiler internals.

### Relationship to existing code

`codegen.jl`, `evolution.jl`, and `codegen_interfaces.jl` become the
`ExprGenome` implementation inside this framework. They are not
refactored away — they are wrapped. The existing test suite and the
x² convergence benchmark become the first entry in the benchmark suite.

---

## 2. The Wallace.jl Lesson: What Not to Do

Wallace died for a single architectural reason: it used Julia's
internal reflection APIs to **generate new concrete struct types at
runtime**, one per problem configuration. When Julia tightened its
compiler internals between 0.3 and 0.4, those hooks changed or
disappeared, and the entire framework became non-functional overnight.

### Rules derived from this failure

**Rule 1: Never generate struct types at runtime.**  
Use parametric types instead. `GPProblem{G<:AbstractGenome,
E<:AbstractEvaluator}` specializes at compile time via Julia's normal
dispatch. Zero reflection required.

**Rule 2: Never depend on `Base` or `Core` internals.**  
Use only the public Julia API. If a function is not in the official
docs, don't use it. This means no `Base.Compiler.*`, no
`Core.Compiler.*`, no undocumented reflection.

**Rule 3: `@eval` for user code is fine; `@eval` for framework
structure is not.**  
The existing `construct_and_define_function` uses `@eval` to compile
evolved programs — that is legitimate and stable (it is what Distributed.jl,
Turing.jl, and many other packages do). Using `@eval` to generate the
framework's own types or methods is what killed Wallace.

**Rule 4: Target a Long-Term Support Julia release as minimum.**  
Minimum version: Julia 1.10 (LTS as of 2024). This gives access to
package extensions (1.9+), stable precompilation (1.9+), and a
maintained release that will receive security patches for years.

**Rule 5: Use package extensions for optional integrations.**  
Julia 1.9 introduced first-class package extensions (`ext/` directory,
`[weakdeps]` in `Project.toml`). Optional integrations — LLM
operators, DynamicExpressions.jl node types, visualization — live in
extensions and are never loaded unless the user explicitly depends on
both packages. This is the modern replacement for Requires.jl and is
part of the stable public API.

---

## 3. Naming

**Proposed name: `GenProg.jl`**

Rationale: maximum discoverability for the target audience (people
searching for genetic programming in Julia), unambiguous about what it
does, not yet taken in the registry. Alternatives if taken:
`EvoProgram.jl`, `Arborist.jl` (emphasizes tree/AST focus).

---

## 4. Architecture: Problem / Algorithm / Solve

Modeled on the SciML/DiffEq pattern. The user constructs a problem
struct, picks an algorithm, calls `solve`. The framework dispatches
on genome type and algorithm type to select the right methods.

```julia
# User-facing API — the entire public surface in three lines:
problem  = GPProblem(evaluator, ExprGenome; function_set=fset, num_temps=4)
algorithm = GeneticProgramming(pop_size=100, generations=200, mutation_rate=0.3)
result   = solve(problem, algorithm; verbose=true)
```

### 4.1 Abstract type hierarchy

```julia
# genome_types.jl
abstract type AbstractGenome end

# Required interface for any AbstractGenome subtype:
#   initialize(::Type{G}, problem::GPProblem) -> G
#   mutate(g::G, rng::AbstractRNG) -> G
#   crossover(g1::G, g2::G, rng::AbstractRNG) -> Tuple{G,G}
#   distance(g1::G, g2::G) -> Float64
#   complexity(g::G) -> Float64
#   serialize(g::G) -> String          # for LLM operator
#   deserialize(::Type{G}, s::String) -> Union{G, Nothing}  # for LLM operator

# evaluator_types.jl
abstract type AbstractEvaluator end

# Required interface:
#   evaluate(e::AbstractEvaluator, f::Function) -> Float64
#   input_signature(e::AbstractEvaluator) -> Dict{Symbol,DataType}
#   output_signature(e::AbstractEvaluator) -> Dict{Symbol,DataType}

# operator_types.jl
abstract type AbstractMutationOperator end
abstract type AbstractCrossoverOperator end
abstract type AbstractSelectionStrategy end
abstract type AbstractSpeciation end

# algorithm_types.jl
abstract type AbstractEvolutionaryAlgorithm end

# result_types.jl
abstract type AbstractEvolutionResult end
```

### 4.2 Concrete problem struct

```julia
struct GPProblem{G<:AbstractGenome, E<:AbstractEvaluator}
    evaluator::E
    genome_type::Type{G}
    function_set::FunctionSet       # from codegen.jl
    num_temps::Int
    seed::Union{Int, Nothing}
end

function GPProblem(evaluator::E, ::Type{G};
                   function_set=default_function_set(),
                   num_temps::Int=4,
                   seed=nothing) where {G<:AbstractGenome, E<:AbstractEvaluator}
    GPProblem{G,E}(evaluator, G, function_set, num_temps, seed)
end
```

### 4.3 Algorithm structs (all immutable, all keyword-constructed)

```julia
struct GeneticProgramming <: AbstractEvolutionaryAlgorithm
    pop_size::Int
    generations::Int
    mutation_rate::Float64
    crossover_rate::Float64
    elitism::Int
    tournament_size::Int
    max_depth::Int
    bloat_penalty::Float64          # coefficient on complexity(g)
    speciation::AbstractSpeciation
    mutation_ops::Vector{AbstractMutationOperator}
    crossover_ops::Vector{AbstractCrossoverOperator}
    selection::AbstractSelectionStrategy
end

# Sensible defaults:
function GeneticProgramming(;
    pop_size::Int = 100,
    generations::Int = 200,
    mutation_rate::Float64 = 0.3,
    crossover_rate::Float64 = 0.3,
    elitism::Int = 2,
    tournament_size::Int = 3,
    max_depth::Int = 8,
    bloat_penalty::Float64 = 0.0,
    speciation::AbstractSpeciation = NoSpeciation(),
    mutation_ops = [SubtreeMutation(), PointMutation()],
    crossover_ops = [SubtreeCrossover()],
    selection::AbstractSelectionStrategy = TournamentSelection(3)
)
```

### 4.4 Result struct

```julia
mutable struct GPResult{G<:AbstractGenome} <: AbstractEvolutionResult
    best_genome::G
    best_fitness::Float64
    population::Vector{G}
    fitness_history::Vector{Float64}   # best fitness per generation
    mean_history::Vector{Float64}
    generations_run::Int
    wall_time::Float64
    converged::Bool
end
```

### 4.5 The solve entry point

```julia
function solve(problem::GPProblem{G,E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing) where {G,E}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)
    pop = initialize_population(problem, algorithm, rng)
    run_evolution!(pop, problem, algorithm, rng; verbose, callback)
end
```

`run_evolution!` is the internal loop — not part of the public API.
Users who need deep control subtype `AbstractEvolutionaryAlgorithm`
and define their own `solve` method.

---

## 5. Genome Implementations

### 5.1 ExprGenome (wraps existing code)

```julia
struct ExprGenome <: AbstractGenome
    body::Vector{Expr}       # body statements, not yet wrapped
    state::GenState          # type context from codegen.jl
end
```

All existing `mutate!`, `crossover`, `create_random_assignment`, etc.
from `codegen.jl` and `evolution.jl` become the implementation of the
`AbstractGenome` interface for `ExprGenome`. The wrapping is thin —
no rewrite needed.

`complexity(g::ExprGenome)` = total node count across `unravel.(g.body)`.  
`serialize(g::ExprGenome)` = `repr.(g.body) |> join("\n")`.  
`deserialize(::Type{ExprGenome}, s::String)` = `Meta.parse` each line,
return `nothing` on any parse failure.

### 5.2 Future genome types (not in v0.1, but planned slots)

- `LinearGenome{T}` — fixed-length vector, covers classical GA territory
- `TreeGenome` — backed by DynamicExpressions.jl `Node{T}` for fast
  evaluation; lives in a package extension
- `GraphGenome` — for NEAT-style neural topology evolution; would make
  NeatEvolution.jl a dependent rather than a peer

---

## 6. Operator System

### 6.1 Mutation operators

```julia
struct SubtreeMutation <: AbstractMutationOperator end
struct PointMutation <: AbstractMutationOperator end
struct HoistMutation <: AbstractMutationOperator end   # replaces subtree with one of its children
struct ExpansionMutation <: AbstractMutationOperator end  # wraps leaf in random function call

# Dispatch:
function mutate(op::SubtreeMutation, g::ExprGenome, rng::AbstractRNG) -> ExprGenome
function mutate(op::PointMutation, g::ExprGenome, rng::AbstractRNG) -> ExprGenome
```

Multiple mutation operators in `algorithm.mutation_ops` are sampled
proportionally (uniform by default; weights can be specified).

### 6.2 LLM mutation operator (built into core, not an extension)

The LLM operator is a first-class mutation operator, not an
afterthought. It requires an HTTP call, so it does live in a package
extension (`ext/LLMOperatorExt.jl`), but the abstract type and
interface are defined in core so any genome type can implement
`serialize`/`deserialize` to support it.

```julia
# In ext/LLMOperatorExt.jl — loaded when HTTP.jl is a dependency:
struct LLMMutationOperator <: AbstractMutationOperator
    endpoint::String          # e.g. "https://api.anthropic.com/v1/messages"
    model::String
    system_prompt::String
    temperature::Float64
    max_tokens::Int
    timeout_seconds::Float64
end
```

The operator:
1. Calls `serialize(g)` on the genome.
2. Sends it to the LLM with a task-specific system prompt.
3. Calls `deserialize(typeof(g), response)`.
4. Returns the original genome unchanged if deserialization fails
   (graceful degradation — LLM failures never crash the evolutionary
   loop).

This is exactly the FunSearch/AlphaEvolve mutation pattern, but the
evaluator and evolutionary loop are the framework's rather than
bespoke infrastructure.

### 6.3 Island model (for LLM operator scaling)

```julia
struct IslandModel <: AbstractEvolutionaryAlgorithm
    islands::Int
    migration_interval::Int
    migration_size::Int
    island_algorithm::AbstractEvolutionaryAlgorithm
end
```

Island model is the natural home for LLM operators: LLM mutations are
expensive, so they run on a small subset of islands while classical
operators run on the rest. Migration between islands spreads LLM
discoveries into the classical population. This is precisely the
FunSearch architecture, but composable.

---

## 7. Speciation

```julia
struct NoSpeciation <: AbstractSpeciation end

struct ThresholdSpeciation <: AbstractSpeciation
    threshold::Float64        # compatibility distance cutoff
    min_species_size::Int
    stagnation_limit::Int    # generations before a species is culled
end
```

`speciate!(pop, spec::ThresholdSpeciation)` groups individuals by
pairwise `distance(g1, g2)`, using the same threshold-based algorithm
as NEAT but dispatching on genome type for the distance function.
This makes speciation reusable across `ExprGenome`, `GraphGenome`,
and any future genome type without modification.

---

## 8. File Layout

```
GenProg.jl/
├── Project.toml
├── README.md
├── LICENSE
├── src/
│   ├── GenProg.jl              # module, exports, includes
│   ├── abstractions.jl         # all abstract types + interface docstrings
│   ├── genome/
│   │   ├── expr_genome.jl      # ExprGenome wrapping codegen.jl
│   │   ├── codegen.jl          # verbatim from existing project
│   │   ├── evolution.jl        # verbatim from existing project
│   │   └── linear_genome.jl    # placeholder for v0.2
│   ├── operators/
│   │   ├── mutation.jl         # SubtreeMutation, PointMutation, Hoist, Expansion
│   │   ├── crossover.jl        # SubtreeCrossover
│   │   └── selection.jl        # TournamentSelection, FitnessProportional
│   ├── speciation.jl           # NoSpeciation, ThresholdSpeciation
│   ├── evaluators.jl           # TableFitnessEvaluator (from evolution.jl)
│   ├── algorithm.jl            # GeneticProgramming, IslandModel structs
│   ├── solve.jl                # solve() entry points and run_evolution!
│   ├── result.jl               # GPResult
│   └── defaults.jl             # default_function_set(), sensible presets
├── ext/
│   ├── LLMOperatorExt.jl       # LLMMutationOperator (weakdep: HTTP.jl)
│   └── DynExprExt.jl           # TreeGenome (weakdep: DynamicExpressions.jl)
├── test/
│   ├── runtests.jl
│   ├── unit/
│   │   ├── test_genome.jl
│   │   ├── test_operators.jl
│   │   ├── test_speciation.jl
│   │   └── test_evaluators.jl
│   └── benchmarks/
│       ├── max_ones.jl         # from Wallace.jl
│       ├── symbolic_regression.jl  # x², sin(x), etc.
│       ├── boolean_parity.jl   # from Wallace.jl
│       ├── tsp_heuristic.jl    # from Wallace.jl
│       └── lorenz_ctrnn.jl     # from DTSE work
└── docs/
    ├── make.jl
    └── src/
        ├── index.md
        ├── quickstart.md
        ├── genome_interface.md  # how to implement AbstractGenome
        ├── operator_interface.md
        └── llm_operator.md
```

### Project.toml structure

```toml
[deps]
Random = "9a3f8284-..."   # stdlib, always available

[weakdeps]
HTTP = "..."
DynamicExpressions = "..."

[extensions]
LLMOperatorExt = ["HTTP"]
DynExprExt = ["DynamicExpressions"]

[compat]
julia = "1.10"
```

Zero mandatory external dependencies. The framework runs on stdlib
alone. Optional integrations are weakdeps loaded via extensions.

---

## 9. Wallace.jl Benchmark Suite

Mine the Wallace.jl repository for problem definitions and use them as
the canonical benchmark suite. These are the standard comparators the
EC literature expects.

| Benchmark | Source | What it tests |
|---|---|---|
| Max Ones | Wallace.jl examples | Simplest GA sanity check |
| Boolean Even Parity | Wallace.jl examples | Boolean GP expressiveness |
| Symbolic Regression (x²) | existing test suite | Already passing |
| Symbolic Regression (Koza suite) | Wallace.jl / Koza 1992 | Standard GP benchmark |
| Ant Trail (Santa Fe) | Wallace.jl examples | Control / sequencing GP |
| TSP (nearest neighbor heuristic) | Wallace.jl examples | Combinatorial |
| Lorenz attractor recovery | DTSE work | Scientific GP |
| CTRNN dynamics | neat-python / DTSE work | Temporal GP |

The Lorenz and CTRNN benchmarks are original contributions — nothing
in the existing EC literature benchmarks GP on continuous dynamical
system recovery in Julia. These differentiate the package.

---

## 10. Development Phases

### Phase 1 — Core framework (target: 2 weeks with Claude Code)

- [ ] Package skeleton with Project.toml, src/, test/, ext/
- [ ] All abstract types and interface docstrings (`abstractions.jl`)
- [ ] `ExprGenome` wrapping existing codegen.jl/evolution.jl
- [ ] `GeneticProgramming` algorithm struct and `solve` entry point
- [ ] `TableFitnessEvaluator` (migrate from evolution.jl)
- [ ] `SubtreeMutation`, `PointMutation`, `SubtreeCrossover`
- [ ] `TournamentSelection`, `NoSpeciation`
- [ ] `GPResult` with fitness history
- [ ] Unit tests for all abstract interface methods
- [ ] Max Ones and x² benchmarks passing

### Phase 2 — Speciation and operators (target: 1 week)

- [ ] `ThresholdSpeciation` with stagnation culling
- [ ] `HoistMutation`, `ExpansionMutation`
- [ ] `IslandModel` algorithm struct
- [ ] Boolean parity and Koza symbolic regression benchmarks
- [ ] Bloat penalty via `complexity()` in fitness scoring

### Phase 3 — LLM operator extension (target: 1 week)

- [ ] `ext/LLMOperatorExt.jl` with `LLMMutationOperator`
- [ ] `serialize`/`deserialize` for `ExprGenome`
- [ ] Graceful degradation on LLM failure
- [ ] Integration test: LLM operator on symbolic regression
- [ ] Document the FunSearch/AlphaEvolve connection explicitly

### Phase 4 — DynamicExpressions extension and registration (target: 1 week)

- [ ] `ext/DynExprExt.jl` with `TreeGenome` backed by `Node{T}`
- [ ] Benchmark showing evaluation speedup vs `ExprGenome`
- [ ] Santa Fe Ant Trail benchmark
- [ ] Register in Julia General registry
- [ ] Announce on Julia Discourse

---

## 11. Future-Proofing Checklist

Before each release, verify:

- [ ] No uses of `Base.Compiler.*` or `Core.Compiler.*`
- [ ] No runtime struct type generation via `@eval`
- [ ] All `@eval` uses are for user-provided code (evolved programs),
      not framework structure
- [ ] `pkg> test` passes on Julia 1.10 (LTS) and current stable
- [ ] CI runs on both LTS and nightly to catch upcoming breakage early
- [ ] No dependencies on packages that themselves use compiler internals
      (check with `Pkg.dependencies()`)
- [ ] Package extensions declared in `[weakdeps]`, not `[deps]`

The CI matrix (GitHub Actions) should run:
```
julia-version: ['1.10', '1.11', 'nightly']
os: [ubuntu-latest, windows-latest]
```

Running on nightly gives 6–12 months of warning before any
compatibility break reaches a stable release.

---

## 12. Connection to Broader Work

### NeatEvolution.jl

`GraphGenome <: AbstractGenome` (Phase 5+) would represent neural
topologies and implement the NEAT genetic operators. NeatEvolution.jl
could either be reimplemented as a genome type here or remain
independent with an adapter. The key point: GenProg.jl provides the
evolutionary loop infrastructure that NeatEvolution.jl currently
bundles internally, so the NEAT genome type gains island models,
speciation, LLM operators, and the Problem/Algorithm/Solve API for
free.

### Mind Kernel

`evolve_routing_program` from `codegen_interfaces.jl` becomes a
one-liner using the public API:

```julia
result = solve(
    GPProblem(routing_evaluator, ExprGenome; function_set=fset),
    GeneticProgramming(pop_size=50, generations=100)
)
best_fn = compile!(DomainRoutingProgram(result.best_genome))
```

The sleep cycle triggers `solve`; the LLM operator (if enabled) uses
the Mind Kernel's own Qwen instance as the mutation model — closing the
self-modification loop that the Mind Kernel design anticipates.

### DTSE / neat-python citations

The Lorenz and CTRNN benchmarks will produce citable results. A paper
describing GenProg.jl with those benchmarks would sit naturally
alongside the DTSE paper and the NeatEvolution.jl announcement, giving
CodeReclaimers three related Julia/EC contributions in the literature
within a short window.
