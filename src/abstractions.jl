"""
    AbstractGenome

Base type for all genome representations in Arborist.

A concrete subtype `G <: AbstractGenome` participates in evolution by
providing the operations the solve path needs. At minimum:

- `mutate(op, g::G, rng::AbstractRNG) -> G` for each mutation operator
  that dispatches on `G` (or a direct `mutate(g::G, rng)` method for
  genome types that use direct dispatch, e.g. `AntGenome`, `GraphGenome`).
- `crossover(op, g1::G, g2::G, rng::AbstractRNG) -> Tuple{G, G}` (or a
  direct `crossover(g1, g2, rng)` method for direct-dispatch genomes).
- `distance(g1::G, g2::G) -> Float64` — used by `ThresholdSpeciation`.
- `complexity(g::G) -> Real` — used by bloat penalty and
  `ParsimonyEvaluator`.
- `serialize(g::G) -> String` — used by the LLM operator and logging.

Population initialization is genome-specific. Each genome type defines
its own construction path invoked from the matching `solve` method; the
signature is not fixed — `ExprGenome` uses a `GenState`, `TreeGenome`
takes an `OperatorEnum` and feature count, `AntGenome` takes a primitive
set, and `GraphGenome` takes input/output counts. See the per-genome
`solve(::GPProblem{G,E}, ::GeneticProgramming)` methods.

`deserialize(::Type{G}, s::String, ctx...)` is required only when using
the LLM mutation operator on `G`; its extra arguments depend on `G`
(e.g. `GenState` for `ExprGenome`, `(OperatorEnum, n_features)` for
`TreeGenome`). `LLMMutationOperator` currently dispatches on `ExprGenome`
only.
"""
abstract type AbstractGenome end

"""
    AbstractEvaluator

Base type for fitness evaluators.

Any concrete subtype `E <: AbstractEvaluator` must implement:

- `evaluate(e::E, f::Function) -> Float64` (lower is better)
- `input_signature(e::E) -> Dict{Symbol, DataType}`
- `output_signature(e::E) -> Dict{Symbol, DataType}`

Optionally, evaluators that can decompose fitness into independent per-case
losses (e.g. per-row MSE, per-sample squared error) may implement
`evaluate_cases(g::AbstractGenome, e::E) -> Vector{Float64}`. This is
required for lexicase selection; evaluators that cannot meaningfully
decompose (e.g. `AntEvaluator`, `EpisodicEvaluator`) should leave it
unimplemented — lexicase will then raise a clear `MethodError`.
"""
abstract type AbstractEvaluator end

"""
    evaluate_cases(g::AbstractGenome, e::AbstractEvaluator) -> Vector{Float64}

Per-case loss vector (lower = better) for evaluators that can decompose
their fitness into independent cases (per-row, per-sample). Used by
lexicase selection.

No default implementation: evaluators that can support lexicase must opt in
explicitly. If not implemented, calling it raises `MethodError`.
"""
function evaluate_cases end

"""
    AbstractMutationOperator

Base type for mutation operators.

Concrete subtypes define `mutate(op, g::AbstractGenome, rng::AbstractRNG) -> AbstractGenome`.
"""
abstract type AbstractMutationOperator end

"""
    AbstractCrossoverOperator

Base type for crossover operators.

Concrete subtypes define `crossover(op, g1::AbstractGenome, g2::AbstractGenome, rng::AbstractRNG) -> Tuple`.
"""
abstract type AbstractCrossoverOperator end

"""
    AbstractSelectionStrategy

Base type for parent selection strategies (e.g., tournament selection).
"""
abstract type AbstractSelectionStrategy end

"""
    AbstractSpeciation

Base type for speciation strategies.
"""
abstract type AbstractSpeciation end

"""
    AbstractEvolutionaryAlgorithm

Base type for evolutionary algorithm configurations.
"""
abstract type AbstractEvolutionaryAlgorithm end

"""
    AbstractEvolutionResult

Base type for results returned by `solve`.
"""
abstract type AbstractEvolutionResult end

"""
    AbstractTopology

Base type for island migration topologies. Concrete subtypes define
`migration_targets(t, i, n_islands, rng)` returning destination island indices.
"""
abstract type AbstractTopology end
