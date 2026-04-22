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
Concrete subtypes may also override `operator_name(op) -> Symbol` to expose a
friendly key for RunLog's per-operator tallies. The default derives the name
from the struct type.
"""
abstract type AbstractMutationOperator end

"""
    operator_name(op) -> Symbol

Stable name for an operator, used as the key in `GenerationLog.operator_attempted`
/ `operator_success`. Default: the concrete type's `nameof`.
"""
operator_name(op::AbstractMutationOperator) = Symbol(nameof(typeof(op)))
operator_name(op) = Symbol(nameof(typeof(op)))

"""
    AbstractCrossoverOperator

Base type for crossover operators.

Concrete subtypes define `crossover(op, g1::AbstractGenome, g2::AbstractGenome, rng::AbstractRNG) -> Tuple`.
May also override `operator_name(op) -> Symbol` for RunLog tallies.
"""
abstract type AbstractCrossoverOperator end

"""
    AbstractSelectionStrategy

Base type for parent selection strategies (e.g., tournament selection,
lexicase selection).

Concrete subtypes must implement:

- `select_parent(s::S, selection_fitnesses::Vector{Float64}, case_fitnesses, rng)`
  returning the integer index of the selected parent in `genomes` /
  `selection_fitnesses`. Case-based strategies (lexicase) use the matrix;
  scalar-fitness strategies (tournament) ignore it.
- `needs_cases(s::S) -> Bool` (default `false`). Strategies that return
  `true` cause the solve loop to materialize a per-case fitness vector
  for every individual each generation via `evaluate_cases`. Returning
  `true` requires the evaluator to implement `evaluate_cases`; otherwise
  the solve loop raises `MethodError` the first time it tries.
"""
abstract type AbstractSelectionStrategy end

"""
    select_parent(s::AbstractSelectionStrategy, selection_fitnesses, case_fitnesses, rng) -> Int

Select a parent index. `selection_fitnesses::Vector{Float64}` is the
sharing-adjusted scalar fitness used by classical strategies (lower is
better). `case_fitnesses::Union{Nothing, Vector{Vector{Float64}}}` is the
per-individual per-case loss matrix used by lexicase strategies (same
convention: lower is better; `nothing` when `needs_cases(s) == false`).
"""
function select_parent end

"""
    needs_cases(s::AbstractSelectionStrategy) -> Bool

Return `true` if the strategy requires per-case fitnesses
(`evaluate_cases`-derived). Default: `false`.
"""
needs_cases(::AbstractSelectionStrategy) = false

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
