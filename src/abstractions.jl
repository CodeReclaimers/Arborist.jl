"""
    AbstractGenome

Base type for all genome representations in Arborist.

Any concrete subtype `G <: AbstractGenome` must implement:

- `initialize(::Type{G}, problem::GPProblem) -> G`
- `mutate(g::G, rng::AbstractRNG) -> G`
- `crossover(g1::G, g2::G, rng::AbstractRNG) -> Tuple{G, G}`
- `distance(g1::G, g2::G) -> Float64`
- `complexity(g::G) -> Float64`
- `serialize(g::G) -> String`
- `deserialize(::Type{G}, s::String) -> Union{G, Nothing}`
"""
abstract type AbstractGenome end

"""
    AbstractEvaluator

Base type for fitness evaluators.

Any concrete subtype `E <: AbstractEvaluator` must implement:

- `evaluate(e::E, f::Function) -> Float64` (lower is better)
- `input_signature(e::E) -> Dict{Symbol, DataType}`
- `output_signature(e::E) -> Dict{Symbol, DataType}`
"""
abstract type AbstractEvaluator end

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
