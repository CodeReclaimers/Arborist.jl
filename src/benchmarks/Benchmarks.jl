"""
    Arborist.Benchmarks

Canonical genetic-programming and neuroevolution benchmark problems as
reusable data generators. Every function returns a `NamedTuple` carrying
the dataset (or environment callables), along with the fields a user
needs to build an `Arborist` problem + evaluator: shape metadata,
target-function descriptions, sensible success thresholds, and a
human-readable name.

Generators are decoupled from evaluator choice. A user can feed
`Benchmarks.nguyen(1).X, Benchmarks.nguyen(1).y` into a
`TreeFitnessEvaluator`, an `ExprGenome` table evaluator, a
`NoveltySearchEvaluator` wrapping the same target, or a custom
behavior-fingerprinting evaluator. Likewise, control benchmarks return
the raw `(dynamics, reward, done, observe, initial_state,
decode_action)` tuple suitable for `EpisodicEvaluator`.

## Coverage

- **Symbolic regression**: `nguyen(n)` (Nguyen-1..10), `keijzer(variant)`
  (K-4, K-11), `koza(name)` (:quartic, :septic, :nonic), `pagie()`
  (Pagie-1).
- **Classification**: `iris()` (Fisher's 3-class), `two_spirals()`
  (Lang–Witbrock 1989).
- **Boolean**: `multiplexer(address_bits)`, `parity(n_bits)`.
- **Control (closed-loop)**: `cartpole()`, `mountain_car()`, `acrobot()`,
  `double_pole(; markovian=true)`.
- **Sequence / memory**: `sequence_memory(length)`, `sequence_recall(; delay)`.
- **Classic neuroevolution**: `xor_env()` (truth table in
  `GraphEvaluator`-ready shape).

## Usage sketch (symbolic regression)

```julia
using Arborist
using DynamicExpressions

prob = Arborist.Benchmarks.nguyen(1)
ops = OperatorEnum(binary_operators=[+, -, *, /],
                   unary_operators=[sin, cos, exp])
evaluator = TreeFitnessEvaluator(prob.X, prob.y, ops)
problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
result = solve(problem, GeneticProgramming())
```

## Usage sketch (control)

```julia
cp = Arborist.Benchmarks.cartpole()
evaluator = EpisodicEvaluator(
    cp.n_states, cp.n_actions,
    cp.initial_state, cp.dynamics,
    cp.reward, cp.done, cp.observe, cp.decode_action;
    max_steps=cp.max_steps, n_episodes=5,
)
```
"""
module Benchmarks

using Random

# Protected Nguyen-flavor operators (matches the historical benchmark
# semantics — division-by-zero lifted to 1.0 at threshold 1e-6 rather
# than the library's more conservative 1e-10). Re-exported here so
# `using Arborist.Benchmarks` gives users a ready operator set for SR.
"""Protected division with the Nguyen-canonical 1e-6 threshold."""
nguyen_pdiv(a::T, b::T) where T<:AbstractFloat = abs(b) < T(1e-6) ? one(T) : a / b
"""Protected natural log: `log(x)` for positive `x`, else `0`."""
nguyen_plog(x::T) where T<:AbstractFloat = x > zero(T) ? log(x) : zero(T)
"""Protected square root: `sqrt(|x|)`."""
nguyen_psqrt(x::T) where T<:AbstractFloat = sqrt(abs(x))

# --- Canonical operator set for SR benchmarks ------------------------------

"""
    canonical_sr_operators(::Type{T}=Float32)

Return the operator set used across the Nguyen/Keijzer suite:
`[+, -, *, nguyen_pdiv]` binary, `[sin, cos, exp, nguyen_plog, nguyen_psqrt]`
unary. Requires `DynamicExpressions` to be imported by the caller.

Separate from the library-wide `default_protected_function_set()` because
the Nguyen tradition uses a slightly different `pdiv` threshold than the
library default. Users who prefer library-consistent protected operators
can build their own `OperatorEnum` with `Arborist.pdiv` / `plog` / `psqrt`.
"""
function canonical_sr_operators(::Type{T}=Float32) where T
    # Forward declaration; resolved when the caller has DynamicExpressions
    # in scope. We return a symbol-tuple spec instead of an OperatorEnum so
    # Benchmarks does not carry a hard dep on DynamicExpressions.
    return (
        binary = [+, -, *, nguyen_pdiv],
        unary  = [sin, cos, exp, nguyen_plog, nguyen_psqrt],
    )
end

include("symbolic_regression.jl")
include("classification.jl")
include("boolean.jl")
include("control.jl")
include("sequence.jl")

end # module Benchmarks
