# Evaluators

An evaluator defines the fitness landscape. Arborist evaluators are
composable — for example, `ParsimonyEvaluator` wraps any
single-objective evaluator into a two-objective `(fitness,
complexity)` problem. All evaluators follow the Arborist convention
of **lower fitness is better** (mean squared error, negated reward,
etc.).

## `TableFitnessEvaluator`

Evaluates `ExprGenome` programs against a table of input/output
examples. Fitness is mean squared error. Returns `Inf` if more than
50% of rows fail or exceed a per-row time limit.

```@example evaluator-table
using Arborist
xs = Float32.(range(-1, 1, length=20))
input_rows  = [Dict(:x => Float32(v)) for v in xs]
output_rows = [Dict(:y => Float32(v^2 + v)) for v in xs]
evaluator   = TableFitnessEvaluator(
    Dict(:x => Float32),    # input column types
    Dict(:y => Float32),    # output column types
    input_rows, output_rows,
)
nothing # hide
```

## `TreeFitnessEvaluator`

Evaluates `TreeGenome` expression trees directly over a data matrix.
No `@eval` needed. Roughly 8× faster than `TableFitnessEvaluator` on
the Koza benchmark suite (1000-point dataset).

```@example evaluator-tree
using Arborist, DynamicExpressions
xs = Float32.(range(-1, 1, length=20))
X = reshape(xs, 1, :)
y = xs.^2 .+ xs
operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[abs])

evaluator = TreeFitnessEvaluator(X, y, operators)
nothing # hide
```

Implements `evaluate_cases(g, e)` — per-sample squared error — so it
works directly with `LexicaseSelection` and
`EpsilonLexicaseSelection`.

## `SymbolicRegressionEvaluator`

Convenience wrapper over `TreeFitnessEvaluator` that takes a target
function and sampling domain instead of pre-materialized `X`, `y`:

```@example evaluator-sr
using Arborist, DynamicExpressions

evaluator = SymbolicRegressionEvaluator(
    x -> x^2 + x;
    domain = (-1f0, 1f0),
    points = 20,
)
nothing # hide
```

## `ParsimonyEvaluator`

Wraps any single-objective `AbstractEvaluator` into a two-objective
`["fitness", "complexity"]` evaluator. Recovers the accuracy /
parsimony tradeoff as a real Pareto front (see the
[Multi-Objective tutorial](tutorials/nsga2_parsimony.md)) rather
than a single bloat-penalty compromise.

```@example evaluator-parsimony
using Arborist, DynamicExpressions
inner     = SymbolicRegressionEvaluator(x -> x^2 + x, domain=(-1f0, 1f0), points=20)
evaluator = ParsimonyEvaluator(inner)

result = solve(
    GPProblem(evaluator, TreeGenome{Float32}; seed=42),
    NSGAII(pop_size=200, generations=100),
)
println("Pareto front size: ", length(result.pareto_front))
```

## `GraphEvaluator`

Evaluates `GraphGenome` neural networks by forward propagation over
labeled input/output data. Handles feedforward nets by default;
`allow_recurrent=true` with `relaxation_passes=N` supports recurrent
topologies (treats the sample set as a time sequence with persistent
node state).

```@example evaluator-graph
using Arborist
# XOR truth table — 2 inputs × 4 samples; 1 output × 4 samples.
input_matrix  = Float64[0 0 1 1; 0 1 0 1]
output_matrix = reshape(Float64[0, 1, 1, 0], 1, 4)
evaluator = GraphEvaluator(input_matrix, output_matrix;
                           allow_recurrent = false)
nothing # hide
```

## `EpisodicEvaluator`

Closed-loop control-task evaluator for `GraphGenome`. Declarative:
provide `(initial_state, dynamics, reward, done, observe,
decode_action)` callables, and the evaluator runs `n_episodes`
rollouts per fitness call and returns `-mean_reward`. See the
[Cart-Pole tutorial](tutorials/control_cartpole.md) for a full
worked example.

## `AntEvaluator`

Evaluates `AntGenome` programs by running an ant simulation. Fitness
is the number of uneaten food pellets. Used by the Santa Fe Ant
Trail benchmark. Note: not thread-safe
(see [Genome Types — Known Limitations](genome_types.md#Known-Limitations)).

## Custom evaluators

Custom evaluators subtype `AbstractEvaluator` and implement:

- `evaluate(e::E, g) -> Float64` (or `evaluate_genome(g, e)` — both
  are accepted by the solve loop).
- `input_signature(e) -> Dict{Symbol, DataType}`
- `output_signature(e) -> Dict{Symbol, DataType}`

Optionally implement `evaluate_cases(g, e) -> Vector{Float64}` for
lexicase selection support. Multi-objective evaluators subtype
`AbstractMultiObjectiveEvaluator` and implement `evaluate_multi` and
`objective_names`.
