# Quick Start

The snippets on this page are executed during the documentation build, so
the printed output below is what you actually get with these settings.

## Two-Line Symbolic Regression

The simplest way to use Arborist.jl for symbolic regression:

```@example quickstart-sr
using Arborist, DynamicExpressions

evaluator = SymbolicRegressionEvaluator(
    x -> x^4 + x^3 + x^2 + x,
    domain=(-1f0, 1f0), points=20
)

result = solve(
    GPProblem(evaluator, TreeGenome{Float32}; seed=42),
    GeneticProgramming(pop_size=100, generations=200)
)

println("Best fitness: ", result.best_fitness)
println("Best expression: ", serialize(result.best_genome))
```

The evolved expression at this point is bloated — that is normal output
for plain GP. To recover the underlying polynomial, pair the run with
`HoistMutation()` (bloat reduction) and the periodic constant-optimization
pass; see the
[Symbolic Regression tutorial](tutorials/symbolic_regression.md).

## XOR with NEAT (GraphGenome)

`GraphGenome` requires NEAT-compatible mutation and crossover operators
— the default `GeneticProgramming` operators dispatch on `ExprGenome` and
will be rejected by `_validate_ops`. Use `neat_defaults()` to get the
canonical `(mutation_ops, crossover_ops)` pair:

```@example quickstart-xor
using Arborist
reset_innovation_counter!()

input_data  = Float64[0 0 1 1; 0 1 0 1]
output_data = reshape(Float64[0, 1, 1, 0], 1, 4)

ops = neat_defaults()

result = solve(
    GPProblem(GraphEvaluator(input_data, output_data), GraphGenome; seed=42),
    GeneticProgramming(
        pop_size=150, generations=150,
        mutation_rate=0.5, crossover_rate=0.3,
        mutation_ops  = ops.mutation_ops,
        crossover_ops = ops.crossover_ops,
        speciation    = ThresholdSpeciation(threshold=3.0),
    )
)

println("Best fitness: ", result.best_fitness)
```

## Choosing the Right Genome Type

| Genome Type | Use Case | Speed | Control Flow |
|---|---|---|---|
| `TreeGenome` | Symbolic regression, function approximation | Fast (no @eval) | No |
| `ExprGenome` | General program synthesis | Slow (@eval) | Yes |
| `AntGenome` | Agent control (ant trail, robotics) | Slow (@eval) | Yes |
| `GraphGenome` | Neural topology (NEAT, XOR) | Medium | N/A |
