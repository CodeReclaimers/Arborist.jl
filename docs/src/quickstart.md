# Quick Start

## Two-Line Symbolic Regression

The simplest way to use GenProg.jl for symbolic regression:

```julia
using GenProg, DynamicExpressions
const DynExt = Base.get_extension(GenProg, :DynExprExt)
const TreeGenome = DynExt.TreeGenome

evaluator = DynExt.SymbolicRegressionEvaluator(
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

## XOR with NEAT (GraphGenome)

```julia
using GenProg

input_data = Float64[0 0 1 1; 0 1 0 1]
output_data = Float64[0 1 1 0]

result = solve(
    GPProblem(GraphEvaluator(input_data, output_data), GraphGenome; seed=42),
    GeneticProgramming(
        pop_size=150, generations=150,
        mutation_rate=0.5, crossover_rate=0.3,
        speciation=ThresholdSpeciation(threshold=3.0)
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
