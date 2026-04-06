# Evaluators

## TableFitnessEvaluator

Evaluates ExprGenome programs against a table of input/output examples. Fitness is mean squared error. Returns `Inf` if more than 50% of rows fail or exceed the time limit.

## TreeFitnessEvaluator

Evaluates `TreeGenome` expression trees directly over a data matrix. No `@eval` needed. Dramatically faster than `TableFitnessEvaluator` for large datasets.

```julia
using Arborist, DynamicExpressions
evaluator = TreeFitnessEvaluator(X, y, operators)
```

## SymbolicRegressionEvaluator

Convenience wrapper that generates a `TreeFitnessEvaluator` from a function and domain:
```julia
using Arborist, DynamicExpressions
evaluator = SymbolicRegressionEvaluator(
    x -> x^2 + x, domain=(-1f0, 1f0), points=20
)
```

## ParsimonyEvaluator

Wraps any `AbstractEvaluator` into a two-objective evaluator with `["fitness", "complexity"]`. Used together with the `NSGAII` algorithm to recover the accuracy/complexity tradeoff as a real Pareto front instead of a single bloat-penalty compromise.

```julia
inner = SymbolicRegressionEvaluator(x -> x^2 + x, domain=(-1f0, 1f0), points=20)
evaluator = ParsimonyEvaluator(inner)
result = solve(GPProblem(evaluator, TreeGenome{Float32}; seed=42),
               NSGAII(pop_size=200, generations=100))
```

## GraphEvaluator

Evaluates `GraphGenome` neural networks by forward propagation over input data.

## AntEvaluator

Evaluates `AntGenome` programs by running an ant simulation. Fitness is the number of uneaten food pellets.
