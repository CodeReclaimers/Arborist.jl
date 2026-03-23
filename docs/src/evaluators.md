# Evaluators

## TableFitnessEvaluator

Evaluates ExprGenome programs against a table of input/output examples. Fitness is mean squared error. Returns `Inf` if more than 50% of rows fail or exceed the time limit.

## TreeFitnessEvaluator

Evaluates TreeGenome expression trees directly over a data matrix. No `@eval` needed. Dramatically faster than TableFitnessEvaluator for large datasets.

Available via the DynamicExpressions extension:
```julia
using DynamicExpressions
const DynExt = Base.get_extension(Arborist, :DynExprExt)
evaluator = DynExt.TreeFitnessEvaluator(X, y, operators)
```

## SymbolicRegressionEvaluator

Convenience wrapper that generates a TreeFitnessEvaluator from a function and domain:
```julia
evaluator = DynExt.SymbolicRegressionEvaluator(
    x -> x^2 + x, domain=(-1f0, 1f0), points=20
)
```

## GraphEvaluator

Evaluates GraphGenome neural networks by forward propagation over input data.

## AntEvaluator

Evaluates AntGenome programs by running an ant simulation. Fitness is the number of uneaten food pellets.
