# TreeGenome — DynamicExpressions.jl Backend

## Why Two Genome Types

Arborist.jl provides two genome types for different problem classes:

### TreeGenome (DynamicExpressions.jl backed)

Appropriate for:
- **Pure function approximation**: symbolic regression, system identification
- Problems where the genome is `f(x1, x2, ...) -> y` with no side effects
- Workloads where evaluation speed over large datasets matters
- Boolean expression problems (parity, multiplexer) expressible as trees

TreeGenome evaluates expression trees directly via DynamicExpressions.jl's
vectorized evaluation engine. No `@eval` compilation is needed, making
evaluation 10-100x faster than ExprGenome for large datasets.

### ExprGenome (@eval backed)

Required for:
- Programs with **control flow**: loops, conditionals, break/continue
- Programs with **sequential mutable state**
- **Side-effectful** operations (robotics control, the ant trail)
- General **program synthesis** where arbitrary Julia is needed

The Santa Fe Ant Trail is a canonical example of the second category.
The Koza symbolic regression suite is a canonical example of the first.

## Accessing TreeGenome

`TreeGenome` and `TreeFitnessEvaluator` are exported directly from `Arborist`.
DynamicExpressions.jl is a hard dependency of Arborist, so no extension dance
is needed — just `using` both packages:

```julia
using Arborist
using DynamicExpressions
```

## Quick-Start Example

```julia
using Arborist, DynamicExpressions

# Define operators
operators = OperatorEnum(;
    binary_operators = [+, -, *, /],
    unary_operators  = [abs]
)

# Build dataset: y = x^2
xs = Float32.(range(-1, 1, length=100))
X = reshape(xs, 1, :)         # 1 feature × 100 samples
y = xs .^ 2

# Create evaluator and problem
evaluator = TreeFitnessEvaluator(X, y, operators)
problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)

# Run GP
algorithm = GeneticProgramming(
    pop_size = 100,
    generations = 200,
    mutation_rate = 0.4,
    crossover_rate = 0.2,
)

result = solve(problem, algorithm; verbose=true)
println("Best fitness: ", result.best_fitness)
println("Best tree: ", serialize(result.best_genome))
```

## Mutation Operators

TreeGenome implements three mutation types, chosen with equal probability:

1. **Point mutation**: replaces a random node with a new random subtree of depth ≤ 2
2. **Constant perturbation**: adds Gaussian noise (scale 0.1) to a random constant
3. **Hoist mutation**: replaces a subtree with one of its children (bloat reduction)

These are invoked automatically when using `SubtreeMutation()` or `PointMutation()`
in the algorithm's `mutation_ops` — the dispatch routes to the TreeGenome
implementation.

## Known Limitations

- **Distance metric**: TreeGenome's `distance` function uses absolute node count
  difference, which is less semantically meaningful than ExprGenome's
  symmetric-difference metric. This is sufficient for `ThresholdSpeciation`
  but could be improved in future versions.

- **Deserialize**: Parsing expression trees from arbitrary string representations
  is not implemented. `deserialize` returns `nothing`. The LLM operator
  is not yet supported for TreeGenome.

- **Boolean problems**: Boolean logic must be encoded as Float32 operations
  (0.0 = false, 1.0 = true) since DynamicExpressions.jl operates on numeric
  types. Custom boolean operators (AND, OR, XOR, etc.) must be defined as
  Float32 → Float32 functions and passed via `OperatorEnum`.
