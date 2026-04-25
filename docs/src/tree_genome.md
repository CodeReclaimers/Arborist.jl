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
evaluation roughly 8x faster than ExprGenome on the Koza symbolic regression
suite (1000-point dataset); the gap widens further as dataset size grows.

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

```@example treegenome-using
using Arborist
using DynamicExpressions
nothing # hide
```

## Quick-Start Example

```@example treegenome-quick
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
    generations = 100,
    mutation_rate = 0.4,
    crossover_rate = 0.2,
)

result = solve(problem, algorithm)
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

## Ephemeral Random Constants

Both random tree creation and point-mutation leaf creation sample numeric
literals via an *ephemeral random constant* (ERC) sampler. The default is
`T(randn(rng))` — standard normal, scaled by `T`. To override, pass a
`(rng) -> T` callable as `GeneticProgramming(; constant_sampler=...)`. The
sampler is threaded into TreeGenome creation via task-local storage, so no
global state is mutated.

`erc_uniform(lo, hi)` is the canonical helper, returning a sampler that draws
uniformly from `[lo, hi]` (Koza-style ERCs):

```@example treegenome-erc
using Arborist
algorithm = GeneticProgramming(
    pop_size       = 100,
    generations    = 200,
    constant_sampler = erc_uniform(-5.0f0, 5.0f0),
)
nothing # hide
```

For other distributions, write a closure directly:

```@example treegenome-erc-loguniform
using Arborist
# Log-uniform over [0.1, 10.0] for problems where constants span orders of magnitude
algorithm = GeneticProgramming(;
    constant_sampler = rng -> Float32(exp(log(0.1) + log(100.0) * rand(rng))),
)
nothing # hide
```

`constant_sampler === nothing` (default) preserves the historical
`T(randn(rng))` behavior; the field is ignored by genome types other than
`TreeGenome`.

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
