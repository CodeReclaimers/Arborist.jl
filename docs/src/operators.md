# Operators

Operators are composable: `GeneticProgramming` takes a
`Vector{AbstractMutationOperator}` and a
`Vector{AbstractCrossoverOperator}` and applies a uniform random
choice per offspring.

## Mutation Operators

For `ExprGenome` / `TreeGenome` / `ADFGenome`:

- **`SubtreeMutation()`** — replace a random subtree with a new random
  one. Standard canonical mutation.
- **`PointMutation()`** — modify a single node (variable, literal, or
  operator). Smaller steps than `SubtreeMutation`; good for
  fine-tuning once the structure is close.
- **`HoistMutation()`** — replace a subtree with one of its own
  children. A bloat-reduction operator; shrinks expressions.
- **`ExpansionMutation()`** — wrap a leaf in a function call. The
  complement to `HoistMutation`; grows expressions.

For `GraphGenome` (via `neat_defaults()`):

- **`WeightPerturbMutation`** — perturb an existing connection weight
  by a normally-distributed delta.
- **`WeightReplaceMutation`** — resample an existing connection weight.
- **`AddConnectionMutation`** — add a new connection between two
  previously unconnected nodes, assigning a new innovation number.
- **`AddNodeMutation`** — insert a node into an existing connection,
  splitting it into two. Assigns two new innovation numbers. The new
  hidden node's activation is sampled uniformly from the
  `hidden_activations` keyword (default `[:sigmoid, :tanh, :relu]`).
- **`ToggleConnectionMutation`** — flip the `enabled` flag on a random
  connection.

`GraphEvaluator` looks every activation symbol up in the
`ACTIVATION_FNS` dictionary, which ships with the canonical CPPN /
HyperNEAT activation set:

| Symbol      | Function                                           |
|-------------|----------------------------------------------------|
| `:sigmoid`  | NEAT-style steepened logistic `1 / (1 + exp(-4.9·x))` |
| `:tanh`     | hyperbolic tangent                                 |
| `:relu`     | `max(0, x)`                                        |
| `:identity` | pass-through                                       |
| `:gauss`    | `exp(-x²)`                                         |
| `:sin`      | `sin(x)`                                           |
| `:abs`      | `abs(x)`                                           |
| `:step`     | Heaviside, `1.0` if `x > 0` else `0.0`             |

`AddNodeMutation` and `NEATDefaultMutation` *do not* default to the
full set — they default `hidden_activations` to `[:sigmoid, :tanh,
:relu]`. To enable the CPPN activations during structural search,
construct the operator explicitly:

```julia
mutation_ops = AbstractMutationOperator[
    NEATDefaultMutation(;
        hidden_activations = [:sigmoid, :tanh, :gauss, :sin, :abs],
    ),
]
crossover_ops = AbstractCrossoverOperator[NEATCrossover()]
```

`neat_defaults()` itself takes no arguments and returns the standard
three-activation defaults — use it for vanilla NEAT and bypass it
when you need a custom activation set.

Custom activations can also be registered at runtime by assigning
into `ACTIVATION_FNS` before constructing the evaluator.

For `ExprGenome` and `GraphGenome`:

- **`LLMMutationOperator`** — use an LLM to produce a semantically
  meaningful variation. The `ExprGenome` path serializes a Julia
  source string; the `GraphGenome` path serializes the
  node-and-connection text format and uses
  `deserialize(GraphGenome, ...; reassign_innovations=true)` to keep
  LLM-generated innovation IDs disjoint from the parent pool. When
  pairing the operator with `GraphGenome`, override `fallback_op` with
  a NEAT-compatible operator (typically `NEATDefaultMutation()`) — the
  default `SubtreeMutation()` fallback only dispatches on `ExprGenome`.
  See [LLM Operator](llm_operator.md) and the
  [tutorial](tutorials/llm_binpacking.md).

### Per-operator depth and size caps

Every mutation operator on tree-shaped genomes (`SubtreeMutation`,
`PointMutation`, `HoistMutation`, `ExpansionMutation`) and the
`SubtreeCrossover` operator accept optional `max_depth` and `max_size`
keyword arguments:

```julia
SubtreeMutation(; max_depth=8, max_size=64)
SubtreeCrossover(; max_depth=12)
```

When set, the operator measures the resulting child's depth via
[`tree_depth`](@ref) (longest root-to-leaf path) and total node count
via [`complexity`](@ref). If either cap is exceeded the operator
returns the parent unchanged rather than producing an oversized
offspring. Both default to `nothing` (no cap). The caps are evaluated
per-operator, so different operators in the same `mutation_ops` vector
can carry different limits — useful for combining a permissive
exploration mutation with a tighter bloat-control mutation.

`tree_depth` is exported and is implemented for every tree-shaped
genome (`ExprGenome`, `TreeGenome`, `AntGenome`, `ADFGenome`); call it
directly to inspect a genome's structure.

### Example: mixing mutation operators

```julia
algorithm = GeneticProgramming(
    pop_size      = 100,
    generations   = 200,
    mutation_rate = 0.4,
    mutation_ops  = [
        SubtreeMutation(),
        PointMutation(),
        HoistMutation(),         # adds bloat-reduction pressure
    ],
)
```

## Crossover Operators

- **`SubtreeCrossover()`** — swap compatible subtrees between two
  parents (types must match at the swap points).
- **`NEATCrossover()`** — innovation-number-aligned crossover for
  `GraphGenome`. Disjoint and excess genes inherit from the fitter
  parent; matching genes are chosen randomly.

## Selection Strategies

- **`TournamentSelection(k)`** — select the best of `k` uniformly
  sampled individuals. Default for `GeneticProgramming` is
  `TournamentSelection(3)`.
- **`LexicaseSelection()`** — filter the population case-by-case,
  keeping only individuals best-on-the-current-case, randomizing case
  order per selection. Requires the evaluator to implement
  `evaluate_cases(g, e)`. Returns `MethodError` if not supported.
- **`EpsilonLexicaseSelection(; epsilon=0.0)`** — lexicase with a
  per-case tolerance band. `epsilon=0.0` auto-tunes using MAD per
  case (La Cava et al. 2016). Empirically more reliable than strict
  lexicase on continuous-output regression.
- **`FitnessProportionateSelection()`** — classical roulette-wheel
  selection adapted to the lower-is-better convention. Weight is
  `1 / (ε + f_i − f_min)` so that the best individual carries the
  largest weight and equal-fitness populations degenerate to uniform
  sampling. Sensitive to fitness scaling; prefer `RankSelection` when
  fitnesses span many orders of magnitude.
- **`RankSelection(; selection_pressure=1.5)`** — linear-ranking
  selection. Individuals are ranked by fitness and sampled with
  probability that grows linearly in rank. `selection_pressure ∈
  [1.0, 2.0]`; `1.0` reduces to uniform sampling and `2.0` is the
  maximum linear pressure.
- **`TruncationSelection(; ratio=0.5)`** — keep the top `ratio`
  fraction of the population by fitness, then sample uniformly from
  that pool. Strong, fast convergence; loses diversity quickly.

### Example: epsilon-lexicase selection

```julia
algorithm = GeneticProgramming(
    pop_size    = 200,
    generations = 100,
    selection   = EpsilonLexicaseSelection(),
)
```

The solve loop detects `needs_cases(selection)` and materializes the
per-individual per-case fitness matrix before each generation.
