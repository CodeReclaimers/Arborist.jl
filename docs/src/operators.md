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
  splitting it into two. Assigns two new innovation numbers.
- **`ToggleConnectionMutation`** — flip the `enabled` flag on a random
  connection.

For `ExprGenome` only:

- **`LLMMutationOperator`** — use an LLM to produce a semantically
  meaningful variation. See [LLM Operator](llm_operator.md) and the
  [tutorial](tutorials/llm_binpacking.md).

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
