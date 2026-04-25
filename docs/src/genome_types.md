# Genome Types

Arborist.jl provides five genome types for different problem classes.

## Decision Table

| Feature | TreeGenome | ExprGenome | AntGenome | GraphGenome | ADFGenome |
|---|---|---|---|---|---|
| Backend | DynamicExpressions.jl | @eval | @eval | Custom | DynamicExpressions.jl |
| Speed | Fast (vectorized) | Slow (compilation) | Slow (compilation) | Medium | Fast (post-expansion) |
| Control flow | No | Yes | Yes | N/A | No |
| Side effects | No | Yes | Yes | N/A | No |
| Variables | Feature indices | Typed vars | Primitives | Weights | Feature indices + ARGs |
| Best for | Symbolic regression | Program synthesis | Agent control | Neural topology | Reusable subroutines |

## TreeGenome

`TreeGenome{T}` wraps a `DynamicExpressions.Node{T}` expression tree.
Evaluation is vectorized over data matrices without any `@eval`
compilation, making it 8–10× faster than ExprGenome for symbolic
regression. Use this for pure function approximation problems.

DynamicExpressions.jl is a hard dependency of Arborist, so `TreeGenome`
and `TreeFitnessEvaluator` are exported directly from the `Arborist`
module — no extension lookup required.

See the [Symbolic Regression tutorial](tutorials/symbolic_regression.md)
for an end-to-end walkthrough on the Koza-1 benchmark.

## ExprGenome

`ExprGenome` represents programs as Julia `Expr` ASTs. The program body
is a `Vector{Expr}` of typed assignment statements, compiled via
`@eval` into callable functions. Supports loops, conditionals, and
mutable state. Use this for general program synthesis.

The `LLMMutationOperator` dispatches on `ExprGenome` and `GraphGenome`;
see the [LLM-Enhanced GP tutorial](tutorials/llm_binpacking.md) and
[`examples/bin_packing.jl`](https://github.com/CodeReclaimers/Arborist.jl/blob/master/examples/bin_packing.jl)
for a worked `ExprGenome` heuristic-discovery example, and the
[LLM Operator page](llm_operator.md) for the `GraphGenome` path.

## AntGenome

`AntGenome` represents imperative programs that call side-effectful
primitives (move, turn, sense). The program is a single `:block`
expression of nested primitive calls and `if`-then-else control flow.
Use this for agent control problems like the Santa Fe Ant Trail.

## GraphGenome

`GraphGenome` represents neural network topologies following the NEAT
encoding (Stanley & Miikkulainen, 2002). Supports structural mutation
(add node, add connection), innovation-number-aligned crossover, and
compatibility-distance-based speciation. Use this for neural topology
evolution.

Two evaluator types pair with `GraphGenome`:
- `GraphEvaluator` for supervised tasks (labeled input/output data),
  with `allow_recurrent=true` optional for non-Markovian problems.
- `EpisodicEvaluator` for closed-loop control tasks — see the
  [Cart-Pole tutorial](tutorials/control_cartpole.md).

The [NEAT XOR tutorial](tutorials/neat_xor.md) walks through innovation
numbers and speciation.

## ADFGenome

`ADFGenome{T}` is a TreeGenome variant that learns *Automatically
Defined Functions* (Koza, 1992): a main expression tree plus N ADF
subroutines that can be called from the main tree like ordinary
binary primitives. Useful when the target has reusable structure —
the evolved program size can stay small even as the expressive
complexity grows. ADF-from-ADF calls are disallowed to prevent
infinite expansion.

## Inspecting Genome Structure

Every genome supports `complexity(g)` (total node count) for bloat
tracking. Tree-shaped genomes (`TreeGenome`, `ExprGenome`, `AntGenome`,
`ADFGenome`) additionally implement `tree_depth(g)` — the longest
root-to-leaf path. Both are exported and used by mutation / crossover
operators that enforce `max_depth` / `max_size` caps; see
[Operators](operators.md#per-operator-depth-and-size-caps).

For visualization, `to_dot(g)` produces a Graphviz DOT document for
every concrete genome — `TreeGenome`, `ExprGenome`, `ADFGenome`,
`AntGenome`, and `GraphGenome`. Tree-shaped genomes render as a
top-down DAG of operator / constant / variable nodes; `GraphGenome`
renders left-to-right with distinct node shapes by role
(input / output / bias / hidden) and edges labeled by connection
weight, with disabled connections drawn dashed:

```julia
open("genome.dot", "w") do io
    print(io, to_dot(genome))
end
run(`dot -Tsvg genome.dot -o genome.svg`)
```

The `dot` binary itself is *not* a runtime dependency of Arborist —
`to_dot` produces the source document only and the user invokes
Graphviz separately. The two-argument form `to_dot(io, g)` streams
directly to an `IO` target.

## Known Limitations

These constraints are baked into the current implementation and
important to know before committing to a genome choice:

- **ExprGenome `serialize` / `deserialize` round-trip is partial**
  (~80% reliability). `repr()`-style `Float32(literal)` forms fail
  type-checking on round-trip. LLMMutationOperator handles failures
  gracefully (silent fallback to a classical operator), but
  checkpoint/resume or cross-process migration can lose a fraction
  of individuals. Plan for this.
- **ExprGenome `@eval` grows Julia's method table** monotonically
  across generations. Long runs (thousands of generations × hundreds
  of individuals) accumulate tens of thousands of methods, slowing
  dispatch. `TreeGenome` avoids this via DynamicExpressions' compiled
  eval and is preferred for anything that will run more than a few
  hundred generations.
- **AntGenome is not thread-safe.** The simulator uses a
  module-level `Ref` for state. `GeneticProgramming(; parallel=true)`
  with `AntGenome` raises a runtime error. The bin-packing and sorting
  examples demonstrate the thread-local-state workaround for custom
  `@eval`-based genomes.
- **Distributed NEAT innovation matching is disjoint-range, not
  content-aware.** Each worker gets a unique innovation ID range via
  `init_innovation_range!((island_id - 1) * INNOVATION_STRIDE)`. The
  cost: structurally identical mutations on different workers receive
  different IDs, so NEAT crossover treats them as disjoint rather
  than aligned. Per-generation cross-worker innovation dedup is not
  implemented.
- **EpisodicEvaluator is not parallel-ready for stateful
  environments.** The declarative API is structurally thread-safe
  (no shared state across evaluations), but if your `dynamics`
  closure captures mutable state, `parallel=true` will race. The
  planned `StatefulEpisodicEvaluator` (see `CLAUDE.md` Deferred
  Research Roadmap) will address this when it lands.
