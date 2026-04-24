# NEAT Neural Topology Evolution: XOR

This tutorial solves the canonical NEAT benchmark — learning XOR by
evolving the *topology* as well as the weights of a small neural
network — via `GraphGenome` and the `neat_defaults()` operator set.
It mirrors Stanley & Miikkulainen (2002) closely.

## Setup

```julia
using Arborist
```

`GraphGenome` is a native Arborist type, not a DynamicExpressions
wrapper, so no other imports are needed.

## Why `reset_innovation_counter!()`?

NEAT assigns every structural mutation (add-node, add-connection) a
globally-unique *innovation number* so that crossover between two
topologies can align structurally homologous genes. The counter is
process-global for a single run and must be reset at the start of
each `solve` so that innovations from a previous run don't pollute
the new one.

```julia
reset_innovation_counter!()
```

Arborist does this automatically at the top of the `solve` method for
`GraphGenome` under single-threaded execution; the explicit call here
is only needed when combining runs in one Julia process (e.g. in a
script that calls `solve` repeatedly).

## Training data

XOR as a 2-input, 1-output lookup:

```julia
input_data  = Float64[0 0 1 1;
                      0 1 0 1]
output_data = Float64[0 1 1 0]
```

Shape convention: `input_data` is `n_inputs × n_samples`;
`output_data` is `n_outputs × n_samples` for multi-output problems,
or a flat vector when there is a single output (as here — the
`GraphEvaluator` broadcasts correctly).

```julia
evaluator = GraphEvaluator(input_data, output_data)
```

## Pick the NEAT operators

```julia
ops = neat_defaults()   # returns (mutation_ops, crossover_ops)
```

This yields five mutation operators (`WeightPerturbMutation`,
`WeightReplaceMutation`, `AddConnectionMutation`, `AddNodeMutation`,
`ToggleConnectionMutation`) and one crossover (`NEATCrossover`) — the
canonical NEAT operator set with reasonable per-operator rates baked
in.

## Configure the algorithm

```julia
algorithm = GeneticProgramming(
    pop_size       = 150,
    generations    = 150,
    mutation_rate  = 0.5,
    crossover_rate = 0.3,
    elitism        = 2,
    mutation_ops   = ops.mutation_ops,
    crossover_ops  = ops.crossover_ops,
    speciation     = ThresholdSpeciation(
        threshold        = 3.0,
        min_species_size = 2,
        stagnation_limit = 15,
    ),
)
```

`ThresholdSpeciation` is the other half of NEAT's innovation
mechanism: each genome belongs to a species defined by compatibility
distance from a representative. Fitness sharing within species
protects structurally novel individuals long enough for the new
topology to refine.

## Solve

```julia
problem = GPProblem(evaluator, GraphGenome; seed=42)
result  = solve(problem, algorithm; verbose=true)
```

A typical run finds fitness `< 0.01` in roughly 30–60 generations.
Across 5 seeds, 4 out of 5 typically converge under this budget
(see the Benchmarks table in the
[project README](https://github.com/CodeReclaimers/Arborist.jl#benchmarks)).

## Inspect the topology

```julia
g = result.best_genome
println("Nodes:       ", length(g.nodes))
println("Connections: ", count(c -> c.enabled, values(g.connections)))
println("Best fit:    ", result.best_fitness)
```

A representative XOR champion has 5–7 nodes (2 inputs + 1 output +
1–4 hidden) and 6–10 enabled connections.

## Next steps

- Harder control problems: [cart-pole tutorial](control_cartpole.md).
- Multi-objective NEAT with topology size as the second objective:
  `test/benchmarks/xor_nsga2_neat.jl`.
- Recurrent networks for non-Markovian tasks: set
  `allow_recurrent=true` with `relaxation_passes=N` on `GraphEvaluator`.
