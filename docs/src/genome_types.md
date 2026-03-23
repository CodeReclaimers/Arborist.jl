# Genome Types

GenProg.jl provides four genome types for different problem classes.

## Decision Table

| Feature | TreeGenome | ExprGenome | AntGenome | GraphGenome |
|---|---|---|---|---|
| Backend | DynamicExpressions.jl | @eval | @eval | Custom |
| Speed | Fast (vectorized) | Slow (compilation) | Slow (compilation) | Medium |
| Control flow | No | Yes | Yes | N/A |
| Side effects | No | Yes | Yes | N/A |
| Variables | Feature indices | Typed vars | Primitives | Weights |
| Best for | Symbolic regression | Program synthesis | Agent control | Neural topology |
| Extension? | Yes (weakdep) | Core | Core | Core |

## TreeGenome

`TreeGenome{T}` wraps a `DynamicExpressions.Node{T}` expression tree. Evaluation is vectorized over data matrices without any `@eval` compilation, making it 8–10x faster than ExprGenome for symbolic regression. Use this for pure function approximation problems.

Requires: `using DynamicExpressions`

## ExprGenome

`ExprGenome` represents programs as Julia `Expr` ASTs. The program body is a `Vector{Expr}` of typed assignment statements, compiled via `@eval` into callable functions. Supports loops, conditionals, and mutable state. Use this for general program synthesis.

## AntGenome

`AntGenome` represents imperative programs that call side-effectful primitives (move, turn, sense). The program is a single `:block` expression of nested primitive calls and `if`-then-else control flow. Use this for agent control problems like the Santa Fe Ant Trail.

## GraphGenome

`GraphGenome` represents neural network topologies following the NEAT encoding (Stanley & Miikkulainen, 2002). Supports structural mutation (add node, add connection), innovation-number-aligned crossover, and compatibility-distance-based speciation. Use this for neural topology evolution.
