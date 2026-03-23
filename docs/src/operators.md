# Operators

## Mutation Operators

- **SubtreeMutation**: Replace a random statement with a new random subtree
- **PointMutation**: Modify a single node (variable, literal, or operator)
- **HoistMutation**: Replace a subtree with one of its children (bloat reduction)
- **ExpansionMutation**: Wrap a leaf node in a function call (complexity increase)
- **LLMMutationOperator**: Use an LLM to generate semantically meaningful variations (see [LLM Operator](@ref))

## Crossover Operators

- **SubtreeCrossover**: Swap compatible subtrees between two parents

## Selection Strategies

- **TournamentSelection(k)**: Select the best of `k` random individuals
