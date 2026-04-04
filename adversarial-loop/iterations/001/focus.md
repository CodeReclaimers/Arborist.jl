# Iteration 001 — Focus

**Area:** Core evolutionary loop, genetic operators, and selection — the heart of the GP system.

**Files under review:**
- `src/solve.jl` — main solve loop (GeneticProgramming + IslandModel)
- `src/algorithm.jl` — algorithm configuration structs
- `src/operators/mutation.jl` — SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
- `src/operators/crossover.jl` — SubtreeCrossover
- `src/operators/selection.jl` — TournamentSelection
- `src/speciation.jl` — NoSpeciation, ThresholdSpeciation, BehavioralSpeciation, fitness sharing
- `src/genome/evolution.jl` — legacy evolution API (Individual, Population, evolve!)
- `src/genome/codegen.jl` — code generation, GenState, tree manipulation
- `src/genome/expr_genome.jl` — ExprGenome, GPProblem, serialize/deserialize
- `src/evaluators.jl` — TableFitnessEvaluator
- `src/topology.jl` — migration topologies
- `src/migration.jl` — MigrantGenome

**Why this focus:** This is iteration 1, so we start with the core algorithms that everything
else builds on. The evolutionary loop, operator implementations, and selection/speciation
logic are where correctness matters most — bugs here silently corrupt all downstream results.
The boundaries between these components (how operators interact with the solve loop, how
speciation interacts with selection, how the island model delegates to the base algorithm)
are the highest-value targets.

**Not in scope:** TreeGenome, GraphGenome, AntGenome, LLM operator, distributed island model,
sanitizer. These will be covered in subsequent iterations.
