# Iteration 002 — Focus

**Area:** TreeGenome, GraphGenome (NEAT), and AntGenome correctness — the three genome types not covered in iteration 001, plus follow-up on `max_depth` unused field.

**Files under review:**
- `src/tree_genome.jl` — TreeGenome, TreeFitnessEvaluator, random tree generation, mutation/crossover, prefix parser, solve method
- `src/genome/graph_genome.jl` — GraphGenome (NEAT), innovation tracking, mutation operators, NEAT crossover, NEAT distance, network evaluation, topological sort, solve method
- `src/genome/ant_genome.jl` — AntGenome, AntSimulator, random program generation, mutation/crossover, AntEvaluator, solve method
- `src/algorithm.jl` — `max_depth` field (unused follow-up from iteration 001)

**Why this focus:**
1. These three genome types each have their own solve methods, mutation/crossover implementations, and evaluators that are independent of the ExprGenome path reviewed in iteration 001.
2. GraphGenome implements NEAT, a well-specified algorithm with specific requirements (innovation-aligned crossover, speciation-based selection, structural mutation). Deviations from the NEAT specification are the highest-value findings.
3. TreeGenome wraps DynamicExpressions.jl and has its own tree manipulation code (_replace_nth_node!) that could have subtle index-counting bugs.
4. AntGenome uses module-level mutable state and @eval, a combination prone to concurrency and state-leaking issues.
5. Follow-up: `GeneticProgramming.max_depth` is declared and accepted but never read by any code — confirmed dead configuration.
