# Changelog

All notable changes to GenProg.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-03-23

### Added
- Problem/Algorithm/Solve API following SciML conventions
- ExprGenome: typed AST-based genome with @eval compilation
- TreeGenome: DynamicExpressions.jl-backed genome for fast vectorized
  evaluation (8.4x faster than ExprGenome on Koza suite)
- GraphGenome: NEAT-style neural topology genome with innovation numbers,
  structural mutation, and compatibility distance
- AntGenome: side-effectful genome for agent control problems
- GeneticProgramming algorithm with tournament selection, elitism,
  subtree crossover, subtree/point/hoist/expansion mutation
- IslandModel algorithm with ring-topology migration
- ThresholdSpeciation with stagnation culling and fitness sharing
  (adapted for minimization: shared = raw × species_size)
- LLMMutationOperator via HTTP.jl extension (FunSearch/AlphaEvolve
  pattern) with Anthropic, OpenAI, and Ollama support
- ASTSanitizer for @eval security (function call whitelist)
- SymbolicRegressionEvaluator convenience wrapper
- TreeGenome prefix-notation parser for LLM deserialization
- TableFitnessEvaluator, TreeFitnessEvaluator, GraphEvaluator,
  AntEvaluator
- GPResult with per-generation fitness and mean history
- Explicit RNG threading for reproducible runs (parallel=false)
- Parallel population evaluation via Threads.@threads
- Tiered test execution (GENPROG_RUN_BENCHMARKS=true)
- Benchmarks: Koza-1/2/3, 4-bit boolean parity, XOR NEAT,
  Max Ones, x² symbolic regression, Santa Fe Ant Trail (smoke test)
- GitHub Actions CI with nightly benchmark runs
