# Arborist.jl

Generic, extensible genetic programming framework for Julia. Problem/Algorithm/Solve pattern modeled on SciML/DiffEq.

## Project Status

**Phase 1–5: COMPLETE** (2026-03-22 to 2026-03-23)
**Phase 6 — Public release preparation: COMPLETE** (2026-03-23)

All 1154 tests pass. Fast tier: ~17s. Full benchmarks: ~83s.

## Post-Registration Manual Steps

These steps happen after this Claude Code session:

1. **Registry PR**: Comment `@JuliaRegistrator register` on the latest commit at https://github.com/CodeReclaimers/Arborist.jl (requires JuliaRegistrator GitHub App). Alternatively, follow https://github.com/JuliaRegistries/General#registering-a-new-package
2. **Wait for merge**: New package registry PRs typically take 3 days
3. **Tag release**: `git tag v0.1.0 && git push --tags`
4. **Mint Zenodo DOI**: Connect the GitHub repo to Zenodo and create a release
5. **Update README.md**: Replace the placeholder citation block with the actual DOI
6. **Post Discourse announcement**: Review `docs/discourse_announcement.md` and post to https://discourse.julialang.org/c/package-announcements
7. **Set up Documenter deployment**: Add `DOCUMENTER_KEY` secret to GitHub repo settings, then push to trigger docs build

## Architecture

```
src/
  Arborist.jl              # module, exports, includes
  abstractions.jl         # 8 abstract types
  sanitizer.jl            # ASTSanitizer — function call whitelist
  genome/
    codegen.jl            # Expr-tree code generation
    evolution.jl          # crossover, Individual, Population, evolve!
    expr_genome.jl        # ExprGenome, GPProblem, serialize/deserialize
    ant_genome.jl         # AntGenome — side-effectful programs
    graph_genome.jl       # GraphGenome — NEAT-style neural topology
  operators/              # SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation
  evaluators.jl           # TableFitnessEvaluator
  algorithm.jl            # GeneticProgramming (parallel field), IslandModel
  solve.jl                # solve(), _parallel_evaluate!
  result.jl, defaults.jl, speciation.jl
ext/
  LLMOperatorExt.jl       # LLMMutationOperator (weakdep: HTTP.jl)
  DynExprExt.jl           # TreeGenome, TreeFitnessEvaluator, SymbolicRegressionEvaluator
```

## Key Conventions

- **Explicit RNG everywhere.** No global `rand()`.
- **Deterministic ordering.** All Dict/Set sampling uses `_sorted_*` helpers.
- **`parallel=true`** enables `Threads.@threads` evaluation. Set `false` for reproducibility.
- **Fitness sharing for minimization**: `shared = raw × species_size` (not `raw / size`).
- **Output flushing**: `flush(stdout)` after progress output in long-running loops.
- **Innovation counter**: `reset_innovation_counter!()` before each GraphGenome solve.
- **AST Sanitizer**: opt-in whitelist for @eval security. See `docs/src/security.md`.

## Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                          # fast (~17s)
GENPROG_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'  # full (~83s)
```

## Dependencies

Zero mandatory external dependencies. HTTP.jl and DynamicExpressions.jl are weakdeps.
