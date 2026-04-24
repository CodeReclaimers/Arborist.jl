---
name: Bug report
about: Report incorrect or unexpected behavior
title: ''
labels: bug
assignees: ''
---

## Summary

A clear, one-sentence description of the bug.

## Minimal reproducer

A minimal, self-contained Julia snippet that demonstrates the bug. If
randomness is involved, pin the seed with `Random.seed!` or via the
`seed=` kwarg on `GPProblem`:

```julia
using Arborist
# ... minimal code here ...
```

## Expected behavior

What should happen.

## Actual behavior

What actually happens. Include any stack trace, error message, or
incorrect output.

## Environment

- Arborist version: (`Pkg.status("Arborist")` or the git SHA if working
  from a checkout)
- Julia version: (`versioninfo()`)
- OS: (Linux / macOS / Windows + version)
- Thread count: (`Threads.nthreads()`)
- Parallel setting: (`parallel=true` or `parallel=false` in your
  `GeneticProgramming` / `IslandModel` / `NSGAII` construction)

## Additional context

Anything else that might help diagnose — upstream dependency
versions, custom function sets, unusual evaluator shapes, etc.
