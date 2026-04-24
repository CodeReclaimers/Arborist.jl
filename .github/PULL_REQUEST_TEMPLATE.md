<!-- Thank you for contributing to Arborist.jl! Please fill in the
     sections below. Delete any that don't apply. -->

## Motivation

What is this PR doing and why? Link any issue it closes (e.g.
`closes #42`).

## Changes

- High-level bullet list of what changed.
- User-visible API changes, if any, flagged explicitly.

## Tests

- [ ] Added tests covering the new behavior (unit or integration).
- [ ] Existing tests still pass: `julia --project=. -e 'using Pkg; Pkg.test()'`.
- [ ] Ran the full benchmark tier where relevant:
  `ARBORIST_RUN_BENCHMARKS=true julia --project=. -e 'using Pkg; Pkg.test()'`.

## Documentation

- [ ] Docstrings updated for new / changed exports.
- [ ] `docs/src/` updated if the change is user-visible.
- [ ] `CHANGELOG.md` updated under the next-release heading.

## Breaking changes

- [ ] No breaking changes, OR
- [ ] Breaking changes are documented above and gated behind a
  deprecation shim where practical.

## Reproducibility

For any change touching selection / mutation / crossover / evaluation:

- [ ] Tested with explicit `seed=` to confirm runs are deterministic.
- [ ] Verified behavior with `parallel=true` and `parallel=false` (if
  the change could interact with threading).
