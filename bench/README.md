# Arborist bench/ — benchstone integration

This directory is Arborist's customer-side implementation of the benchstone
benchmark harness contract. The harness ships separately on PyPI as
[`benchstone`](https://pypi.org/project/benchstone/)
([source](https://github.com/CodeReclaimers/benchstone)); this directory only
holds Arborist-side plumbing.

## Contract summary

| File | Role |
|---|---|
| `manifest.toml` | Project + per-benchmark metadata the harness reads |
| `benchmarks.jl` | One `main()` dispatching to per-benchmark entry-point functions |
| `corpus/<name>/` | Frozen corpus directory, hashed in the manifest |

The harness writes an `InvocationConfig` JSON to a temp path, passes it as
`--config=<path>`, and expects the entry point to write a `ProjectResult` JSON
to `--output=<path>`. Stderr is captured as log; stdout is progress-only.

See the [benchstone repository](https://github.com/CodeReclaimers/benchstone)
for the authoritative `InvocationConfig` and `ProjectResult` schemas.

## Benchmarks

### `nsga2_binpack_mean_fitness` (quality, ~18 min/rep)

One repetition = one NSGA-II bin packing run at `seed = config["seed"]`,
evaluated on the held-out test seed (`seed + test_seed_offset`). The metric is
the test-seed fitness of the lowest-primary-fitness Pareto member. Mirrors the
2026-04-18 Karpathy-loop setup (pop_size=200, 200 generations, behavioral init
with pool=10000 via `BP_POOL_SIZE`).

Editable surface for the autoloop: `examples/research/nsga2_tuning_config.jl`
(`build_tuning_algorithm` kwargs only). Do not edit `bench/benchmarks.jl` to
tune — it is wrapping code, not a tuning target.

### `koza_regression_mean_fitness` (quality, ~2 s/rep)

Koza-1 symbolic regression (`y = x^4 + x^3 + x^2 + x`) on `TreeGenome{Float32}`
via `TreeFitnessEvaluator` and `GeneticProgramming`. Selected as the
structurally-different second benchmark: different genome, different algorithm,
single-objective, deterministic bytewise corpus, short runtime.

## Corpus hashing convention

Each corpus directory contains a `corpus_spec.toml` that pins the deterministic
generation parameters. The `corpus_hash` field in `manifest.toml` is
`sha256:<hex>` of that single file. Recompute with:

```bash
sha256sum bench/corpus/<name>/corpus_spec.toml
```

If a corpus grows to multiple files in the future, switch the hash rule to
`sha256` of sorted `sha256sum <file>` output across the directory and document
the change here.

## Stochastic benchmarks and corpus semantics

Items for `nsga2_binpack_mean_fitness` are generated in-process from
`Random.MersenneTwister(rng_seed + episode_index)` at eval time. The
`corpus_spec.toml` pins the generation parameters (n_items, n_episodes,
item_dist, capacity, test_seed_offset, generations), which is sufficient to
make every per-seed item stream byte-deterministic. This is a lighter
interpretation of "bytewise-frozen corpus" than freezing the items themselves
— it was chosen because candidate seeds (derived from meta-seed) are not known
ahead of time, so only the generation spec can be frozen once for all seeds.
Flagged here so it is not forgotten.

## Running locally

Install the harness from PyPI (Python 3.11+):

```bash
pip install benchstone
```

Then, from inside a clone of Arborist.jl:

```bash
# One-off register + exercise run
bench register .
bench run Arborist koza_regression_mean_fitness --seed-set baseline --foreground

# 90-minute baseline
bench baseline establish Arborist nsga2_binpack_mean_fitness --notes "initial baseline on <sha>"

# Evaluate current working tree against baseline
bench evaluate Arborist nsga2_binpack_mean_fitness
```

`bench status` lists any detached jobs. `--allow-dirty` is refused by default
for baselines (provenance check).
