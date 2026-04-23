# Arborist.jl — Examples

Runnable end-to-end scripts that exercise the framework on concrete
problems. Each `.jl` file is self-contained; run from the repository
root with `julia --project=. examples/<file>.jl`.

## Where to start

| File | What it does |
|---|---|
| [`nsga2_regression.jl`](nsga2_regression.jl) | Multi-objective symbolic regression. Shortest end-to-end example; good first read. |
| [`feynman_regression.jl`](feynman_regression.jl) | Physics-flavored symbolic regression from the Feynman Lectures dataset. |
| [`lorenz_recovery.jl`](lorenz_recovery.jl) | Recovers the Lorenz attractor dynamics from a trajectory sample. |
| [`bin_packing.jl`](bin_packing.jl) | LLM-augmented heuristic discovery for online bin packing. Includes classical, template-baseline, bimodal, extended-classical, extended-LLM, multi-seed, and behavioral-speciation experiment modes behind `--experiment=` flags. |
| [`sorting.jl`](sorting.jl) | Evolves a sorting program with curriculum learning and comparison-count penalty. |
| [`sorting_distributed.jl`](sorting_distributed.jl) | Same problem as `sorting.jl` but uses `Distributed.jl` workers via the island model. |

## Results synthesis

- [`bin_packing_overnight_results.md`](bin_packing_overnight_results.md)
  — write-up of the canonical overnight experiments covering the
  LLM operator, behavioral speciation, and multi-seed ablations.
- [`bin_packing_combined_results.md`](bin_packing_combined_results.md)
  — cross-experiment comparison of the same canonical runs.

## Research apparatus

The [`research/`](research/) subdirectory holds the paper-supporting
ablation drivers and hyperparameter tuning scripts. They are **not**
recommended as a starting point for new users — they assume a working
familiarity with the framework and a willingness to run hour-to-day
length jobs. See [`research/README.md`](research/README.md).
