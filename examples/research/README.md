# Arborist.jl — Research Apparatus

These scripts are the paper-supporting ablation drivers and
hyperparameter-tuning harnesses. They are **not** beginner-friendly
examples of how to use Arborist.jl — for that, see
[`../README.md`](../README.md) and the top-level `examples/*.jl` files.

The scripts here assume:

- A working copy of the repo at the current HEAD.
- `julia --project=.` can be invoked from the repo root.
- In the LLM-involving variants: either a local Ollama daemon reachable
  at `http://localhost:11434` or a valid `ANTHROPIC_API_KEY` in the
  environment, depending on the experiment.
- Willingness to commit hour-to-day-length wall-clock budget to a single
  experiment. Runs typically produce a `.log` under `examples/logs/`
  (gitignored) and a per-experiment `.md` summary next to the canonical
  results file.

## Files

### NSGA-II tuning harness

- [`nsga2_tuning_config.jl`](nsga2_tuning_config.jl) — the editable
  surface for overnight NSGA-II hyperparameter tuning. Returns an
  `NSGAII` algorithm plus its operator list.
- [`nsga2_tuning_metric.jl`](nsga2_tuning_metric.jl) — the measurement
  harness. Runs three training seeds, evaluates each run's best Pareto
  member on a held-out test seed, and prints the mean test fitness as
  a single scalar on the last line of stdout.

### NSGA-II ablation experiments

- [`run_nsga2_ablations.jl`](run_nsga2_ablations.jl) — driver for the
  2026-04-14 ablation study. Covers `no_llm`, `no_behavioral`,
  `three_objective`, `pop100/pop50`, `llm_only`, `neutral_prompt`, and
  the `_extended` variants. Invoke with `--ablation=<name>` or
  `--ablation=all`.
- [`run_all_ablations.sh`](run_all_ablations.sh) — shell driver that
  runs every ablation in sequence, capturing per-ablation logs and
  continuing past failures.

### Unseeded / instrumented LLM experiments

- [`run_nsga2_unseeded.jl`](run_nsga2_unseeded.jl) — canonical two-objective
  NSGA-II bin-packing run with `qwen3-coder` and `ElitesSection(3)`
  prompt enrichment.
- [`run_instrumented_unseeded.jl`](run_instrumented_unseeded.jl) — same
  shape but with raw LLM I/O debug logging enabled.

### Prompt enrichment ablation

- [`run_ablation_enrichment.sh`](run_ablation_enrichment.sh) — runs four
  enrichment variants (`baseline`, `fitness_only`, `elites_3`, `full`)
  across five seeds each.

### Model scaling experiments

- [`run_model_scaling.sh`](run_model_scaling.sh) — compares local Ollama
  models (`gemma4:e2b`..`qwen3-coder:30b`) and API models
  (`claude-haiku-4-5`, `claude-sonnet-4-6`) with the `elites_3`
  enrichment variant.
- [`run_model_scaling_unseeded.sh`](run_model_scaling_unseeded.sh) —
  same comparison but with fully random initialization (no Best-Fit
  template seeding) to test whether model scale matters for structural
  discovery.

### Overnight orchestration

- [`run_overnight.sh`](run_overnight.sh) — launches the canonical
  overnight experiment set (extended-classical, extended-LLM,
  multiseed-classical, multiseed-behavioral, multiseed-LLM,
  template-baseline, timenorm-classical, combine-results).
- [`run_postfix_rerun.sh`](run_postfix_rerun.sh) — trimmed variant
  of `run_overnight.sh` covering only the G1A / G1B / G2C subset.

## Reproducibility notes

Per-run `.md` output files (e.g. `bin_packing_results_scaling_*.md`)
and the `examples/logs/` and `examples/data/` directories are
gitignored. The canonical synthesis files
(`bin_packing_overnight_results.md`,
`bin_packing_combined_results.md`) are tracked at
`../` and represent the reference record of these experiments.
