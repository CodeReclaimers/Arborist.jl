#!/bin/bash
# Prompt enrichment ablation experiment.
#
# Runs 4 variants x 5 seeds = 20 bin-packing experiments sequentially
# (GPU contention makes parallelism counterproductive for LLM experiments).
#
# Variants:
#   baseline     — no enrichment (current behavior)
#   fitness_only — FitnessSection: parent fitness/rank, population best/mean
#   elites_3     — ElitesSection(3): top 3 programs with fitnesses
#   full         — FitnessSection + ElitesSection(3) + GenerationSection
#
# Usage:
#   bash examples/run_ablation_enrichment.sh           # all variants
#   bash examples/run_ablation_enrichment.sh elites_3  # single variant
set -u

JULIA="julia --project -t auto"
LOG_DIR="examples/logs/ablation"
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "============================================================"
echo "Prompt enrichment ablation starting at $(date)"
echo "============================================================"

NTHREADS=$($JULIA -e 'println(Threads.nthreads())')
echo "Julia threads: $NTHREADS"

# Write TSV header once
DATA_DIR="examples/data"
mkdir -p "$DATA_DIR"
TSV="$DATA_DIR/ablation_enrichment.tsv"
if [ ! -f "$TSV" ]; then
    echo -e "variant\tseed\ttrain\ttest\twall_time\tff_test\tbf_test\tllm_calls\tllm_ok\tllm_latency" > "$TSV"
fi

if [ $# -ge 1 ]; then
    VARIANTS="$1"
    echo "Running single variant: $VARIANTS"
else
    VARIANTS="baseline fitness_only elites_3 full"
    echo "Running all variants: $VARIANTS"
fi

echo "Expected: ~25 min per variant (5 seeds x 100 gen)"
echo

for variant in $VARIANTS; do
    echo "------------------------------------------------------------"
    echo "Starting variant: $variant at $(date)"
    echo "------------------------------------------------------------"
    $JULIA examples/bin_packing.jl \
        --experiment=ablation_enrichment \
        --variant=$variant \
        > "$LOG_DIR/${variant}.log" 2>&1
    RC=$?
    echo "Variant $variant exit=$RC at $(date)"
    if [ $RC -ne 0 ]; then
        echo "ERROR: $variant failed. Check $LOG_DIR/${variant}.log"
        tail -20 "$LOG_DIR/${variant}.log"
    fi
    echo
done

END=$(date +%s)
echo "============================================================"
echo "All ablation experiments complete. Total wall: $((END - START))s"
echo "Results: $TSV"
echo "Logs: $LOG_DIR/"
echo "============================================================"
