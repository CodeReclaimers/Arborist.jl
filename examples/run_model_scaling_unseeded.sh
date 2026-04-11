#!/bin/bash
# Model scaling experiment WITHOUT Best Fit template seeding.
#
# Runs 4 local models × 5 seeds × 200 generations with fully random
# initialization. This tests whether model scale matters for structural
# discovery (discovering the while-loop Best Fit pattern from scratch)
# rather than parameter tuning within a pre-seeded template.
#
# Models: gemma4:e2b, gemma4:e4b, gemma4:26b, qwen3-coder:30b
# All local (no API cost). Runs sequentially (GPU contention).
#
# Usage:
#   bash examples/run_model_scaling_unseeded.sh              # all models
#   bash examples/run_model_scaling_unseeded.sh qwen3_30b    # single model
set -u

JULIA="julia --project -t auto"
LOG_DIR="examples/logs/model_scaling_unseeded"
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "============================================================"
echo "Model scaling (unseeded) starting at $(date)"
echo "============================================================"

NTHREADS=$($JULIA -e 'println(Threads.nthreads())')
echo "Julia threads: $NTHREADS"

# Write TSV header once
DATA_DIR="examples/data"
mkdir -p "$DATA_DIR"
TSV="$DATA_DIR/model_scaling_unseeded.tsv"
if [ ! -f "$TSV" ]; then
    echo -e "model\tseed\ttrain\ttest\twall_time\tff_test\tbf_test\tllm_calls\tllm_ok\tllm_latency" > "$TSV"
fi

if [ $# -ge 1 ]; then
    VARIANTS="$1"
    echo "Running single model: $VARIANTS"
else
    VARIANTS="gemma4_e2b gemma4_e4b gemma4_26b qwen3_30b"
    echo "Running all models: $VARIANTS"
fi

echo "200 generations, random init (no Best Fit template seeding)"
echo "Expected: ~50 min per model (5 seeds × 200 gen)"
echo

for variant in $VARIANTS; do
    echo "------------------------------------------------------------"
    echo "Starting model: $variant at $(date)"
    echo "------------------------------------------------------------"
    $JULIA examples/bin_packing.jl \
        --experiment=model_scaling_unseeded \
        --variant=$variant \
        > "$LOG_DIR/${variant}.log" 2>&1
    RC=$?
    echo "Model $variant exit=$RC at $(date)"
    if [ $RC -ne 0 ]; then
        echo "ERROR: $variant failed. Check $LOG_DIR/${variant}.log"
        tail -20 "$LOG_DIR/${variant}.log"
    fi
    echo
done

END=$(date +%s)
echo "============================================================"
echo "All models complete. Total wall: $((END - START))s"
echo "Results: $TSV"
echo "Logs: $LOG_DIR/"
echo "============================================================"
