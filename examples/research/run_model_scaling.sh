#!/bin/bash
# Model scaling experiment: elites_3 enrichment across model families/sizes.
#
# Local models (Ollama) run sequentially to avoid GPU contention.
# API models (Anthropic) run in parallel with local models since they
# use different compute. API models also run sequentially with each other
# to avoid rate-limit issues.
#
# Models:
#   Local:  gemma4:e2b, gemma4:e4b, gemma4:26b, gemma4:31b, qwen3-coder:30b
#   API:    claude-haiku-4-5, claude-sonnet-4-6
#
# Usage:
#   ANTHROPIC_API_KEY=sk-... bash examples/research/run_model_scaling.sh
#   ANTHROPIC_API_KEY=sk-... bash examples/research/run_model_scaling.sh gemma4_e2b  # single model
set -u

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Export it before running."
    exit 1
fi
export ANTHROPIC_API_KEY

JULIA="julia --project -t auto"
LOG_DIR="examples/logs/model_scaling"
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "============================================================"
echo "Model scaling experiment starting at $(date)"
echo "============================================================"

NTHREADS=$($JULIA -e 'println(Threads.nthreads())')
echo "Julia threads: $NTHREADS"

# Write TSV header once
DATA_DIR="examples/data"
mkdir -p "$DATA_DIR"
TSV="$DATA_DIR/model_scaling.tsv"
if [ ! -f "$TSV" ]; then
    echo -e "model\tseed\ttrain\ttest\twall_time\tff_test\tbf_test\tllm_calls\tllm_ok\tllm_latency" > "$TSV"
fi

if [ $# -ge 1 ]; then
    # Single model mode
    echo "Running single model: $1"
    echo
    $JULIA examples/bin_packing.jl \
        --experiment=model_scaling --variant=$1 \
        > "$LOG_DIR/$1.log" 2>&1
    RC=$?
    echo "Model $1 exit=$RC at $(date)"
    [ $RC -ne 0 ] && echo "ERROR: Check $LOG_DIR/$1.log" && tail -20 "$LOG_DIR/$1.log"
else
    echo "Running all models."
    echo "Local models run sequentially, API models in parallel with local."
    echo

    # Start API models in background (they don't use GPU)
    echo "Starting API models in background..."
    for model in haiku_4_5 sonnet_4_6; do
        echo "  Launching $model..."
        $JULIA examples/bin_packing.jl \
            --experiment=model_scaling --variant=$model \
            > "$LOG_DIR/$model.log" 2>&1 &
        eval "PID_$model=$!"
    done
    echo

    # Run local models sequentially (GPU contention)
    for model in gemma4_e2b gemma4_e4b gemma4_26b gemma4_31b qwen3_30b; do
        echo "------------------------------------------------------------"
        echo "Starting local model: $model at $(date)"
        echo "------------------------------------------------------------"
        $JULIA examples/bin_packing.jl \
            --experiment=model_scaling --variant=$model \
            > "$LOG_DIR/$model.log" 2>&1
        RC=$?
        echo "Model $model exit=$RC at $(date)"
        [ $RC -ne 0 ] && echo "ERROR: Check $LOG_DIR/$model.log" && tail -20 "$LOG_DIR/$model.log"
        echo
    done

    # Wait for API models
    echo "Waiting for API models..."
    for model in haiku_4_5 sonnet_4_6; do
        eval "PID=\$PID_$model"
        wait $PID; RC=$?
        echo "  $model exit=$RC at $(date)"
        [ $RC -ne 0 ] && echo "  ERROR: Check $LOG_DIR/$model.log" && tail -20 "$LOG_DIR/$model.log"
    done
fi

END=$(date +%s)
echo
echo "============================================================"
echo "All models complete. Total wall: $((END - START))s"
echo "Results: $TSV"
echo "Logs: $LOG_DIR/"
echo "============================================================"
