#!/bin/bash
# Driver script that runs all 7 NSGA-II bin-packing ablations sequentially,
# skipping any that fail and continuing with the rest. Each ablation's
# stdout/stderr is captured in its own log file.

set -u  # catch unset vars, but do NOT set -e (we want to continue on failure)

cd "$(dirname "$0")/.."  # project root
mkdir -p examples/logs/ablation

ABLATIONS=(no_llm no_behavioral two_objective pop100 pop50 llm_only neutral_prompt)

DRIVER_LOG="examples/logs/ablation/driver.log"
echo "=== Ablation driver started at $(date -Iseconds) ===" | tee -a "$DRIVER_LOG"

for a in "${ABLATIONS[@]}"; do
    LOG="examples/logs/ablation/${a}.log"
    echo "" | tee -a "$DRIVER_LOG"
    echo "=== [$(date -Iseconds)] Starting ablation: $a ===" | tee -a "$DRIVER_LOG"
    echo "    log: $LOG" | tee -a "$DRIVER_LOG"

    START=$(date +%s)
    julia --project=. -t auto examples/run_nsga2_ablations.jl --ablation="$a" \
        > "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    DURATION=$((END - START))

    if [ $RC -eq 0 ]; then
        echo "=== [$(date -Iseconds)] Ablation $a COMPLETED in ${DURATION}s (rc=0) ===" | tee -a "$DRIVER_LOG"
    else
        echo "=== [$(date -Iseconds)] Ablation $a FAILED after ${DURATION}s (rc=$RC) — continuing ===" | tee -a "$DRIVER_LOG"
        echo "    Last 20 lines of $LOG:" | tee -a "$DRIVER_LOG"
        tail -20 "$LOG" | sed 's/^/      /' | tee -a "$DRIVER_LOG"
    fi
done

echo "" | tee -a "$DRIVER_LOG"
echo "=== All ablations done at $(date -Iseconds) ===" | tee -a "$DRIVER_LOG"
