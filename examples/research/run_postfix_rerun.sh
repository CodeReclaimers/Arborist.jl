#!/bin/bash
# Re-run the LLM bin-packing experiments after the deserialize control-flow
# fix and the updated DEFAULT_GP_SYSTEM_PROMPT. Includes one classical control
# (G1A) at the same code revision so the LLM-vs-classical comparison comes
# from a single, internally-consistent build.
#
# This is a stripped-down variant of run_overnight.sh — only G1A, G1B, G2C.
set -u

JULIA="julia --project -t auto"
LOG_DIR="examples/logs/postfix_rerun"
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "============================================================"
echo "Post-parser-fix bin packing rerun starting at $(date)"
echo "============================================================"
NTHREADS=$($JULIA -e 'println(Threads.nthreads())')
echo "Julia threads: $NTHREADS"
echo "Experiments:"
echo "  G1A: extended_classical 300 gen   (~650s, CPU-only)"
echo "  G1B: extended_llm 300 gen         (~5000s, GPU contended w/ G2C)"
echo "  G2C: multiseed_llm 5 seeds        (~13000s, GPU contended w/ G1B)"
echo

echo "Starting G1A (classical control)..."
$JULIA examples/bin_packing.jl --experiment=extended_classical \
    > "$LOG_DIR/g1a.log" 2>&1 &
PID_G1A=$!

echo "Starting G1B (extended_llm)..."
$JULIA examples/bin_packing.jl --experiment=extended_llm \
    > "$LOG_DIR/g1b.log" 2>&1 &
PID_G1B=$!

echo "Starting G2C (multiseed_llm)..."
$JULIA examples/bin_packing.jl --experiment=multiseed_llm \
    > "$LOG_DIR/g2c.log" 2>&1 &
PID_G2C=$!

echo
echo "PIDs: G1A=$PID_G1A G1B=$PID_G1B G2C=$PID_G2C"
echo "Logs: $LOG_DIR/{g1a,g1b,g2c}.log"
echo

wait $PID_G1A; RC_G1A=$?
echo "[$(date)] G1A exit=$RC_G1A"
wait $PID_G1B; RC_G1B=$?
echo "[$(date)] G1B exit=$RC_G1B"
wait $PID_G2C; RC_G2C=$?
echo "[$(date)] G2C exit=$RC_G2C"

END=$(date +%s)
echo
echo "All experiments finished. Total wall: $((END - START))s"
echo "G1A=$RC_G1A G1B=$RC_G1B G2C=$RC_G2C"
exit $((RC_G1A | RC_G1B | RC_G2C))
