#!/bin/bash
set -e

JULIA="julia --project -t auto"
LOG_DIR="examples/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Overnight bin packing experiments starting at $(date)"
echo "============================================================"
echo ""
NTHREADS=$($JULIA -e 'println(Threads.nthreads())')
echo "Julia threads: $NTHREADS"
echo ""
echo "Expected experiments:"
echo "  G1A: Classical 300 gen          (~650s)"
echo "  G1B: LLM 300 gen                (~5000s)"
echo "  G2A: Classical 5 seeds          (~1100s)"
echo "  G2B: Behavioral 5 seeds         (~1500s)"
echo "  G2C: LLM 5 seeds                (~13000s) [LONGEST]"
echo "  G2D: Template baseline          (~30s)"
echo "  G3A: Classical 800 gen          (~1750s)"
echo ""

# Group 1: Extended LLM run
echo "Starting G1A (classical 300 gen)..."
$JULIA examples/bin_packing.jl --experiment=extended_classical \
    > "$LOG_DIR/g1a.log" 2>&1 &
PID_G1A=$!

echo "Starting G1B (LLM 300 gen)..."
$JULIA examples/bin_packing.jl --experiment=extended_llm \
    > "$LOG_DIR/g1b.log" 2>&1 &
PID_G1B=$!

# Group 2: Multi-seed
echo "Starting G2A (multiseed classical)..."
$JULIA examples/bin_packing.jl --experiment=multiseed_classical \
    > "$LOG_DIR/g2a.log" 2>&1 &
PID_G2A=$!

echo "Starting G2B (multiseed behavioral)..."
$JULIA examples/bin_packing.jl --experiment=multiseed_behavioral \
    > "$LOG_DIR/g2b.log" 2>&1 &
PID_G2B=$!

echo "Starting G2C (multiseed LLM)..."
$JULIA examples/bin_packing.jl --experiment=multiseed_llm \
    > "$LOG_DIR/g2c.log" 2>&1 &
PID_G2C=$!

# G2D is fast, run synchronously
echo "Running G2D (template baseline)..."
$JULIA examples/bin_packing.jl --experiment=template_baseline \
    > "$LOG_DIR/g2d.log" 2>&1
echo "G2D complete."

# Group 3: Time-normalized
echo "Starting G3A (classical 800 gen)..."
$JULIA examples/bin_packing.jl --experiment=timenorm_classical \
    > "$LOG_DIR/g3a.log" 2>&1 &
PID_G3A=$!

echo ""
echo "All experiments launched. PIDs:"
echo "  G1A: $PID_G1A  G1B: $PID_G1B"
echo "  G2A: $PID_G2A  G2B: $PID_G2B  G2C: $PID_G2C"
echo "  G3A: $PID_G3A"
echo ""
echo "Monitor with: tail -f examples/logs/g1b.log"
COMPLETION=$(date -d "+5 hours" 2>/dev/null || date -v+5H 2>/dev/null || echo "~5 hours from now")
echo "Expected completion: $COMPLETION (G2C is longest)"
echo ""

# Wait for all and report
wait $PID_G1A && echo "G1A done ($(date))" || echo "G1A FAILED"
wait $PID_G1B && echo "G1B done ($(date))" || echo "G1B FAILED"
wait $PID_G2A && echo "G2A done ($(date))" || echo "G2A FAILED"
wait $PID_G2B && echo "G2B done ($(date))" || echo "G2B FAILED"
wait $PID_G2C && echo "G2C done ($(date))" || echo "G2C FAILED"
wait $PID_G3A && echo "G3A done ($(date))" || echo "G3A FAILED"

echo ""
echo "All experiments complete: $(date)"
echo "Generating combined results..."
$JULIA examples/bin_packing.jl --experiment=combine_results
echo "Done. See examples/bin_packing_overnight_results.md"
