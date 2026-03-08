#!/bin/bash
# ============================================================
# Run All RQ3 Experiments (1 → 2 → 3)
# ============================================================
#
# Usage:
#   bash run_all.sh                       # full pipeline, LLaVA-7B
#   bash run_all.sh --all --h200          # all 3 models, H200-optimized
#   bash run_all.sh --all --rtx6000       # all 3 models, RTX 6000 Ada
#   bash run_all.sh --dual-gpu --rtx6000  # all 3 models, 2x RTX 6000 parallel
#   bash run_all.sh --quick               # quick test run (5 entries)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  RQ3: Safety Implications of Circuit-Based Compression"
echo "  Running full experiment pipeline"
echo "============================================================"

echo ""
echo ">>> Experiment 1: Safety Circuit Discovery"
bash run_experiment1.sh "$@"

echo ""
echo ">>> Experiment 2: Circuit-Aware vs Blind Compression"
bash run_experiment2.sh "$@"

echo ""
echo ">>> Experiment 3: Cross-Model Comparison"
bash run_experiment3.sh

echo ""
echo "============================================================"
echo "  All RQ3 experiments complete!"
echo "============================================================"
