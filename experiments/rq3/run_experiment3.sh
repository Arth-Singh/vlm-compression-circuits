#!/bin/bash
# ============================================================
# Experiment 3: Cross-Model Safety Circuit Comparison
# ============================================================
#
# Requires: Experiment 1 results for multiple models in results/
#
# Usage:
#   bash run_experiment3.sh
#   bash run_experiment3.sh --results-dir /path/to/results
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="results"

for i in "$@"; do
    case $i in
        --results-dir)
            shift
            RESULTS_DIR="$1"
            shift
            ;;
    esac
done

echo ""
echo "============================================"
echo "  Experiment 3: Cross-Model Comparison"
echo "  Results dir: $RESULTS_DIR"
echo "============================================"

python cross_model_comparison.py --results-dir "$RESULTS_DIR"

echo ""
echo "============================================"
echo "  Experiment 3 complete! Results:"
echo "============================================"
ls -la "$RESULTS_DIR"/cross_model_* "$RESULTS_DIR"/per_type_* 2>/dev/null || echo "  (no plot files found — need results from multiple models)"
