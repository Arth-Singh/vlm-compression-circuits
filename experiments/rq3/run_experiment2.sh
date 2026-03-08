#!/bin/bash
# ============================================================
# Experiment 2: Circuit-Aware vs Blind Compression
# ============================================================
#
# Requires: Experiment 1 results in results/ directory
#
# Usage:
#   bash run_experiment2.sh                    # LLaVA-7B, default sparsities
#   bash run_experiment2.sh --quick            # 5 entries, single sparsity
#   bash run_experiment2.sh --all              # all 3 models (sequential)
#   bash run_experiment2.sh --h200             # H200-optimized
#   bash run_experiment2.sh --rtx6000          # RTX PRO 6000 Blackwell
#   bash run_experiment2.sh --multi-gpu        # all 3 models across 4 GPUs
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODELS=()
NUM_PROMPTS="-1"
DEVICE="auto"
DTYPE="float16"
EXTRA_FLAGS=""
SPARSITIES="0.3 0.5 0.7"
MULTI_GPU=false

for arg in "$@"; do
    case $arg in
        --quick)
            NUM_PROMPTS="5"
            SPARSITIES="0.5"
            ;;
        --blip)        MODELS+=("Salesforce/blip-vqa-base") ;;
        --tinyllava)   MODELS+=("bczhou/TinyLLaVA-3.1B") ;;
        --llava)       MODELS+=("llava-hf/llava-1.5-7b-hf") ;;
        --cpu)         DEVICE="cpu" ;;
        --h200|--rtx6000)
            DTYPE="bfloat16"
            EXTRA_FLAGS="--flash-attn --compile"
            ;;
        --multi-gpu)
            MULTI_GPU=true
            DTYPE="bfloat16"
            EXTRA_FLAGS="--flash-attn --compile"
            MODELS=("Salesforce/blip-vqa-base" "bczhou/TinyLLaVA-3.1B" "llava-hf/llava-1.5-7b-hf")
            ;;
        --all)
            MODELS=("Salesforce/blip-vqa-base" "bczhou/TinyLLaVA-3.1B" "llava-hf/llava-1.5-7b-hf")
            ;;
    esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("llava-hf/llava-1.5-7b-hf")
fi

run_model() {
    local MODEL="$1"
    local GPU_DEVICE="$2"

    echo ""
    echo "============================================"
    echo "  Experiment 2: Compression Comparison"
    echo "  Model: $MODEL"
    echo "  Device: $GPU_DEVICE | Dtype: $DTYPE"
    echo "  Sparsities: $SPARSITIES"
    if [ -n "$EXTRA_FLAGS" ]; then
        echo "  Extra: $EXTRA_FLAGS"
    fi
    echo "============================================"

    python compression_experiment.py \
        --model "$MODEL" \
        --device "$GPU_DEVICE" \
        --dtype "$DTYPE" \
        --num_prompts "$NUM_PROMPTS" \
        --sparsity $SPARSITIES \
        $EXTRA_FLAGS
}

if [ "$MULTI_GPU" = true ]; then
    echo "============================================"
    echo "  MULTI-GPU MODE: 3 models across 3 GPUs"
    echo "============================================"

    # GPU 0: BLIP
    (
        run_model "Salesforce/blip-vqa-base" "cuda:0"
    ) 2>&1 | while IFS= read -r line; do echo "[GPU0] $line"; done &
    PID_GPU0=$!

    # GPU 1: TinyLLaVA
    (
        run_model "bczhou/TinyLLaVA-3.1B" "cuda:1"
    ) 2>&1 | while IFS= read -r line; do echo "[GPU1] $line"; done &
    PID_GPU1=$!

    # GPU 2: LLaVA-7B
    (
        run_model "llava-hf/llava-1.5-7b-hf" "cuda:2"
    ) 2>&1 | while IFS= read -r line; do echo "[GPU2] $line"; done &
    PID_GPU2=$!

    wait $PID_GPU0 && echo "  GPU 0 (BLIP) done." || echo "  GPU 0 failed!"
    wait $PID_GPU1 && echo "  GPU 1 (TinyLLaVA) done." || echo "  GPU 1 failed!"
    wait $PID_GPU2 && echo "  GPU 2 (LLaVA) done." || echo "  GPU 2 failed!"
else
    for MODEL in "${MODELS[@]}"; do
        run_model "$MODEL" "$DEVICE"
    done
fi

echo ""
echo "============================================"
echo "  Experiment 2 complete! Results:"
echo "============================================"
ls -la results/compression_* 2>/dev/null || echo "  (no compression results yet)"
