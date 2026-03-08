#!/bin/bash
# ============================================================
# Experiment 1: Safety Circuit Discovery in VLMs
# ============================================================
#
# Usage:
#   bash run_experiment1.sh                    # LLaVA-7B on auto device
#   bash run_experiment1.sh --quick            # 5 entries only
#   bash run_experiment1.sh --blip             # BLIP-VQA-base
#   bash run_experiment1.sh --tinyllava        # TinyLLaVA
#   bash run_experiment1.sh --all              # run all 3 models (sequential)
#   bash run_experiment1.sh --h200             # H200-optimized (bf16 + flash attn + compile)
#   bash run_experiment1.sh --rtx6000          # RTX PRO 6000 Blackwell (bf16 + flash attn)
#   bash run_experiment1.sh --multi-gpu        # all 3 models across 4 GPUs in parallel
#   bash run_experiment1.sh --type image_counterfactual
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODELS=()
NUM_PROMPTS="-1"
DEVICE="auto"
DTYPE="float16"
EXTRA_FLAGS=""
CF_TYPE=""
MULTI_GPU=false

for arg in "$@"; do
    case $arg in
        --quick)       NUM_PROMPTS="5" ;;
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
        --type)        ;; # handled below
        image_counterfactual|text_counterfactual|typographic_attack)
            CF_TYPE="$arg"
            ;;
        --all)
            MODELS=("Salesforce/blip-vqa-base" "bczhou/TinyLLaVA-3.1B" "llava-hf/llava-1.5-7b-hf")
            ;;
    esac
done

# Handle --type <value> pattern
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--type" ]] && [[ $((i+1)) -lt ${#ARGS[@]} ]]; then
        CF_TYPE="${ARGS[$((i+1))]}"
    fi
done

# Default to LLaVA-7B if no model specified
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("llava-hf/llava-1.5-7b-hf")
fi

TYPE_FLAG=""
if [ -n "$CF_TYPE" ]; then
    TYPE_FLAG="--type $CF_TYPE"
fi

run_model() {
    local MODEL="$1"
    local GPU_DEVICE="$2"

    echo ""
    echo "============================================"
    echo "  Model: $MODEL"
    echo "  Device: $GPU_DEVICE | Dtype: $DTYPE | Prompts: $NUM_PROMPTS"
    if [ -n "$CF_TYPE" ]; then
        echo "  Counterfactual type: $CF_TYPE"
    fi
    if [ -n "$EXTRA_FLAGS" ]; then
        echo "  Extra: $EXTRA_FLAGS"
    fi
    echo "============================================"

    echo ""
    echo "--- Part A: Activation Patching ---"
    python activation_patching.py \
        --model "$MODEL" \
        --device "$GPU_DEVICE" \
        --dtype "$DTYPE" \
        --num_prompts "$NUM_PROMPTS" \
        $TYPE_FLAG \
        $EXTRA_FLAGS

    echo ""
    echo "--- Part B: Refusal Direction Analysis ---"
    python refusal_direction.py \
        --model "$MODEL" \
        --device "$GPU_DEVICE" \
        --dtype "$DTYPE" \
        --num_prompts "$NUM_PROMPTS" \
        $EXTRA_FLAGS
}

if [ "$MULTI_GPU" = true ]; then
    echo "============================================"
    echo "  MULTI-GPU MODE: 4x GPUs"
    echo "  GPU 0: BLIP-VQA-base (~0.5GB)"
    echo "  GPU 1: TinyLLaVA-3.1B (~6GB)"
    echo "  GPU 2: LLaVA-1.5-7B image_counterfactual (~14GB)"
    echo "  GPU 3: LLaVA-1.5-7B text_counterfactual (~14GB)"
    echo "============================================"

    # GPU 0: BLIP (tiny, finishes fast)
    (
        echo "[GPU 0] Running BLIP-VQA-base..."
        run_model "Salesforce/blip-vqa-base" "cuda:0"
    ) 2>&1 | while IFS= read -r line; do echo "[GPU0] $line"; done &
    PID_GPU0=$!

    # GPU 1: TinyLLaVA
    (
        echo "[GPU 1] Running TinyLLaVA-3.1B..."
        run_model "bczhou/TinyLLaVA-3.1B" "cuda:1"
    ) 2>&1 | while IFS= read -r line; do echo "[GPU1] $line"; done &
    PID_GPU1=$!

    # GPU 2: LLaVA — image counterfactuals
    (
        echo "[GPU 2] Running LLaVA-7B (image_counterfactual)..."
        python activation_patching.py \
            --model "llava-hf/llava-1.5-7b-hf" \
            --device "cuda:2" \
            --dtype "$DTYPE" \
            --num_prompts "$NUM_PROMPTS" \
            --type image_counterfactual \
            $EXTRA_FLAGS
    ) 2>&1 | while IFS= read -r line; do echo "[GPU2] $line"; done &
    PID_GPU2=$!

    # GPU 3: LLaVA — text counterfactuals + typographic attacks
    (
        echo "[GPU 3] Running LLaVA-7B (text_counterfactual + typographic_attack)..."
        python activation_patching.py \
            --model "llava-hf/llava-1.5-7b-hf" \
            --device "cuda:3" \
            --dtype "$DTYPE" \
            --num_prompts "$NUM_PROMPTS" \
            --type text_counterfactual \
            $EXTRA_FLAGS
        python activation_patching.py \
            --model "llava-hf/llava-1.5-7b-hf" \
            --device "cuda:3" \
            --dtype "$DTYPE" \
            --num_prompts "$NUM_PROMPTS" \
            --type typographic_attack \
            $EXTRA_FLAGS
    ) 2>&1 | while IFS= read -r line; do echo "[GPU3] $line"; done &
    PID_GPU3=$!

    echo "  Waiting for all GPUs to finish..."
    wait $PID_GPU0 && echo "  GPU 0 (BLIP) done." || echo "  GPU 0 (BLIP) failed!"
    wait $PID_GPU1 && echo "  GPU 1 (TinyLLaVA) done." || echo "  GPU 1 (TinyLLaVA) failed!"
    wait $PID_GPU2 && echo "  GPU 2 (LLaVA image) done." || echo "  GPU 2 (LLaVA image) failed!"
    wait $PID_GPU3 && echo "  GPU 3 (LLaVA text+typo) done." || echo "  GPU 3 (LLaVA text+typo) failed!"

    # Run refusal direction on LLaVA after patching is done (needs a free GPU)
    echo ""
    echo "--- Refusal Direction Analysis for LLaVA ---"
    python refusal_direction.py \
        --model "llava-hf/llava-1.5-7b-hf" \
        --device "cuda:0" \
        --dtype "$DTYPE" \
        --num_prompts "$NUM_PROMPTS" \
        $EXTRA_FLAGS
else
    for MODEL in "${MODELS[@]}"; do
        run_model "$MODEL" "$DEVICE"
    done
fi

echo ""
echo "============================================"
echo "  All done! Results:"
echo "============================================"
ls -la results/
