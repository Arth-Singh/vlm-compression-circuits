#!/bin/bash
# Full evaluation on 4 GPUs in parallel
# GPU 0: Uniform magnitude compression (3 sparsity levels × 150 entries)
# GPU 1: Wanda compression (3 sparsity levels × 150 entries)
# GPU 2: Random compression (3 sparsity levels × 150 entries)
# GPU 3: Targeted ablation (safety vs non-safety vs projector × 150 entries)

MODEL="llava-hf/llava-v1.6-vicuna-13b-hf"
WORK_DIR="/data/arth/rq3_experiments"
cd "$WORK_DIR"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate rq3

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"

echo "=========================================="
echo "Full Evaluation Run — 4 GPUs"
echo "=========================================="
echo "Start: $(date)"

# GPU 0: Uniform magnitude
CUDA_VISIBLE_DEVICES=0 python llm_judge_compression.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --method uniform_magnitude --max-entries 150 \
    > eval_uniform.log 2>&1 &
PID_UNIFORM=$!

# GPU 1: Wanda
CUDA_VISIBLE_DEVICES=1 python llm_judge_compression.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --method wanda --max-entries 150 \
    > eval_wanda.log 2>&1 &
PID_WANDA=$!

# GPU 2: Random
CUDA_VISIBLE_DEVICES=2 python llm_judge_compression.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --method random --max-entries 150 \
    > eval_random.log 2>&1 &
PID_RANDOM=$!

# GPU 3: Targeted ablation
CUDA_VISIBLE_DEVICES=3 python targeted_ablation.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --max-entries 150 --sparsity 0.5 \
    > eval_ablation.log 2>&1 &
PID_ABLATION=$!

echo "  Uniform magnitude (GPU 0): PID $PID_UNIFORM"
echo "  Wanda (GPU 1): PID $PID_WANDA"
echo "  Random (GPU 2): PID $PID_RANDOM"
echo "  Targeted ablation (GPU 3): PID $PID_ABLATION"

echo "Waiting for all jobs..."
wait $PID_UNIFORM
echo "  Uniform done (exit: $?) — $(date)"
wait $PID_WANDA
echo "  Wanda done (exit: $?) — $(date)"
wait $PID_RANDOM
echo "  Random done (exit: $?) — $(date)"
wait $PID_ABLATION
echo "  Ablation done (exit: $?) — $(date)"

echo ""
echo "=========================================="
echo "All evaluations complete: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -la results/llm_judge_v2_*.json results/targeted_ablation_*.json 2>/dev/null
