#!/bin/bash
# RQ3 Complete Experiment Suite — 3 GPUs in parallel
#
# Phase 1a (parallel, ~10 min):
#   GPU 0: Activation patching (logit difference metric, ~5 min)
#   GPU 1: Logit lens safety analysis (~3 min)
#   GPU 2: AWQ int4 compression eval (cached checkpoint, ~10 min)
#
# Phase 1b/2 (after patching done, parallel ~135 min):
#   GPU 0: Targeted ablation (3 + 10 random baselines) — needs patching results
#   GPU 1: Wanda compression eval (4 sparsity levels: 10/20/30/50%)

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$WORK_DIR"

# Activate Python env. Override PYENV_ACTIVATE to point at your activate script.
if [ -n "${PYENV_ACTIVATE:-}" ] && [ -f "$PYENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$PYENV_ACTIVATE"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV:-rq3}" || true
fi

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"

echo "=========================================="
echo "RQ3 Complete Experiment Suite v2"
echo "=========================================="
echo "Start: $(date)"
echo "Model: $MODEL"

mkdir -p logs

# ===== PHASE 1a: patching + logit_lens + AWQ in parallel =====
echo ""
echo "===== PHASE 1a: patching + logit_lens + AWQ (3 GPUs) ====="

CUDA_VISIBLE_DEVICES=0 python -u activation_patching.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    > logs/phase1_patching.log 2>&1 &
PID_PATCHING=$!

CUDA_VISIBLE_DEVICES=1 python -u logit_lens.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --max-entries 150 \
    > logs/phase1_logit_lens.log 2>&1 &
PID_LOGIT_LENS=$!

CUDA_VISIBLE_DEVICES=2 python -u llm_judge_compression.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --method awq_int4 --max-entries 150 \
    > logs/phase1_awq.log 2>&1 &
PID_AWQ=$!

echo "  Patching (GPU 0): PID $PID_PATCHING"
echo "  Logit lens (GPU 1): PID $PID_LOGIT_LENS"
echo "  AWQ (GPU 2): PID $PID_AWQ"

# Wait for patching first — ablation needs its output
wait $PID_PATCHING
PATCHING_EXIT=$?
echo "  Patching done (exit: $PATCHING_EXIT) — $(date)"

if [ "$PATCHING_EXIT" -ne 0 ]; then
    echo "ERROR: patching failed; aborting before ablation"
    exit "$PATCHING_EXIT"
fi

# ===== PHASE 1b/2: ablation (GPU 0) + Wanda (GPU 1, after logit_lens) =====
echo ""
echo "===== PHASE 1b/2: ablation + Wanda (parallel) ====="

CUDA_VISIBLE_DEVICES=0 python -u targeted_ablation.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --max-entries 150 --sparsity 0.5 --n-random 10 \
    > logs/phase2_ablation.log 2>&1 &
PID_ABLATION=$!
echo "  Ablation (GPU 0): PID $PID_ABLATION"

# logit_lens is small; wait for it before reusing GPU 1 for wanda
wait $PID_LOGIT_LENS
echo "  Logit lens done (exit: $?) — $(date)"

CUDA_VISIBLE_DEVICES=1 python -u llm_judge_compression.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --method wanda --sparsity 0.1 0.2 0.3 0.5 --max-entries 150 \
    > logs/phase2_wanda.log 2>&1 &
PID_WANDA=$!
echo "  Wanda (GPU 1): PID $PID_WANDA"

wait $PID_AWQ
echo "  AWQ done (exit: $?) — $(date)"
wait $PID_WANDA
echo "  Wanda done (exit: $?) — $(date)"
wait $PID_ABLATION
echo "  Ablation done (exit: $?) — $(date)"

echo ""
echo "=========================================="
echo "All experiments complete: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -la results/patching_*.json results/logit_lens_*.json \
    results/llm_judge_v3_wanda_*.json results/llm_judge_v3_awq_*.json \
    results/targeted_ablation_v2_*.json 2>/dev/null
