#!/bin/bash
# Run all experiments with LLaVA-v1.6-Vicuna-13B (the model that actually refuses)
# Uses 4x RTX PRO 6000 (96GB each)

MODEL="llava-hf/llava-v1.6-vicuna-13b-hf"
WORK_DIR="/data/arth/rq3_experiments"
cd "$WORK_DIR"

eval "$(conda shell.bash hook 2>/dev/null)" || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate rq3

echo "=========================================="
echo "LLaVA-v1.6-Vicuna-13B Full Experiment Run"
echo "=========================================="
echo "Model: $MODEL"
echo "Start: $(date)"

# --- Phase 1: Activation Patching (Experiment 1) ---
# Run 3 counterfactual types in parallel across GPUs 0, 1, 2
echo ""
echo "Phase 1: Activation Patching (3 types in parallel)"

CUDA_VISIBLE_DEVICES=0 python activation_patching.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --type text_counterfactual \
    > v16_patch_text.log 2>&1 &
PID_TEXT=$!

CUDA_VISIBLE_DEVICES=1 python activation_patching.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --type image_counterfactual \
    > v16_patch_image.log 2>&1 &
PID_IMAGE=$!

CUDA_VISIBLE_DEVICES=2 python activation_patching.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --type typographic_attack \
    > v16_patch_typo.log 2>&1 &
PID_TYPO=$!

# Run refusal direction on GPU 3 in parallel
CUDA_VISIBLE_DEVICES=3 python refusal_direction.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    > v16_refusal.log 2>&1 &
PID_REFUSAL=$!

echo "  text_counterfactual (GPU 0): PID $PID_TEXT"
echo "  image_counterfactual (GPU 1): PID $PID_IMAGE"
echo "  typographic_attack (GPU 2): PID $PID_TYPO"
echo "  refusal_direction (GPU 3): PID $PID_REFUSAL"

# Wait for all Phase 1 jobs
echo "  Waiting for Phase 1..."
wait $PID_TEXT
echo "  text_counterfactual done (exit: $?)"
wait $PID_IMAGE
echo "  image_counterfactual done (exit: $?)"
wait $PID_TYPO
echo "  typographic_attack done (exit: $?)"
wait $PID_REFUSAL
echo "  refusal_direction done (exit: $?)"

echo "Phase 1 complete: $(date)"

# --- Merge per-type patching results ---
echo ""
echo "Merging per-type patching results..."
python -c "
import json, glob
files = glob.glob('results/patching_llava-v1.6-vicuna-13b-hf_*.json')
all_data = []
for f in sorted(files):
    with open(f) as fh:
        data = json.load(fh)
        all_data.extend(data)
        print(f'  Loaded {f}: {len(data)} entries')
if all_data:
    out = 'results/patching_llava-v1.6-vicuna-13b-hf.json'
    with open(out, 'w') as fh:
        json.dump(all_data, fh, indent=2)
    print(f'  Merged {len(all_data)} total entries -> {out}')
else:
    print('  WARNING: No patching results found to merge!')
"

# --- Phase 2: Compression Experiment ---
echo ""
echo "Phase 2: Compression Experiment"
echo "  Using GPUs 0+1 (13B model needs ~26GB)"

CUDA_VISIBLE_DEVICES=0 python compression_experiment.py \
    --model "$MODEL" --device cuda --dtype bfloat16 \
    --sparsity 0.3 0.5 0.7 \
    > v16_compression.log 2>&1

echo "Phase 2 complete: $(date)"

# --- Phase 3: Cross-Model Comparison ---
echo ""
echo "Phase 3: Cross-Model Comparison"
python cross_model_comparison.py > v16_crossmodel.log 2>&1
echo "Phase 3 complete: $(date)"

echo ""
echo "=========================================="
echo "All experiments complete: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -la results/patching_llava-v1.6-vicuna-13b-hf*.json 2>/dev/null
ls -la results/compression_llava-v1.6-vicuna-13b-hf*.json 2>/dev/null
ls -la results/refusal_direction_llava-v1.6-vicuna-13b-hf*.json 2>/dev/null
