#!/bin/bash
# Full experiment run across 4 GPUs
set -e
cd /data/arth/rq3_experiments

# Activate conda in non-interactive shell
eval "$(conda shell.bash hook)"
conda activate rq3

echo "=== Starting full experiment run: $(date) ==="

# GPU 0: LLaVA image_counterfactual
(
    echo "[GPU0] LLaVA image_counterfactual..."
    python activation_patching.py --model llava-hf/llava-1.5-7b-hf --device cuda:0 --dtype bfloat16 --type image_counterfactual
    echo "[GPU0] DONE at $(date)"
) 2>&1 | tee gpu0.log &
PID0=$!

# GPU 1: TinyLLaVA (all types)
(
    echo "[GPU1] TinyLLaVA activation patching..."
    python activation_patching.py --model bczhou/TinyLLaVA-3.1B --device cuda:1 --dtype bfloat16
    echo "[GPU1] TinyLLaVA refusal direction..."
    python refusal_direction.py --model bczhou/TinyLLaVA-3.1B --device cuda:1 --dtype bfloat16
    echo "[GPU1] DONE at $(date)"
) 2>&1 | tee gpu1.log &
PID1=$!

# GPU 2: LLaVA text_counterfactual
(
    echo "[GPU2] LLaVA text_counterfactual..."
    python activation_patching.py --model llava-hf/llava-1.5-7b-hf --device cuda:2 --dtype bfloat16 --type text_counterfactual
    echo "[GPU2] DONE at $(date)"
) 2>&1 | tee gpu2.log &
PID2=$!

# GPU 3: LLaVA typographic_attack
(
    echo "[GPU3] LLaVA typographic_attack..."
    python activation_patching.py --model llava-hf/llava-1.5-7b-hf --device cuda:3 --dtype bfloat16 --type typographic_attack
    echo "[GPU3] DONE at $(date)"
) 2>&1 | tee gpu3.log &
PID3=$!

echo "PIDs: GPU0=$PID0 GPU1=$PID1 GPU2=$PID2 GPU3=$PID3"
wait $PID0 && echo "GPU 0 done" || echo "GPU 0 FAILED"
wait $PID1 && echo "GPU 1 done" || echo "GPU 1 FAILED"
wait $PID2 && echo "GPU 2 done" || echo "GPU 2 FAILED"
wait $PID3 && echo "GPU 3 done" || echo "GPU 3 FAILED"

# LLaVA refusal direction (after patching finishes)
echo "=== LLaVA refusal direction on GPU 0 ==="
python refusal_direction.py --model llava-hf/llava-1.5-7b-hf --device cuda:0 --dtype bfloat16

# Experiment 2: compression (3 models, 3 GPUs)
echo "=== Experiment 2: Compression ==="
(python compression_experiment.py --model Salesforce/blip-vqa-base --device cuda:0 --dtype bfloat16 --sparsity 0.3 0.5 0.7) 2>&1 | tee comp_gpu0.log &
(python compression_experiment.py --model bczhou/TinyLLaVA-3.1B --device cuda:1 --dtype bfloat16 --sparsity 0.3 0.5 0.7) 2>&1 | tee comp_gpu1.log &
(python compression_experiment.py --model llava-hf/llava-1.5-7b-hf --device cuda:2 --dtype bfloat16 --sparsity 0.3 0.5 0.7) 2>&1 | tee comp_gpu2.log &
wait
echo "Experiment 2 done"

# Experiment 3: cross-model comparison
echo "=== Experiment 3: Cross-Model Comparison ==="
python cross_model_comparison.py --results-dir results/

echo "=== ALL DONE at $(date) ==="
ls -la results/
