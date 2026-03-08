#!/bin/bash
# Run remaining experiments after fixes
set -e
cd /data/arth/rq3_experiments

# Activate conda
eval "$(conda shell.bash hook)"
conda activate rq3

echo "=== Remaining experiments: $(date) ==="

# Wait for image_counterfactual to finish (check if PID still running)
echo "Waiting for image_counterfactual patching to finish..."
while kill -0 436827 2>/dev/null; do
    sleep 10
    echo "  Still running... $(date)"
done
echo "Image counterfactual patching done at $(date)"

# Merge LLaVA per-type results into combined file
echo "=== Merging LLaVA per-type results ==="
python -c "
import json, glob
all_data = []
for f in sorted(glob.glob('results/patching_llava-1.5-7b-hf_*.json')):
    with open(f) as fh:
        data = json.load(fh)
        all_data.extend(data)
        print(f'  {f}: {len(data)} experiments')
with open('results/patching_llava-1.5-7b-hf.json', 'w') as fh:
    json.dump(all_data, fh, indent=2)
print(f'Combined: {len(all_data)} experiments total')
"

# Run remaining experiments in parallel on 3 GPUs
echo "=== Starting remaining experiments ==="

# GPU 0: LLaVA compression (with circuit-aware methods now available)
(
    echo "[GPU0] LLaVA compression (circuit-aware)..."
    python compression_experiment.py --model llava-hf/llava-1.5-7b-hf --device cuda:0 --dtype bfloat16 --sparsity 0.3 0.5 0.7
    echo "[GPU0] DONE at $(date)"
) 2>&1 | tee comp_llava_v2.log &
PID0=$!

# GPU 1: TinyLLaVA compression (with skip_entries fix)
(
    echo "[GPU1] TinyLLaVA compression..."
    python compression_experiment.py --model bczhou/TinyLLaVA-3.1B --device cuda:1 --dtype bfloat16 --sparsity 0.3 0.5 0.7
    echo "[GPU1] DONE at $(date)"
) 2>&1 | tee comp_tinyllava_v2.log &
PID1=$!

echo "PIDs: GPU0=$PID0 GPU1=$PID1"
wait $PID0 && echo "GPU 0 done" || echo "GPU 0 FAILED"
wait $PID1 && echo "GPU 1 done" || echo "GPU 1 FAILED"

# Cross-model comparison (Experiment 3)
echo "=== Experiment 3: Cross-Model Comparison ==="
python cross_model_comparison.py --results-dir results/

echo "=== ALL REMAINING DONE at $(date) ==="
ls -la results/
