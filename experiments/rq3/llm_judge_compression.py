"""
LLM-as-a-Judge for compressed models — parallelizable across GPUs.

Usage (run one method per GPU):
  CUDA_VISIBLE_DEVICES=0 python llm_judge_compression.py --method wanda
  CUDA_VISIBLE_DEVICES=1 python llm_judge_compression.py --method awq_int4

Compression methods:
  - wanda:    weight×activation magnitude pruning (Sun et al., ICLR 2024)
              Multi-level sparsity sweep (default 0.1, 0.2, 0.3, 0.5).
  - awq_int4: AWQ 4-bit weight quantization (Lin et al., MLSys 2024),
              quantized on-the-fly using harmful-prompt text as calibration.
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
import numpy as np
import torch
from tqdm import tqdm

from activation_patching import (
    format_prompt,
    get_component_map,
    load_dataset_entries,
    load_entry_images,
    load_model,
)
from compression_experiment import (
    apply_magnitude_pruning,
    apply_random_pruning,
    apply_wanda_pruning,
    apply_wanda_with_norms,
    compute_wanda_activation_norms,
    get_prunable_parameters,
    SKIP_ENTRIES,
)
from llm_judge import (
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    _aggregate_results,
    _generate_all_responses,
    judge_batch,
    make_async_client,
)


def evaluate_compressed_safety(
    model, processor, tokenizer, entries, dataset_dir,
    device, model_name, async_client, judge_model, max_entries=150,
    concurrency=DEFAULT_JUDGE_CONCURRENCY,
):
    """Generate sequentially on GPU, then async-batch judge with concurrency."""
    generated = _generate_all_responses(
        model, processor, tokenizer, entries, dataset_dir,
        device, model_name, max_entries,
    )
    items = [(g["image_data_url"], g["prompt"], g["response"]) for g in generated]
    judgments = judge_batch(async_client, items, judge_model, concurrency)
    return _aggregate_results(generated, judgments, response_truncate=300)


def _save_per_config(results_dir, config_key, model_tag, model_name,
                     method_name, result_obj):
    out_path = os.path.join(results_dir,
                            f"llm_judge_v3_{config_key}_{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "method": method_name,
                   "results": {config_key: result_obj}},
                  f, indent=2, default=str)
    print(f"  Saved: {out_path}")


def _snapshot_weights(model, component_map):
    """Clone all prunable weights once. Used to restore between sparsity sweeps."""
    snapshot = {}
    for comp_name, module in component_map.items():
        for param_name, param in get_prunable_parameters(module):
            key = f"{comp_name}::{param_name}"
            snapshot[key] = (param, param.detach().clone())
    return snapshot


def _restore_weights(snapshot):
    with torch.no_grad():
        for _, (param, original) in snapshot.items():
            param.copy_(original)


def _run_wanda_sweep(args, harmful_entries, dataset_dir, async_client, model_tag):
    """Load model once, snapshot weights, sweep sparsity reusing calibration norms."""
    print("  Loading model (once for the full sparsity sweep)...")
    model, tokenizer, processor = load_model(args.model, args.device, args.dtype)
    model.eval()
    component_map = get_component_map(model, args.model)

    print(f"  Computing Wanda activation norms over {min(32, len(harmful_entries))} prompts...")
    norms = compute_wanda_activation_norms(
        model, processor, component_map,
        harmful_entries, dataset_dir, args.device, args.model,
    )

    print("  Snapshotting original weights for sparsity sweep...")
    snapshot = _snapshot_weights(model, component_map)

    all_results = {}
    for sparsity in args.sparsity:
        config_key = f"wanda_{sparsity:.0%}"
        print(f"\n{'='*60}\nConfig: {config_key}\n{'='*60}")

        _restore_weights(snapshot)
        pruned, total = apply_wanda_with_norms(
            model, component_map, sparsity, norms,
        )
        actual_sparsity = pruned / max(total, 1)
        print(f"  Effective compression: {actual_sparsity:.1%}")

        print("  Evaluating with LLM judge...")
        result = evaluate_compressed_safety(
            model, processor, tokenizer,
            harmful_entries, dataset_dir,
            args.device, args.model, async_client, args.judge_model,
            args.max_entries, concurrency=args.concurrency,
        )
        result["method"] = "wanda"
        result["config"] = config_key
        result["target_sparsity"] = sparsity
        result["actual_sparsity"] = actual_sparsity
        all_results[config_key] = result

        print(f"  Refusal rate: {result['refusal_rate']:.1%} "
              f"({result['refused']}/{result['total_evaluated']})")
        print(f"    Genuine refusals: {result['genuine_refusals']} "
              f"({result['genuine_refusal_rate']:.1%})")
        print(f"    Model failures:   {result['model_failures']} "
              f"({result['model_failure_rate']:.1%})")

        _save_per_config(args.results_dir, config_key, model_tag,
                         args.model, "wanda", result)

    del model, snapshot
    torch.cuda.empty_cache()
    return all_results


def _run_awq(args, harmful_entries, dataset_dir, async_client, model_tag):
    """AWQ int4: cache quantized checkpoint to disk and reuse across runs."""
    from transformers import AutoProcessor

    cache_dir = Path(args.awq_cache_dir).expanduser() if args.awq_cache_dir \
        else Path(args.results_dir) / "awq_cache" / model_tag
    cache_dir = Path(cache_dir)
    cache_marker = cache_dir / "config.json"

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    from awq import AutoAWQForCausalLM

    if cache_marker.exists():
        print(f"  Loading cached AWQ checkpoint: {cache_dir}")
        awq_model = AutoAWQForCausalLM.from_quantized(
            str(cache_dir),
            safetensors=True,
            device_map="auto",
            fuse_layers=False,
        )
        processor = AutoProcessor.from_pretrained(args.model)
    else:
        print("  AWQ-quantizing on the fly (~5-15 min)...")
        awq_model = AutoAWQForCausalLM.from_pretrained(
            args.model,
            safetensors=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer_for_calib = getattr(processor, "tokenizer", processor)

        calib_data = [
            format_prompt(e["harmful"]["text"], args.model)
            for e in harmful_entries[:128]
            if e.get("id") not in SKIP_ENTRIES
        ]
        awq_model.quantize(tokenizer_for_calib,
                           quant_config=quant_config,
                           calib_data=calib_data)

        cache_dir.mkdir(parents=True, exist_ok=True)
        awq_model.save_quantized(str(cache_dir))
        processor.save_pretrained(str(cache_dir))
        print(f"  Cached AWQ checkpoint: {cache_dir}")

    tokenizer = getattr(processor, "tokenizer", processor)
    model = awq_model.model

    # AutoAWQ only places the quantized language_model on cuda; vision_tower
    # and multi_modal_projector stay on cpu and trip a device-mismatch at
    # generation. Move just the multimodal wrappers (don't .to() the AWQ-
    # quantized layers — those have custom kernels).
    for attr in ("vision_tower", "multi_modal_projector", "vision_model"):
        sub = getattr(model, attr, None) or getattr(getattr(model, "model", model), attr, None)
        if sub is not None:
            sub.to("cuda")
    model.eval()
    actual_sparsity = 0.75  # 4-bit ≈ 75% size reduction vs fp16
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    config_key = "awq_int4"
    print("  Evaluating with LLM judge...")
    result = evaluate_compressed_safety(
        model, processor, tokenizer,
        harmful_entries, dataset_dir,
        args.device, args.model, async_client, args.judge_model,
        args.max_entries, concurrency=args.concurrency,
    )
    result["method"] = "awq_int4"
    result["config"] = config_key
    result["target_sparsity"] = None
    result["actual_sparsity"] = actual_sparsity

    print(f"  Refusal rate: {result['refusal_rate']:.1%} "
          f"({result['refused']}/{result['total_evaluated']})")
    print(f"    Genuine refusals: {result['genuine_refusals']} "
          f"({result['genuine_refusal_rate']:.1%})")
    print(f"    Model failures:   {result['model_failures']} "
          f"({result['model_failure_rate']:.1%})")

    _save_per_config(args.results_dir, config_key, model_tag,
                     args.model, "awq_int4", result)

    del model, awq_model
    torch.cuda.empty_cache()
    return {config_key: result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-entries", type=int, default=150)
    parser.add_argument("--method", type=str, required=True,
                        choices=["wanda", "awq_int4"],
                        help="Compression method to test")
    parser.add_argument("--sparsity", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.5],
                        help="Sparsity levels (for wanda; ignored for awq_int4)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--awq-cache-dir", default=None,
                        help="Directory for AWQ quantized checkpoint cache "
                             "(default: <results-dir>/awq_cache/<model_tag>).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: max-entries=5, single sparsity level (0.5).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY,
                        help="Async judge concurrency (default 100)")
    args = parser.parse_args()

    if args.smoke:
        args.max_entries = 5
        args.sparsity = [0.5]
        print("[smoke mode] max-entries=5, sparsity=[0.5]")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")
    async_client = make_async_client(api_key)

    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    dataset_path = dataset_dir / "dataset.json"
    entries = load_dataset_entries(str(dataset_path))
    harmful_entries = [e for e in entries if e.get("harmful")]
    print(f"Dataset: {len(harmful_entries)} harmful entries, "
          f"evaluating {args.max_entries} per config")

    method_name = args.method
    model_tag = args.model.split("/")[-1]
    os.makedirs(args.results_dir, exist_ok=True)

    if method_name == "awq_int4":
        all_results = _run_awq(args, harmful_entries, dataset_dir, async_client, model_tag)
    else:
        all_results = _run_wanda_sweep(args, harmful_entries, dataset_dir, async_client, model_tag)

    print(f"\n{'='*60}\nSUMMARY — {method_name}\n{'='*60}")
    for key, r in all_results.items():
        print(f"  {key}: refusal={r['refusal_rate']:.1%} "
              f"(genuine={r['genuine_refusal_rate']:.1%}, "
              f"failure={r['model_failure_rate']:.1%})")


if __name__ == "__main__":
    main()
