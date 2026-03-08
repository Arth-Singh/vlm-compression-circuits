"""
LLM-as-a-Judge for compressed models — parallelizable across GPUs.

Usage (run one method per GPU):
  CUDA_VISIBLE_DEVICES=0 python llm_judge_compression.py --method uniform_magnitude
  CUDA_VISIBLE_DEVICES=1 python llm_judge_compression.py --method wanda
  CUDA_VISIBLE_DEVICES=2 python llm_judge_compression.py --method random
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
    SKIP_ENTRIES,
)
from llm_judge import (
    generate_response,
    judge_response,
)


@torch.inference_mode()
def evaluate_compressed_safety(
    model, processor, tokenizer, entries, dataset_dir,
    device, model_name, client, judge_model, max_entries=150,
):
    """Generate text and judge refusal on a (possibly compressed) model."""
    results = []
    total = 0
    refused = 0
    genuine_refusals = 0
    model_failures = 0

    for entry in tqdm(entries[:max_entries], desc="Judging"):
        if entry.get("id") in SKIP_ENTRIES:
            continue
        try:
            harmful_image, _ = load_entry_images(entry, dataset_dir)
            prompt_text = entry["harmful"]["text"]
            formatted = format_prompt(prompt_text, model_name)

            response = generate_response(
                model, processor, tokenizer, harmful_image,
                formatted, device,
            )

            judgment = judge_response(client, prompt_text, response, judge_model)
            is_refusal = judgment.get("judgment") == "refusal"
            is_coherent = judgment.get("coherence", "coherent") == "coherent"

            if is_refusal:
                refused += 1
                if is_coherent:
                    genuine_refusals += 1
                else:
                    model_failures += 1
            total += 1

            results.append({
                "id": entry.get("id", "?"),
                "prompt": prompt_text,
                "response": response[:300],
                "judgment": judgment,
                "is_refusal": is_refusal,
                "is_coherent": is_coherent,
            })
        except Exception as e:
            print(f"  WARNING: {entry.get('id', '?')}: {e}")

    return {
        "refusal_rate": refused / max(total, 1),
        "genuine_refusal_rate": genuine_refusals / max(total, 1),
        "model_failure_rate": model_failures / max(total, 1),
        "compliance_rate": (total - refused) / max(total, 1),
        "total_evaluated": total,
        "refused": refused,
        "genuine_refusals": genuine_refusals,
        "model_failures": model_failures,
        "complied": total - refused,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-v1.6-vicuna-13b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-entries", type=int, default=150)
    parser.add_argument("--method", type=str, required=True,
                        choices=["uniform_magnitude", "wanda", "random"],
                        help="Compression method to test")
    parser.add_argument("--sparsity", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7])
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY or pass --api-key")
    client = anthropic.Anthropic(api_key=api_key)

    # Load dataset
    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    dataset_path = dataset_dir / "dataset.json"
    entries = load_dataset_entries(str(dataset_path))
    harmful_entries = [e for e in entries if e.get("harmful")]
    print(f"Dataset: {len(harmful_entries)} harmful entries, "
          f"evaluating {args.max_entries} per config")

    all_results = {}
    method_name = args.method

    for sparsity in args.sparsity:
        config_key = f"{method_name}_{sparsity:.0%}"
        print(f"\n{'='*60}")
        print(f"Config: {config_key}")
        print(f"{'='*60}")

        # Reload fresh model
        print("  Loading fresh model...")
        model, processor, tokenizer = load_model(args.model, args.device, args.dtype)
        model.eval()
        component_map = get_component_map(model, args.model)

        # Apply compression
        print(f"  Applying {method_name} at {sparsity:.0%} sparsity...")
        if method_name == "uniform_magnitude":
            pruned, total = apply_magnitude_pruning(
                model, component_map, sparsity
            )
        elif method_name == "random":
            pruned, total = apply_random_pruning(
                model, component_map, sparsity
            )
        elif method_name == "wanda":
            pruned, total = apply_wanda_pruning(
                model, processor, component_map, sparsity,
                harmful_entries, dataset_dir, args.device, args.model,
            )

        actual_sparsity = pruned / max(total, 1)
        print(f"  Actual sparsity: {actual_sparsity:.1%} "
              f"({pruned/1e6:.1f}M / {total/1e6:.1f}M params)")

        # Evaluate with LLM judge
        print("  Evaluating with LLM judge...")
        result = evaluate_compressed_safety(
            model, processor, tokenizer,
            harmful_entries, dataset_dir,
            args.device, args.model, client, args.judge_model,
            args.max_entries,
        )

        result["method"] = method_name
        result["target_sparsity"] = sparsity
        result["actual_sparsity"] = actual_sparsity
        all_results[config_key] = result

        print(f"  Refusal rate: {result['refusal_rate']:.1%} "
              f"({result['refused']}/{result['total_evaluated']})")
        print(f"    Genuine refusals: {result['genuine_refusals']} "
              f"({result['genuine_refusal_rate']:.1%})")
        print(f"    Model failures:   {result['model_failures']} "
              f"({result['model_failure_rate']:.1%})")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Save incrementally
        model_tag = args.model.split("/")[-1]
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(
            args.results_dir,
            f"llm_judge_v2_{method_name}_{model_tag}.json"
        )
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "method": method_name,
                        "results": all_results}, f, indent=2, default=str)
        print(f"  Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {method_name}")
    print(f"{'='*60}")
    for key, r in all_results.items():
        print(f"  {key}: refusal={r['refusal_rate']:.1%} "
              f"(genuine={r['genuine_refusal_rate']:.1%}, "
              f"failure={r['model_failure_rate']:.1%})")


if __name__ == "__main__":
    main()
