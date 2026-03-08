"""
Targeted Ablation: Prune ONLY safety-critical vs ONLY non-safety components.

Tests the causal claim that identified safety circuits actually matter:
  1. Prune ONLY safety-critical components (projector + layers 14-19) → safety should DROP
  2. Prune ONLY non-safety components (everything else) → safety should be PRESERVED
  3. Prune projector ONLY → test projector's unique importance

Usage:
  CUDA_VISIBLE_DEVICES=3 python targeted_ablation.py \
    --model llava-hf/llava-v1.6-vicuna-13b-hf --device cuda --dtype bfloat16
"""

import argparse
import json
import os
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
    get_prunable_parameters,
    load_safety_rankings,
    classify_components,
    SKIP_ENTRIES,
)
from llm_judge import generate_response, judge_response


def prune_specific_components(model, component_map, target_components, sparsity):
    """Prune only the specified components at the given sparsity."""
    pruned_count = 0
    total_count = 0

    for comp_name, module in component_map.items():
        params = get_prunable_parameters(module)
        for param_name, param in params:
            total_count += param.numel()
            if comp_name not in target_components:
                continue
            with torch.no_grad():
                flat = param.abs().flatten()
                k = int(flat.numel() * sparsity)
                if k == 0:
                    continue
                threshold = torch.kthvalue(flat, k).values
                mask = (param.abs() > threshold).to(param.dtype)
                param.mul_(mask)
                pruned_count += (mask == 0).sum().item()

    return pruned_count, total_count


@torch.inference_mode()
def evaluate_with_judge(model, processor, tokenizer, entries, dataset_dir,
                        device, model_name, client, judge_model, max_entries):
    """Generate + judge."""
    results = []
    total = refused = genuine = failures = 0

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
                    genuine += 1
                else:
                    failures += 1
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
        "genuine_refusal_rate": genuine / max(total, 1),
        "model_failure_rate": failures / max(total, 1),
        "total_evaluated": total,
        "refused": refused,
        "genuine_refusals": genuine,
        "model_failures": failures,
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
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Sparsity level for targeted pruning")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY or pass --api-key")
    client = anthropic.Anthropic(api_key=api_key)

    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    dataset_path = dataset_dir / "dataset.json"
    entries = load_dataset_entries(str(dataset_path))
    harmful_entries = [e for e in entries if e.get("harmful")]

    # Load safety rankings from patching results
    model_tag = args.model.split("/")[-1]
    patching_path = os.path.join(args.results_dir, f"patching_{model_tag}.json")
    rankings = load_safety_rankings(patching_path)

    safety_critical, non_critical = classify_components(rankings, safety_threshold=0.05)
    projector_set = {c for c in safety_critical if "projector" in c}

    print(f"Safety-critical components ({len(safety_critical)}): "
          f"{sorted(safety_critical)[:10]}...")
    print(f"Non-critical components ({len(non_critical)}): "
          f"{sorted(non_critical)[:10]}...")
    print(f"Projector: {projector_set}")
    print(f"Sparsity: {args.sparsity:.0%}")
    print(f"Entries per config: {args.max_entries}")

    # Ablation configs
    configs = [
        ("prune_safety_only", safety_critical,
         "Prune ONLY safety-critical components"),
        ("prune_nonsafety_only", non_critical,
         "Prune ONLY non-safety components"),
        ("prune_projector_only", projector_set,
         "Prune ONLY the projector"),
    ]

    all_results = {}

    for config_name, target_set, description in configs:
        print(f"\n{'='*60}")
        print(f"Ablation: {config_name} — {description}")
        print(f"  Target components: {len(target_set)}")
        print(f"{'='*60}")

        # Reload fresh model
        print("  Loading fresh model...")
        model, processor, tokenizer = load_model(args.model, args.device, args.dtype)
        model.eval()
        component_map = get_component_map(model, args.model)

        # Apply targeted pruning
        pruned, total = prune_specific_components(
            model, component_map, target_set, args.sparsity
        )
        actual_sparsity = pruned / max(total, 1)
        print(f"  Pruned: {pruned/1e6:.1f}M / {total/1e6:.1f}M params "
              f"(effective sparsity: {actual_sparsity:.1%})")

        # Evaluate
        result = evaluate_with_judge(
            model, processor, tokenizer,
            harmful_entries, dataset_dir,
            args.device, args.model, client, args.judge_model,
            args.max_entries,
        )
        result["config"] = config_name
        result["description"] = description
        result["target_components"] = sorted(target_set)
        result["sparsity"] = args.sparsity
        result["actual_sparsity"] = actual_sparsity
        result["pruned_params_M"] = pruned / 1e6
        result["total_params_M"] = total / 1e6

        all_results[config_name] = result

        print(f"  Refusal rate: {result['refusal_rate']:.1%}")
        print(f"    Genuine: {result['genuine_refusal_rate']:.1%}")
        print(f"    Failure: {result['model_failure_rate']:.1%}")

        del model
        torch.cuda.empty_cache()

        # Save incrementally
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir,
                                f"targeted_ablation_{model_tag}.json")
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "results": all_results},
                      f, indent=2, default=str)
        print(f"  Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("TARGETED ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (uncompressed): 84.9% refusal")
    for key, r in all_results.items():
        print(f"  {key}: refusal={r['refusal_rate']:.1%} "
              f"(genuine={r['genuine_refusal_rate']:.1%}, "
              f"failure={r['model_failure_rate']:.1%})")


if __name__ == "__main__":
    main()
