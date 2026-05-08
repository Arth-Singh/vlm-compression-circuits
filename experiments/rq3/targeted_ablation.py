"""
Targeted Ablation: Rigorous causal validation of safety circuits.

Tests necessity, sufficiency, and statistical significance:
  1. Necessity: Prune safety-critical components → safety should DROP
  2. Sufficiency: Prune everything EXCEPT safety components → safety should PERSIST
  3. Projector isolation: Prune projector only → isolates projector's role
  4. Random baselines (×N): Prune random component subsets of same size → null distribution

Statistical controls following Shi et al. (NeurIPS 2024):
  - Cohen's d effect size comparing safety ablation vs random distribution
  - One-sided t-test p-value
  - Reports mean ± std of random baselines

Usage:
  CUDA_VISIBLE_DEVICES=3 python targeted_ablation.py \
    --model llava-hf/llava-1.5-7b-hf --device cuda --dtype bfloat16

  # With more random baselines for stronger statistics
  python targeted_ablation.py --n-random 20
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats

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
from llm_judge import (
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    _aggregate_results,
    _generate_all_responses,
    judge_batch,
    make_async_client,
)


def prune_specific_components(model, component_map, target_components, sparsity):
    """Prune only the specified components at the given sparsity using magnitude pruning."""
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


def _snapshot_prunable_weights(model, component_map):
    snap = {}
    for comp_name, module in component_map.items():
        for param_name, param in get_prunable_parameters(module):
            key = f"{comp_name}::{param_name}"
            snap[key] = (param, param.detach().clone())
    return snap


def _restore_prunable_weights(snap):
    with torch.no_grad():
        for _, (param, original) in snap.items():
            param.copy_(original)


def run_single_ablation(args, config_name, target_set, description,
                        harmful_entries, dataset_dir, async_client,
                        model, processor, tokenizer, component_map, snapshot):
    """Restore from snapshot, prune target subset, generate + batch-judge."""
    print(f"\n{'='*60}")
    print(f"Ablation: {config_name} — {description}")
    print(f"  Target components: {len(target_set)} — {sorted(target_set)[:5]}...")
    print(f"{'='*60}")

    _restore_prunable_weights(snapshot)
    pruned, total = prune_specific_components(
        model, component_map, target_set, args.sparsity,
    )
    actual_sparsity = pruned / max(total, 1)
    print(f"  Pruned: {pruned/1e6:.1f}M / {total/1e6:.1f}M params "
          f"(effective sparsity: {actual_sparsity:.1%})")

    generated = _generate_all_responses(
        model, processor, tokenizer, harmful_entries, dataset_dir,
        args.device, args.model, args.max_entries,
    )
    items = [(g["image_data_url"], g["prompt"], g["response"]) for g in generated]
    judgments = judge_batch(async_client, items, args.judge_model, args.concurrency)
    result = _aggregate_results(generated, judgments, response_truncate=300)

    result["config"] = config_name
    result["description"] = description
    result["target_components"] = sorted(target_set)
    result["n_target_components"] = len(target_set)
    result["sparsity"] = args.sparsity
    result["actual_sparsity"] = actual_sparsity
    result["pruned_params_M"] = pruned / 1e6
    result["total_params_M"] = total / 1e6

    print(f"  Refusal: {result['refusal_rate']:.1%} "
          f"(genuine={result['genuine_refusal_rate']:.1%}, "
          f"failure={result['model_failure_rate']:.1%})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-entries", type=int, default=150)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n-random", type=int, default=10,
                        help="Number of random baseline ablations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY,
                        help="Async judge concurrency (default 100)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")
    async_client = make_async_client(api_key)

    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    entries = load_dataset_entries(str(dataset_dir / "dataset.json"))
    harmful_entries = [e for e in entries if e.get("harmful")]

    model_tag = args.model.split("/")[-1]
    patching_path = os.path.join(args.results_dir, f"patching_{model_tag}.json")
    rankings = load_safety_rankings(patching_path)

    safety_critical, non_critical = classify_components(rankings, safety_threshold=0.05)
    projector_set = {c for c in safety_critical if "projector" in c}
    all_components = set(rankings.keys())

    print(f"Safety-critical ({len(safety_critical)}): {sorted(safety_critical)[:8]}...")
    print(f"Non-critical ({len(non_critical)}): {sorted(non_critical)[:8]}...")
    print(f"Sparsity: {args.sparsity:.0%} | Random baselines: {args.n_random}")

    print("Loading model once and snapshotting prunable weights...")
    model, tokenizer, processor = load_model(args.model, args.device, args.dtype)
    model.eval()
    component_map = get_component_map(model, args.model)
    snapshot = _snapshot_prunable_weights(model, component_map)

    all_results = {}
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"targeted_ablation_v2_{model_tag}.json")

    def save_incremental():
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "results": all_results,
                        "statistics": {}}, f, indent=2, default=str)

    # --- Core ablations ---
    configs = [
        ("prune_safety_only", safety_critical,
         "Necessity: prune ONLY safety-critical components"),
        ("prune_nonsafety_only", non_critical,
         "Sufficiency: prune ONLY non-safety components"),
        ("prune_projector_only", projector_set,
         "Isolate projector role"),
    ]

    for config_name, target_set, description in configs:
        result = run_single_ablation(
            args, config_name, target_set, description,
            harmful_entries, dataset_dir, async_client,
            model, processor, tokenizer, component_map, snapshot,
        )
        all_results[config_name] = result
        save_incremental()

    # --- Random baselines (same size as safety-critical set) ---
    safety_size = len(safety_critical)
    all_comps_list = sorted(all_components)
    random_genuine_rates = []

    for i in range(args.n_random):
        random_subset = set(random.sample(all_comps_list, safety_size))
        config_name = f"random_baseline_{i}"
        description = f"Random subset {i+1}/{args.n_random} ({safety_size} components)"

        result = run_single_ablation(
            args, config_name, random_subset, description,
            harmful_entries, dataset_dir, async_client,
            model, processor, tokenizer, component_map, snapshot,
        )
        all_results[config_name] = result
        random_genuine_rates.append(result["genuine_refusal_rate"])
        save_incremental()

    # --- Statistical analysis ---
    safety_genuine = all_results["prune_safety_only"]["genuine_refusal_rate"]
    random_mean = np.mean(random_genuine_rates)
    random_std = np.std(random_genuine_rates, ddof=1) if len(random_genuine_rates) > 1 else 0.001

    # Cohen's d: (safety - random_mean) / random_std
    cohens_d = (safety_genuine - random_mean) / random_std if random_std > 0 else 0

    # One-sided t-test: is safety ablation significantly worse than random?
    if len(random_genuine_rates) > 1:
        t_stat, p_value = stats.ttest_1samp(random_genuine_rates, safety_genuine)
        # One-sided: safety < random (safety ablation hurts more)
        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    else:
        t_stat, p_one_sided = 0, 1.0

    statistics = {
        "safety_genuine_refusal": safety_genuine,
        "random_baseline_mean": random_mean,
        "random_baseline_std": random_std,
        "random_baseline_values": random_genuine_rates,
        "cohens_d": round(cohens_d, 3),
        "t_statistic": round(t_stat, 3),
        "p_value_one_sided": round(p_one_sided, 6),
        "n_random_baselines": args.n_random,
        "significant_at_05": p_one_sided < 0.05,
    }

    # Final save with statistics
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "sparsity": args.sparsity,
            "safety_threshold": 0.05,
            "results": all_results,
            "statistics": statistics,
        }, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print("TARGETED ABLATION SUMMARY")
    print(f"{'='*60}")
    for key, r in all_results.items():
        if key.startswith("random_baseline"):
            continue
        print(f"  {key}: genuine={r['genuine_refusal_rate']:.1%}, "
              f"failure={r['model_failure_rate']:.1%}")

    print(f"\n  Random baselines ({args.n_random} draws):")
    print(f"    Genuine refusal: {random_mean:.1%} ± {random_std:.1%}")
    print(f"    Individual: {[f'{r:.1%}' for r in random_genuine_rates]}")

    print(f"\n  Statistical comparison (safety ablation vs random):")
    print(f"    Safety genuine refusal: {safety_genuine:.1%}")
    print(f"    Random mean: {random_mean:.1%} ± {random_std:.1%}")
    print(f"    Cohen's d: {cohens_d:.3f}")
    print(f"    p-value (one-sided): {p_one_sided:.6f}")
    print(f"    Significant at α=0.05: {p_one_sided < 0.05}")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
