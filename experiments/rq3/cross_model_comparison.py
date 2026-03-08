"""
Experiment 3: Cross-Model Safety Circuit Comparison

Compares safety circuits discovered in Experiment 1 across different VLM
architectures to study whether safety-critical components are architecturally
universal or model-specific.

Comparisons:
  - LLaVA vs TinyLLaVA (same architecture family, different scale)
  - LLaVA vs BLIP-VQA (different architecture families)
  - Per-type comparison (which components matter for text vs image counterfactuals)

Usage:
  python cross_model_comparison.py
  python cross_model_comparison.py --results-dir results/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(results_dir: str, model_tag: str) -> list:
    """Load Experiment 1 patching results for a model (merges all per-type files)."""
    # Try combined file first
    combined_path = os.path.join(results_dir, f"patching_{model_tag}.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            return json.load(f)

    # Otherwise merge all per-type files
    all_data = []
    for suffix in ["_image_counterfactual", "_text_counterfactual",
                    "_typographic_attack", ""]:
        path = os.path.join(results_dir, f"patching_{model_tag}{suffix}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_data.extend(json.load(f))
    return all_data if all_data else None


def compute_rankings(data: list) -> dict:
    """Compute average recovery score per component from patching results."""
    scores = {}
    for exp in data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    return {k: np.mean(v) for k, v in scores.items()}


def compute_rankings_by_type(data: list) -> dict:
    """Compute rankings grouped by counterfactual type."""
    type_scores = {}
    for exp in data:
        cf_type = exp.get("counterfactual_type", "unknown")
        for r in exp["results"]:
            type_scores.setdefault(cf_type, {}).setdefault(
                r["component"], []
            ).append(r["recovery_score"])

    return {
        t: {k: np.mean(v) for k, v in comps.items()}
        for t, comps in type_scores.items()
    }


def normalize_component_name(comp: str) -> tuple:
    """
    Extract (layer_number, component_type) from component name.
    Handles both LLaVA ('layer_5_attn') and BLIP ('enc_layer_3_crossattn').
    Returns (layer_num, type_str) or (None, comp) for non-layer components.
    """
    parts = comp.split("_")
    for i, p in enumerate(parts):
        if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_num = int(parts[i + 1])
            prefix = "_".join(parts[:i]) if i > 0 else ""
            comp_type = "_".join(parts[i + 2:])
            return layer_num, f"{prefix}_{comp_type}".strip("_")
    return None, comp


def compute_layer_type_profile(rankings: dict) -> dict:
    """
    Create a profile: for each component type, its importance at each relative layer position.
    Returns {comp_type: {relative_position: avg_score}}.
    """
    # Find max layer number
    max_layer = 0
    for comp in rankings:
        layer_num, _ = normalize_component_name(comp)
        if layer_num is not None:
            max_layer = max(max_layer, layer_num)

    if max_layer == 0:
        return {}

    profile = {}
    for comp, score in rankings.items():
        layer_num, comp_type = normalize_component_name(comp)
        if layer_num is None:
            continue
        # Normalize to [0, 1] relative position
        rel_pos = round(layer_num / max(max_layer, 1), 2)
        profile.setdefault(comp_type, {})[rel_pos] = score

    return profile


def compute_rank_correlation(rankings_a: dict, rankings_b: dict) -> float:
    """
    Compute Spearman rank correlation between two sets of component rankings.
    Only considers component types that exist in both (normalized by relative position).
    """
    profile_a = compute_layer_type_profile(rankings_a)
    profile_b = compute_layer_type_profile(rankings_b)

    # Find common component types
    common_types = set(profile_a.keys()) & set(profile_b.keys())
    if not common_types:
        return 0.0

    # For each common type, compute correlation of importance across relative positions
    correlations = []
    for comp_type in common_types:
        positions_a = profile_a[comp_type]
        positions_b = profile_b[comp_type]

        # Bin positions to allow comparison across architectures
        bins = np.arange(0, 1.05, 0.1)
        scores_a = []
        scores_b = []
        for low, high in zip(bins[:-1], bins[1:]):
            vals_a = [v for k, v in positions_a.items() if low <= k < high]
            vals_b = [v for k, v in positions_b.items() if low <= k < high]
            if vals_a and vals_b:
                scores_a.append(np.mean(vals_a))
                scores_b.append(np.mean(vals_b))

        if len(scores_a) >= 3:
            from scipy import stats
            corr, _ = stats.spearmanr(scores_a, scores_b)
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def plot_cross_model(model_rankings: dict, output_dir: str):
    """Generate cross-model comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(model_rankings.keys())
    if len(model_names) < 2:
        print("  Need at least 2 models for cross-model comparison.")
        return

    # --- Plot 1: Top components per model (side by side) ---
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        rankings = model_rankings[model_name]
        sorted_comps = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:20]
        names = [c[0] for c in sorted_comps]
        scores = [c[1] for c in sorted_comps]

        colors = ['#e74c3c' if s > 0.1 else '#3498db' if s > 0.01 else '#95a5a6'
                  for s in scores]
        ax.barh(range(len(names)), scores, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Avg Recovery Score")
        ax.set_title(f"{model_name.split('/')[-1]}")
        ax.invert_yaxis()

    plt.suptitle("Top Safety Components by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_model_top_components.png"), dpi=150)
    plt.close()

    # --- Plot 2: Component type importance profile (relative layer position) ---
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    type_colors = {
        "attn": "#3498db",
        "mlp": "#e74c3c",
        "crossattn": "#2ecc71",
        "enc_attn": "#3498db",
        "enc_mlp": "#e74c3c",
        "enc_crossattn": "#2ecc71",
        "dec_attn": "#9b59b6",
        "dec_mlp": "#e67e22",
        "dec_crossattn": "#1abc9c",
    }

    for ax, model_name in zip(axes, model_names):
        profile = compute_layer_type_profile(model_rankings[model_name])
        for comp_type, pos_scores in profile.items():
            positions = sorted(pos_scores.keys())
            scores = [pos_scores[p] for p in positions]
            color = type_colors.get(comp_type, "#95a5a6")
            ax.plot(positions, scores, 'o-', label=comp_type, color=color, markersize=4)

        ax.set_xlabel("Relative Layer Position (0=first, 1=last)")
        ax.set_ylabel("Avg Recovery Score")
        ax.set_title(f"{model_name.split('/')[-1]}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Safety Importance by Layer Position", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_model_layer_profile.png"), dpi=150)
    plt.close()

    # --- Plot 3: Correlation matrix ---
    if len(model_names) >= 2:
        try:
            corr_matrix = np.zeros((n_models, n_models))
            for i, m1 in enumerate(model_names):
                for j, m2 in enumerate(model_names):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr_matrix[i, j] = compute_rank_correlation(
                            model_rankings[m1], model_rankings[m2]
                        )

            fig, ax = plt.subplots(figsize=(8, 6))
            short_names = [m.split("/")[-1] for m in model_names]
            sns.heatmap(
                corr_matrix,
                xticklabels=short_names,
                yticklabels=short_names,
                annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=-1, vmax=1,
                ax=ax,
            )
            ax.set_title("Safety Circuit Correlation Across Models")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cross_model_correlation.png"), dpi=150)
            plt.close()
        except ImportError:
            print("  scipy not available — skipping correlation matrix")

    print(f"  Cross-model plots saved to {output_dir}")


def plot_per_type_comparison(model_type_rankings: dict, output_dir: str):
    """Plot per-counterfactual-type comparison."""
    os.makedirs(output_dir, exist_ok=True)

    for model_name, type_rankings in model_type_rankings.items():
        if len(type_rankings) < 2:
            continue

        model_tag = model_name.split("/")[-1]
        cf_types = list(type_rankings.keys())

        fig, axes = plt.subplots(1, len(cf_types), figsize=(7 * len(cf_types), 6))
        if len(cf_types) == 1:
            axes = [axes]

        for ax, cf_type in zip(axes, cf_types):
            rankings = type_rankings[cf_type]
            sorted_comps = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:15]
            names = [c[0] for c in sorted_comps]
            scores = [c[1] for c in sorted_comps]

            ax.barh(range(len(names)), scores, color='#e74c3c')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("Avg Recovery Score")
            ax.set_title(f"{cf_type}")
            ax.invert_yaxis()

        plt.suptitle(f"Safety Components by Counterfactual Type — {model_tag}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"per_type_{model_tag}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Cross-Model Safety Circuit Comparison"
    )
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    results_dir = args.results_dir
    print("Experiment 3: Cross-Model Safety Circuit Comparison\n")

    # Discover available results
    model_tags = [
        ("llava-1.5-7b-hf", "llava-hf/llava-1.5-7b-hf"),
        ("TinyLLaVA-3.1B", "bczhou/TinyLLaVA-3.1B"),
        ("blip-vqa-base", "Salesforce/blip-vqa-base"),
    ]

    model_rankings = {}
    model_type_rankings = {}

    for tag, full_name in model_tags:
        data = load_results(results_dir, tag)
        if data:
            rankings = compute_rankings(data)
            model_rankings[full_name] = rankings
            model_type_rankings[full_name] = compute_rankings_by_type(data)
            print(f"  Loaded {full_name}: {len(rankings)} components, "
                  f"{len(data)} experiments")

    if not model_rankings:
        print("\n  No Experiment 1 results found. Run activation_patching.py first.")
        return

    print(f"\n  Models available: {len(model_rankings)}")

    # Cross-model comparison
    if len(model_rankings) >= 2:
        print(f"\n{'='*60}")
        print("Cross-Model Comparison")

        model_names = list(model_rankings.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                try:
                    corr = compute_rank_correlation(model_rankings[m1], model_rankings[m2])
                    print(f"  {m1.split('/')[-1]} vs {m2.split('/')[-1]}: "
                          f"correlation = {corr:.3f}")
                except Exception:
                    print(f"  {m1.split('/')[-1]} vs {m2.split('/')[-1]}: "
                          f"correlation = N/A (need scipy)")

        plot_cross_model(model_rankings, results_dir)
    else:
        print("\n  Only 1 model available. Run more models for cross-model comparison.")

    # Per-type comparison
    print(f"\n{'='*60}")
    print("Per-Counterfactual-Type Comparison")

    for model_name, type_rankings in model_type_rankings.items():
        print(f"\n  {model_name.split('/')[-1]}:")
        for cf_type, rankings in type_rankings.items():
            top3 = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{c}: {s:.3f}" for c, s in top3)
            print(f"    {cf_type}: top = [{top_str}]")

    plot_per_type_comparison(model_type_rankings, results_dir)

    # Summary: are safety circuits universal?
    print(f"\n{'='*60}")
    print("Key Findings")
    print("-" * 60)

    for model_name, rankings in model_rankings.items():
        tag = model_name.split("/")[-1]
        top5 = list(rankings.items())[:5]
        types_in_top5 = set()
        for comp, _ in top5:
            _, comp_type = normalize_component_name(comp)
            types_in_top5.add(comp_type)

        print(f"\n  {tag}:")
        print(f"    Top component types: {', '.join(types_in_top5)}")
        print(f"    Top 5: {[c for c, _ in top5]}")

    print(f"\n{'='*60}")
    print("Experiment 3 complete.")


if __name__ == "__main__":
    main()
