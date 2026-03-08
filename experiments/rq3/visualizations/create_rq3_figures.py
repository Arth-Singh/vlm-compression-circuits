"""
Publication-quality figures for RQ3: Safety impact of compression.

Generates:
1. Compression safety bar chart (genuine refusal vs model failure vs compliance)
2. Targeted ablation comparison chart
3. Activation patching top components bar chart
4. Refusal direction buildup line chart (multi-model)
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------- Style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
})

COLORS = {
    "genuine": "#2ecc71",      # green
    "failure": "#e74c3c",      # red
    "compliance": "#3498db",   # blue
    "baseline": "#2c3e50",     # dark
    "wanda": "#9b59b6",        # purple
    "uniform": "#e67e22",      # orange
    "random": "#95a5a6",       # grey
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR = Path(__file__).parent


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Compression Impact on Safety (stacked bar) ──────────────

def fig_compression_safety():
    """Stacked bar chart: genuine refusal / model failure / compliance per config."""
    methods = {
        "uniform_magnitude": load_json(
            RESULTS_DIR / "llm_judge_v2_uniform_magnitude_llava-v1.6-vicuna-13b-hf.json"
        ),
        "wanda": load_json(
            RESULTS_DIR / "llm_judge_v2_wanda_llava-v1.6-vicuna-13b-hf.json"
        ),
        "random": load_json(
            RESULTS_DIR / "llm_judge_v2_random_llava-v1.6-vicuna-13b-hf.json"
        ),
    }

    labels = []
    genuine_vals = []
    failure_vals = []
    compliance_vals = []

    # Baseline
    labels.append("Baseline\n(no pruning)")
    genuine_vals.append(84.9)
    failure_vals.append(0)
    compliance_vals.append(15.1)

    sparsities = [0.3, 0.5, 0.7]
    method_names = [
        ("uniform_magnitude", "Uniform Mag."),
        ("wanda", "Wanda"),
        ("random", "Random"),
    ]

    for sp in sparsities:
        for method_key, method_label in method_names:
            config_key = f"{method_key}_{sp:.0%}"
            r = methods[method_key]["results"].get(config_key, {})
            genuine = r.get("genuine_refusal_rate", 0) * 100
            failure = r.get("model_failure_rate", 0) * 100
            compliance = r.get("compliance_rate", 0) * 100

            labels.append(f"{method_label}\n{sp:.0%}")
            genuine_vals.append(genuine)
            failure_vals.append(failure)
            compliance_vals.append(compliance)

    x = np.arange(len(labels))
    width = 0.7

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_genuine = ax.bar(x, genuine_vals, width, label="Genuine Refusal (coherent)",
                          color=COLORS["genuine"], edgecolor="white", linewidth=0.5)
    bars_failure = ax.bar(x, failure_vals, width, bottom=genuine_vals,
                          label="Model Failure (incoherent)", color=COLORS["failure"],
                          edgecolor="white", linewidth=0.5)
    bars_compliance = ax.bar(x, compliance_vals, width,
                             bottom=[g + f for g, f in zip(genuine_vals, failure_vals)],
                             label="Compliance", color=COLORS["compliance"],
                             edgecolor="white", linewidth=0.5)

    # Annotate genuine refusal % on bars
    for i, (g, f) in enumerate(zip(genuine_vals, failure_vals)):
        if g > 3:
            ax.text(i, g / 2, f"{g:.1f}%", ha="center", va="center",
                    fontweight="bold", fontsize=9, color="white")
        if f > 5:
            ax.text(i, g + f / 2, f"{f:.1f}%", ha="center", va="center",
                    fontsize=8, color="white", alpha=0.8)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Impact of Compression on Safety Behavior (LLaVA-v1.6-Vicuna-13B)",
                 fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)

    # Add vertical separators between sparsity groups
    for sep in [0.5, 3.5, 6.5]:
        ax.axvline(sep, color="#999999", linestyle="--", alpha=0.4, linewidth=0.8)

    # Sparsity group labels
    for sp_i, sp in enumerate(["30%", "50%", "70%"]):
        center = 2 + sp_i * 3
        ax.text(center, 103, f"Sparsity: {sp}", ha="center", fontsize=10,
                fontweight="bold", color="#555555")

    plt.tight_layout()
    out = OUT_DIR / "compression_safety_llava-v1.6-vicuna-13b-hf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Genuine Refusal Degradation Curves ──────────────────────

def fig_genuine_refusal_curves():
    """Line chart showing genuine refusal rate across sparsity levels per method."""
    methods_data = {
        "Uniform Magnitude": load_json(
            RESULTS_DIR / "llm_judge_v2_uniform_magnitude_llava-v1.6-vicuna-13b-hf.json"
        ),
        "Wanda": load_json(
            RESULTS_DIR / "llm_judge_v2_wanda_llava-v1.6-vicuna-13b-hf.json"
        ),
        "Random": load_json(
            RESULTS_DIR / "llm_judge_v2_random_llava-v1.6-vicuna-13b-hf.json"
        ),
    }

    sparsities = [0, 30, 50, 70]
    colors = {"Uniform Magnitude": COLORS["uniform"], "Wanda": COLORS["wanda"],
              "Random": COLORS["random"]}
    markers = {"Uniform Magnitude": "s", "Wanda": "o", "Random": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, data in methods_data.items():
        method_key = data["method"]
        rates = [84.9]  # baseline
        for sp in [0.3, 0.5, 0.7]:
            config_key = f"{method_key}_{sp:.0%}"
            r = data["results"].get(config_key, {})
            rates.append(r.get("genuine_refusal_rate", 0) * 100)

        ax.plot(sparsities, rates, marker=markers[method_name], linewidth=2.5,
                markersize=8, label=method_name, color=colors[method_name])

    ax.axhline(84.9, color=COLORS["baseline"], linestyle="--", alpha=0.5,
               label="Baseline (84.9%)")

    ax.set_xlabel("Sparsity Level (%)")
    ax.set_ylabel("Genuine Refusal Rate (%)")
    ax.set_title("Safety Degradation Under Compression\n(Coherent Refusals Only)",
                 fontweight="bold")
    ax.set_xticks(sparsities)
    ax.set_xticklabels(["0%\n(Baseline)", "30%", "50%", "70%"])
    ax.set_ylim(-5, 100)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / "genuine_refusal_curves_llava-v1.6-vicuna-13b-hf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Targeted Ablation ───────────────────────────────────────

def fig_targeted_ablation():
    """Grouped bar chart comparing targeted ablation configs."""
    data = load_json(
        RESULTS_DIR / "targeted_ablation_llava-v1.6-vicuna-13b-hf.json"
    )

    configs = [
        ("Baseline\n(no pruning)", 84.9, 0.0, 15.1),
    ]
    for key in ["prune_safety_only", "prune_nonsafety_only", "prune_projector_only"]:
        r = data["results"][key]
        label = {
            "prune_safety_only": "Prune Safety\nComponents",
            "prune_nonsafety_only": "Prune Non-Safety\nComponents",
            "prune_projector_only": "Prune Projector\nOnly",
        }[key]
        genuine = r["genuine_refusal_rate"] * 100
        failure = r["model_failure_rate"] * 100
        compliance = (1 - r["refusal_rate"]) * 100
        configs.append((label, genuine, failure, compliance))

    labels = [c[0] for c in configs]
    genuine = [c[1] for c in configs]
    failure = [c[2] for c in configs]
    compliance = [c[3] for c in configs]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(x, genuine, width, label="Genuine Refusal", color=COLORS["genuine"],
           edgecolor="white", linewidth=0.5)
    ax.bar(x, failure, width, bottom=genuine, label="Model Failure",
           color=COLORS["failure"], edgecolor="white", linewidth=0.5)
    ax.bar(x, compliance, width,
           bottom=[g + f for g, f in zip(genuine, failure)],
           label="Compliance", color=COLORS["compliance"],
           edgecolor="white", linewidth=0.5)

    for i, (g, f) in enumerate(zip(genuine, failure)):
        if g > 3:
            ax.text(i, g / 2, f"{g:.1f}%", ha="center", va="center",
                    fontweight="bold", fontsize=10, color="white")
        if f > 5:
            ax.text(i, g + f / 2, f"{f:.1f}%", ha="center", va="center",
                    fontsize=9, color="white", alpha=0.8)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Targeted Ablation: Safety vs. Non-Safety Components (50% Sparsity)",
                 fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / "targeted_ablation_llava-v1.6-vicuna-13b-hf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Top Safety-Critical Components ──────────────────────────

def fig_top_components():
    """Horizontal bar chart of top 15 components by activation patching recovery score."""
    data = load_json(
        RESULTS_DIR / "patching_llava-v1.6-vicuna-13b-hf.json"
    )

    # Compute mean recovery scores across entries
    component_scores = {}
    entries = data if isinstance(data, list) else data.get("entries", data.get("results", []))

    for entry in entries:
        # Each entry has a "results" list of {component, recovery_score, ...}
        results_list = entry.get("results", [])
        if isinstance(results_list, list):
            for r in results_list:
                comp = r.get("component", "")
                score = r.get("recovery_score", 0)
                if comp:
                    if comp not in component_scores:
                        component_scores[comp] = []
                    component_scores[comp].append(score)
        elif isinstance(results_list, dict):
            for comp, score in results_list.items():
                if comp not in component_scores:
                    component_scores[comp] = []
                component_scores[comp].append(score if isinstance(score, (int, float)) else 0)

    mean_scores = {comp: np.mean(vals) for comp, vals in component_scores.items()}
    sorted_comps = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:15]

    names = [c[0] for c in reversed(sorted_comps)]
    scores = [c[1] for c in reversed(sorted_comps)]

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = []
    for name in names:
        if "projector" in name:
            colors.append("#e74c3c")  # red for projector
        elif any(f"layer_{i}_" in name for i in range(14, 20)):
            colors.append("#e67e22")  # orange for safety-critical layers
        else:
            colors.append("#3498db")  # blue for others

    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Mean Recovery Score")
    ax.set_title("Top 15 Safety-Critical Components (Activation Patching)\nLLaVA-v1.6-Vicuna-13B",
                 fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label="Projector"),
        mpatches.Patch(facecolor="#e67e22", label="Safety-critical layers (14-19)"),
        mpatches.Patch(facecolor="#3498db", label="Other layers"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    out = OUT_DIR / "top_safety_components_llava-v1.6-vicuna-13b-hf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Refusal Direction Buildup (multi-model) ─────────────────

def fig_refusal_direction():
    """Line chart of refusal direction norm across layers for all models."""
    models = {}

    for fname in RESULTS_DIR.glob("refusal_direction_*.json"):
        if fname.suffix != ".json":
            continue
        tag = fname.stem.replace("refusal_direction_", "")
        data = load_json(fname)

        if "layer_norms" in data:
            norms = data["layer_norms"]
        elif "direction_norms" in data:
            # Dict like {"layer_0": 1.24, "layer_1": 1.30, ...}
            dn = data["direction_norms"]
            n_layers = len(dn)
            norms = [dn.get(f"layer_{i}", 0) for i in range(n_layers)]
        elif "layers" in data:
            norms = [layer.get("norm", layer.get("refusal_direction_norm", 0))
                     for layer in data["layers"]]
        else:
            continue

        models[tag] = norms

    if not models:
        print("  No refusal direction data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    style_map = {
        "llava-v1.6-vicuna-13b-hf": ("#e74c3c", "-", "LLaVA-v1.6-13B (RLHF)"),
        "llava-1.5-7b-hf": ("#3498db", "--", "LLaVA-1.5-7B"),
        "blip-vqa-base": ("#95a5a6", ":", "BLIP-VQA-Base"),
    }

    for tag, norms in sorted(models.items()):
        color, ls, label = style_map.get(tag, ("#666", "-.", tag))
        layers = list(range(len(norms)))
        ax.plot(layers, norms, color=color, linestyle=ls, linewidth=2.5,
                label=label, marker="o", markersize=3)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Refusal Direction Norm")
    ax.set_title("Refusal Direction Strength Across Layers", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    out = OUT_DIR / "refusal_direction_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: Coherence Matters — Total vs Genuine Refusal ────────────

def fig_coherence_comparison():
    """Side-by-side comparison: total refusal rate vs genuine refusal rate."""
    methods_data = {
        "Uniform\nMagnitude": load_json(
            RESULTS_DIR / "llm_judge_v2_uniform_magnitude_llava-v1.6-vicuna-13b-hf.json"
        ),
        "Wanda": load_json(
            RESULTS_DIR / "llm_judge_v2_wanda_llava-v1.6-vicuna-13b-hf.json"
        ),
        "Random": load_json(
            RESULTS_DIR / "llm_judge_v2_random_llava-v1.6-vicuna-13b-hf.json"
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    sparsities_labels = ["30%", "50%", "70%"]

    for ax, (method_label, data) in zip(axes, methods_data.items()):
        method_key = data["method"]
        total_rates = []
        genuine_rates = []

        for sp in [0.3, 0.5, 0.7]:
            config_key = f"{method_key}_{sp:.0%}"
            r = data["results"].get(config_key, {})
            total_rates.append(r.get("refusal_rate", 0) * 100)
            genuine_rates.append(r.get("genuine_refusal_rate", 0) * 100)

        x = np.arange(3)
        width = 0.35

        ax.bar(x - width/2, total_rates, width, label="Total Refusal",
               color="#e74c3c", alpha=0.7, edgecolor="white")
        ax.bar(x + width/2, genuine_rates, width, label="Genuine Refusal",
               color="#2ecc71", edgecolor="white")

        # Annotate
        for i, (t, g) in enumerate(zip(total_rates, genuine_rates)):
            ax.text(i - width/2, t + 1, f"{t:.0f}%", ha="center", fontsize=8)
            ax.text(i + width/2, g + 1, f"{g:.0f}%", ha="center", fontsize=8,
                    fontweight="bold")

        ax.set_title(method_label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sparsities_labels)
        ax.set_xlabel("Sparsity")
        ax.set_ylim(0, 110)
        if ax == axes[0]:
            ax.set_ylabel("Rate (%)")
            ax.legend(fontsize=9)

    fig.suptitle("Why Coherence Scoring Matters: Total vs Genuine Refusal Rates",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "coherence_comparison_llava-v1.6-vicuna-13b-hf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Creating RQ3 publication figures...")
    fig_compression_safety()
    fig_genuine_refusal_curves()
    fig_targeted_ablation()
    fig_top_components()
    fig_refusal_direction()
    fig_coherence_comparison()
    print("Done!")
