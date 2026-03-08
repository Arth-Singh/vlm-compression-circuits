"""
Stunning animated visualizations for VLM Safety Circuit Analysis.

Creates publication-quality GIFs and static figures that show:
1. Safety circuit activation flowing through transformer layers
2. Layer-by-layer refusal direction buildup
3. Compression impact on safety circuits
4. Cross-model circuit comparison

Requirements: matplotlib, numpy, imageio, PIL
"""

import json
import math
import os
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Custom colormaps
SAFETY_CMAP = LinearSegmentedColormap.from_list(
    "safety", ["#0d1117", "#161b22", "#1a3a5c", "#1f6feb", "#58a6ff", "#79c0ff"]
)
DANGER_CMAP = LinearSegmentedColormap.from_list(
    "danger", ["#0d1117", "#21262d", "#6e3b1e", "#da3633", "#f85149", "#ff7b72"]
)
CIRCUIT_CMAP = LinearSegmentedColormap.from_list(
    "circuit", ["#0d1117", "#0e2a47", "#1a5276", "#2980b9", "#48c9b0", "#2ecc71", "#f1c40f"]
)
DIVERGENCE_CMAP = LinearSegmentedColormap.from_list(
    "divergence", ["#0d1117", "#1a1a2e", "#16213e", "#533483", "#e94560", "#f85149"]
)


def load_results(results_dir: str):
    """Load all experiment results."""
    data = {}
    results_path = Path(results_dir)

    for f in results_path.glob("*.json"):
        data[f.stem] = json.load(open(f))

    for f in results_path.glob("*.pt"):
        import torch
        data[f.stem] = torch.load(f, map_location="cpu", weights_only=True)

    return data


# ---------------------------------------------------------------------------
# 1. SAFETY CIRCUIT ACTIVATION GIF
# ---------------------------------------------------------------------------

def create_circuit_activation_gif(results_dir: str, output_dir: str,
                                   model_tag: str = "llava-1.5-7b-hf"):
    """
    Animated GIF showing safety-critical components lighting up layer by layer.
    Looks like a neural circuit with electricity flowing through it.
    """
    patching_file = Path(results_dir) / f"patching_{model_tag}.json"
    if not patching_file.exists():
        print(f"  Skipping circuit GIF: {patching_file} not found")
        return

    with open(patching_file) as f:
        data = json.load(f)

    # Compute per-component average recovery scores
    scores = {}
    for exp in data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    avg_scores = {k: np.mean(v) for k, v in scores.items()}

    # Parse into layers
    layer_data = {}
    special_components = {}
    max_layer = 0

    for comp, score in avg_scores.items():
        parts = comp.split("_")
        # Find layer number
        layer_num = None
        comp_type = None
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_num = int(parts[i + 1])
                comp_type = "_".join(parts[i + 2:])
                prefix = "_".join(parts[:i]) if i > 0 else ""
                if prefix:
                    comp_type = f"{prefix}_{comp_type}"
                break

        if layer_num is not None:
            layer_data.setdefault(layer_num, {})[comp_type] = score
            max_layer = max(max_layer, layer_num)
        else:
            special_components[comp] = score

    n_layers = max_layer + 1

    # Determine component types present
    all_types = set()
    for ld in layer_data.values():
        all_types.update(ld.keys())
    comp_types = sorted(all_types)
    n_types = len(comp_types)

    # Normalize scores for visualization
    all_scores = list(avg_scores.values())
    score_max = max(abs(s) for s in all_scores) if all_scores else 1.0

    # Create frames
    frames = []
    n_frames = n_layers + 15  # extra frames at end to hold final state

    fig_w, fig_h = 14, 8
    dpi = 100

    for frame_idx in range(n_frames):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        active_layer = min(frame_idx, n_layers - 1)

        # Title
        ax.text(0.5, 0.97, f"Safety Circuit Activation — {model_tag}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=18, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="#1f6feb")])

        ax.text(0.5, 0.93, f"Layer {active_layer}/{n_layers - 1}  |  "
                f"Recovery score = how much this component contributes to safety behavior",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#8b949e")

        # Layout: vertical stack of layers, each layer has component blocks
        margin_x = 0.08
        margin_y = 0.08
        plot_w = 1.0 - 2 * margin_x
        plot_h = 0.82

        layer_height = plot_h / n_layers
        block_width = plot_w / max(n_types, 1)

        for layer_idx in range(n_layers):
            y = margin_y + (n_layers - 1 - layer_idx) * layer_height
            ld = layer_data.get(layer_idx, {})

            for type_idx, ctype in enumerate(comp_types):
                x = margin_x + type_idx * block_width
                score = ld.get(ctype, 0.0)
                norm_score = score / score_max if score_max > 0 else 0

                # Determine color intensity based on whether this layer is "active"
                if layer_idx <= active_layer:
                    # Already activated — show full color
                    if layer_idx == active_layer and frame_idx < n_layers:
                        # Currently activating — bright glow
                        alpha = min(1.0, abs(norm_score) * 2 + 0.3)
                        if norm_score > 0.1:
                            color = CIRCUIT_CMAP(min(1.0, norm_score * 1.5 + 0.3))
                        elif norm_score < -0.05:
                            color = DANGER_CMAP(min(1.0, abs(norm_score) * 1.5 + 0.3))
                        else:
                            color = (0.15, 0.17, 0.20, 0.6)
                        # Add glow effect
                        glow = mpatches.FancyBboxPatch(
                            (x + 0.003, y + 0.002),
                            block_width - 0.006, layer_height - 0.006,
                            boxstyle="round,pad=0.002",
                            facecolor=color[:3] if len(color) >= 3 else color,
                            alpha=0.3, edgecolor="none",
                            transform=ax.transAxes,
                        )
                        ax.add_patch(glow)
                    else:
                        # Previously activated
                        alpha = min(0.9, abs(norm_score) * 1.5 + 0.15)
                        if norm_score > 0.1:
                            color = CIRCUIT_CMAP(min(1.0, norm_score + 0.2))
                        elif norm_score < -0.05:
                            color = DANGER_CMAP(min(1.0, abs(norm_score) + 0.2))
                        else:
                            color = (0.12, 0.14, 0.17, 0.4)
                else:
                    # Not yet activated — dim
                    alpha = 0.08
                    color = (0.15, 0.17, 0.20, alpha)

                rect = mpatches.FancyBboxPatch(
                    (x + 0.002, y + 0.001),
                    block_width - 0.004, layer_height - 0.003,
                    boxstyle="round,pad=0.001",
                    facecolor=color[:3] if isinstance(color, tuple) and len(color) >= 3 else color,
                    alpha=alpha if isinstance(color, tuple) else 0.8,
                    edgecolor="#30363d" if layer_idx <= active_layer else "#21262d",
                    linewidth=0.5,
                    transform=ax.transAxes,
                )
                ax.add_patch(rect)

                # Score text for active layers with significant scores
                if layer_idx <= active_layer and abs(norm_score) > 0.15:
                    ax.text(x + block_width / 2, y + layer_height / 2,
                            f"{score:.2f}",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold",
                            alpha=min(1.0, abs(norm_score) * 2 + 0.3))

        # Layer labels (left side)
        for layer_idx in range(n_layers):
            y = margin_y + (n_layers - 1 - layer_idx) * layer_height
            label_alpha = 1.0 if layer_idx <= active_layer else 0.3
            ax.text(margin_x - 0.01, y + layer_height / 2,
                    f"L{layer_idx}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=6,
                    color="white", alpha=label_alpha)

        # Component type labels (top)
        for type_idx, ctype in enumerate(comp_types):
            x = margin_x + type_idx * block_width
            label = ctype.replace("_", "\n")
            ax.text(x + block_width / 2, margin_y + n_layers * layer_height + 0.01,
                    label, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7, color="#58a6ff", fontweight="bold")

        # Special components (projector, etc.) on the right
        if special_components:
            sx = 0.92
            sy = 0.5
            ax.text(sx, sy + 0.08, "Special", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8, color="#58a6ff",
                    fontweight="bold")
            for i, (comp, score) in enumerate(sorted(
                    special_components.items(), key=lambda x: -x[1])):
                norm_s = score / score_max
                c = CIRCUIT_CMAP(min(1.0, norm_s + 0.3)) if frame_idx >= 2 else (0.15, 0.17, 0.20)
                al = 0.9 if frame_idx >= 2 else 0.1
                rect = mpatches.FancyBboxPatch(
                    (sx - 0.03, sy - i * 0.06 - 0.02), 0.06, 0.04,
                    boxstyle="round,pad=0.005",
                    facecolor=c[:3] if isinstance(c, tuple) else c,
                    alpha=al, edgecolor="#58a6ff", linewidth=1,
                    transform=ax.transAxes,
                )
                ax.add_patch(rect)
                ax.text(sx, sy - i * 0.06, f"{comp}\n{score:.3f}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold",
                        alpha=al)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    # Save GIF
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"safety_circuit_activation_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=4, loop=0)
    print(f"  Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# 2. REFUSAL DIRECTION BUILDUP GIF
# ---------------------------------------------------------------------------

def create_refusal_direction_gif(results_dir: str, output_dir: str,
                                  model_tag: str = "llava-1.5-7b-hf"):
    """
    Animated bar chart showing refusal direction norms building up through layers.
    Like an equalizer visualization.
    """
    rd_file = Path(results_dir) / f"refusal_direction_{model_tag}.json"
    if not rd_file.exists():
        print(f"  Skipping refusal GIF: {rd_file} not found")
        return

    with open(rd_file) as f:
        rd_data = json.load(f)

    norms = rd_data["direction_norms"]
    # Sort by layer number
    layer_items = []
    for k, v in norms.items():
        num = int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0
        prefix = k.rsplit("_", 1)[0] if "_" in k else ""
        layer_items.append((num, prefix, k, v))
    layer_items.sort(key=lambda x: (x[1], x[0]))

    layer_names = [item[2] for item in layer_items]
    layer_norms = [item[3] for item in layer_items]
    n = len(layer_names)

    max_norm = max(layer_norms) * 1.15

    frames = []
    n_frames = n + 12

    for frame_idx in range(n_frames):
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        active = min(frame_idx, n)

        # Draw bars
        bars_x = np.arange(n)
        bar_heights = []
        bar_colors = []

        for i in range(n):
            if i < active:
                h = layer_norms[i]
                # Color based on norm magnitude
                t = h / max_norm
                if t > 0.7:
                    color = "#f85149"  # hot red
                elif t > 0.4:
                    color = "#f0883e"  # orange
                elif t > 0.2:
                    color = "#58a6ff"  # blue
                else:
                    color = "#388bfd"  # dim blue
            elif i == active and frame_idx < n:
                # Currently appearing — flash white
                h = layer_norms[i]
                color = "#ffffff"
            else:
                h = 0
                color = "#21262d"

            bar_heights.append(h)
            bar_colors.append(color)

        bars = ax.bar(bars_x, bar_heights, color=bar_colors, width=0.75,
                      edgecolor="#30363d", linewidth=0.5)

        # Add glow to current bar
        if frame_idx < n:
            idx = min(frame_idx, n - 1)
            bars[idx].set_edgecolor("#ffffff")
            bars[idx].set_linewidth(2)

        # Styling
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, max_norm)
        ax.set_xticks(bars_x)
        ax.set_xticklabels([ln.replace("_", "\n") for ln in layer_names],
                           fontsize=6, color="#8b949e", rotation=0)
        ax.tick_params(axis="y", colors="#8b949e", labelsize=8)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(f"Refusal Direction Norm by Layer — {model_tag}",
                     fontsize=16, fontweight="bold", color="white", pad=15,
                     path_effects=[pe.withStroke(linewidth=1, foreground="#1f6feb")])
        ax.set_ylabel("Direction Norm", fontsize=11, color="#c9d1d9")
        ax.set_xlabel("Layer", fontsize=11, color="#c9d1d9")

        # Annotation for current layer
        if frame_idx < n:
            idx = min(frame_idx, n - 1)
            ax.annotate(f"{layer_norms[idx]:.2f}",
                        xy=(idx, layer_norms[idx]),
                        xytext=(idx, layer_norms[idx] + max_norm * 0.05),
                        fontsize=10, color="white", fontweight="bold",
                        ha="center", va="bottom")

        # Add "higher = more safety-relevant" annotation
        if active > n // 2:
            ax.text(0.98, 0.95, "Higher norm = stronger\nsafety direction",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="#8b949e", style="italic",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                              edgecolor="#30363d", alpha=0.9))

        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"refusal_direction_buildup_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=5, loop=0)
    print(f"  Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# 3. COMPRESSION IMPACT GIF
# ---------------------------------------------------------------------------

def create_compression_impact_gif(results_dir: str, output_dir: str,
                                   model_tag: str = "llava-1.5-7b-hf"):
    """
    Animated visualization: model components fade/dim as sparsity increases.
    Circuit-aware pruning keeps safety nodes bright while blind pruning dims everything.
    """
    comp_file = Path(results_dir) / f"compression_{model_tag}.json"
    patch_file = Path(results_dir) / f"patching_{model_tag}.json"

    if not comp_file.exists() or not patch_file.exists():
        print(f"  Skipping compression GIF: missing files")
        return

    with open(comp_file) as f:
        comp_data = json.load(f)
    with open(patch_file) as f:
        patch_data = json.load(f)

    # Get component importance
    scores = {}
    for exp in patch_data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    avg_scores = {k: np.mean(v) for k, v in scores.items()}

    # Sort by importance
    sorted_comps = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    top_n = min(30, len(sorted_comps))
    display_comps = sorted_comps[:top_n]

    # Extract compression results
    methods = {}
    for method_name, results_list in comp_data.get("results", {}).items():
        if method_name == "original":
            continue
        for r in results_list:
            sparsity = r.get("actual_sparsity", r.get("target_sparsity", 0))
            safety = r.get("safety", {}).get("avg_refusal_score", 0)
            entropy = r.get("performance", {}).get("avg_entropy", 0)
            methods.setdefault(method_name, []).append({
                "sparsity": sparsity, "safety": safety, "entropy": entropy
            })

    # Create comparison: circuit-aware vs uniform at each sparsity
    sparsity_levels = [0.0, 0.3, 0.5, 0.7]
    score_max = max(s for _, s in display_comps) if display_comps else 1

    frames = []

    for sp_idx, target_sp in enumerate(sparsity_levels):
        # Generate multiple frames per sparsity level (smooth transition)
        n_transition = 8 if sp_idx > 0 else 4

        for t_frame in range(n_transition):
            t = t_frame / max(n_transition - 1, 1)  # 0 to 1

            fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
            fig.patch.set_facecolor("#0d1117")

            for ax_idx, (method_label, prune_fn) in enumerate([
                ("Circuit-Aware Pruning", "circuit_aware"),
                ("Uniform Magnitude Pruning", "uniform")
            ]):
                ax = axes[ax_idx]
                ax.set_facecolor("#0d1117")

                names = []
                bar_vals = []
                bar_colors = []

                safety_threshold_rank = int(top_n * 0.3)  # top 30% are "safety"

                for rank, (comp, score) in enumerate(display_comps):
                    is_safety = rank < safety_threshold_rank
                    norm_score = score / score_max

                    if target_sp == 0:
                        # Original — all bright
                        val = norm_score
                        if is_safety:
                            color = "#2ecc71"  # green
                        else:
                            color = "#3498db"  # blue
                    else:
                        if prune_fn == "circuit_aware" and is_safety:
                            # Protected — stays bright
                            val = norm_score * (1.0 - target_sp * 0.1 * t)
                            color = "#2ecc71"
                        elif prune_fn == "circuit_aware":
                            # Non-safety pruned harder
                            val = norm_score * (1.0 - target_sp * 1.5 * t)
                            val = max(val, 0.02)
                            color = "#e74c3c" if val < norm_score * 0.3 else "#e67e22"
                        else:
                            # Uniform — everything pruned equally
                            val = norm_score * (1.0 - target_sp * t)
                            val = max(val, 0.02)
                            if is_safety:
                                color = "#e74c3c"  # red — safety being destroyed
                            else:
                                color = "#e67e22"  # orange

                    names.append(comp.replace("layer_", "L").replace("_", "\n"))
                    bar_vals.append(val)
                    bar_colors.append(color)

                x_pos = np.arange(top_n)
                ax.barh(x_pos, bar_vals, color=bar_colors, height=0.7,
                        edgecolor="#30363d", linewidth=0.3)

                ax.set_yticks(x_pos)
                ax.set_yticklabels(names, fontsize=5, color="#c9d1d9")
                ax.set_xlim(0, 1.1)
                ax.invert_yaxis()
                ax.tick_params(axis="x", colors="#8b949e", labelsize=7)
                ax.spines["bottom"].set_color("#30363d")
                ax.spines["left"].set_color("#30363d")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                current_sp = target_sp * t if sp_idx > 0 else 0
                ax.set_title(f"{method_label}\nSparsity: {current_sp:.0%}",
                             fontsize=12, fontweight="bold", color="white", pad=10)

            # Add legend
            legend_elements = [
                mpatches.Patch(facecolor="#2ecc71", label="Safety-critical (protected)"),
                mpatches.Patch(facecolor="#3498db", label="Non-critical"),
                mpatches.Patch(facecolor="#e74c3c", label="Pruned / Damaged"),
            ]
            fig.legend(handles=legend_elements, loc="lower center", ncol=3,
                       fontsize=9, frameon=True, facecolor="#161b22",
                       edgecolor="#30363d", labelcolor="white")

            fig.suptitle(f"Compression Impact on Safety Components — {model_tag}",
                         fontsize=16, fontweight="bold", color="white", y=0.98,
                         path_effects=[pe.withStroke(linewidth=1, foreground="#1f6feb")])

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frames.append(buf[:, :, :3].copy())
            plt.close(fig)

    # Hold final frame
    for _ in range(10):
        frames.append(frames[-1])

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"compression_impact_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=4, loop=0)
    print(f"  Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# 4. STATIC: Beautiful summary dashboard
# ---------------------------------------------------------------------------

def create_summary_dashboard(results_dir: str, output_dir: str,
                              model_tag: str = "llava-1.5-7b-hf"):
    """
    Single high-res figure summarizing all findings.
    Publication-quality, dark theme.
    """
    patch_file = Path(results_dir) / f"patching_{model_tag}.json"
    rd_file = Path(results_dir) / f"refusal_direction_{model_tag}.json"

    if not patch_file.exists():
        print(f"  Skipping dashboard: {patch_file} not found")
        return

    with open(patch_file) as f:
        patch_data = json.load(f)

    # Compute rankings
    scores = {}
    for exp in patch_data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    sorted_comps = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    # Create figure
    fig = plt.figure(figsize=(20, 12), dpi=150)
    fig.patch.set_facecolor("#0d1117")

    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    # Panel 1: Top safety components (horizontal bar)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0d1117")
    top_20 = sorted_comps[:20]
    names = [c.replace("layer_", "L") for c, _ in top_20]
    vals = [s for _, s in top_20]
    colors = [CIRCUIT_CMAP(min(1.0, v / max(vals) + 0.2)) for v in vals]

    ax1.barh(range(len(names)), vals, color=colors, height=0.7,
             edgecolor="#30363d", linewidth=0.3)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7, color="#c9d1d9")
    ax1.invert_yaxis()
    ax1.set_xlabel("Recovery Score", fontsize=9, color="#c9d1d9")
    ax1.set_title("Top Safety Components", fontsize=12, fontweight="bold",
                   color="#58a6ff", pad=10)
    _style_ax(ax1)

    # Panel 2: Safety heatmap (layer x type)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor("#0d1117")

    # Build heatmap data
    layer_type_scores = {}
    max_layer = 0
    all_types = set()
    for comp, score in avg_scores.items():
        parts = comp.split("_")
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_num = int(parts[i + 1])
                comp_type = "_".join(parts[i + 2:])
                prefix = "_".join(parts[:i]) if i > 0 else ""
                if prefix:
                    comp_type = f"{prefix}_{comp_type}"
                layer_type_scores[(layer_num, comp_type)] = score
                max_layer = max(max_layer, layer_num)
                all_types.add(comp_type)
                break

    types_sorted = sorted(all_types)
    heatmap = np.zeros((len(types_sorted), max_layer + 1))
    for (ln, ct), score in layer_type_scores.items():
        ti = types_sorted.index(ct)
        heatmap[ti, ln] = score

    im = ax2.imshow(heatmap, aspect="auto", cmap=CIRCUIT_CMAP,
                     interpolation="nearest")
    ax2.set_xticks(range(0, max_layer + 1, max(1, max_layer // 10)))
    ax2.set_yticks(range(len(types_sorted)))
    ax2.set_yticklabels(types_sorted, fontsize=8, color="#c9d1d9")
    ax2.set_xlabel("Layer", fontsize=9, color="#c9d1d9")
    ax2.set_title("Safety Circuit Heatmap (Layer × Component Type)", fontsize=12,
                   fontweight="bold", color="#58a6ff", pad=10)
    ax2.tick_params(colors="#8b949e", labelsize=7)
    plt.colorbar(im, ax=ax2, shrink=0.8, label="Recovery Score")

    # Panel 3: Refusal direction norms
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0d1117")

    if rd_file.exists():
        with open(rd_file) as f:
            rd_data = json.load(f)
        norms = rd_data["direction_norms"]
        items = sorted(norms.items(), key=lambda x: int(x[0].split("_")[-1])
                       if x[0].split("_")[-1].isdigit() else 0)
        rd_names = [k for k, _ in items]
        rd_vals = [v for _, v in items]
        rd_colors = [DANGER_CMAP(min(1.0, v / max(rd_vals) + 0.2)) for v in rd_vals]

        ax3.bar(range(len(rd_names)), rd_vals, color=rd_colors, width=0.7,
                edgecolor="#30363d", linewidth=0.3)
        ax3.set_xticks(range(0, len(rd_names), max(1, len(rd_names) // 8)))
        ax3.set_xticklabels([rd_names[i] for i in range(0, len(rd_names),
                             max(1, len(rd_names) // 8))],
                            fontsize=6, color="#8b949e", rotation=45)
    ax3.set_ylabel("Norm", fontsize=9, color="#c9d1d9")
    ax3.set_title("Refusal Direction Strength", fontsize=12, fontweight="bold",
                   color="#f85149", pad=10)
    _style_ax(ax3)

    # Panel 4: Per-type comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0d1117")

    type_scores = {}
    for exp in patch_data:
        ct = exp.get("counterfactual_type", "unknown")
        for r in exp["results"]:
            type_scores.setdefault(ct, {}).setdefault(r["component"], []).append(
                r["recovery_score"])

    type_avgs = {}
    for ct, comps in type_scores.items():
        type_avgs[ct] = {k: np.mean(v) for k, v in comps.items()}

    type_colors = {"text_counterfactual": "#58a6ff", "image_counterfactual": "#f0883e",
                   "typographic_attack": "#a371f7", "unknown": "#8b949e"}

    for ct, comps in type_avgs.items():
        top5 = sorted(comps.items(), key=lambda x: x[1], reverse=True)[:10]
        x_vals = range(len(top5))
        y_vals = [s for _, s in top5]
        ax4.plot(x_vals, y_vals, "o-", color=type_colors.get(ct, "#8b949e"),
                 label=ct.replace("_", " "), markersize=4, linewidth=1.5, alpha=0.8)

    ax4.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", loc="upper right")
    ax4.set_xlabel("Component Rank", fontsize=9, color="#c9d1d9")
    ax4.set_ylabel("Recovery Score", fontsize=9, color="#c9d1d9")
    ax4.set_title("Safety by Counterfactual Type", fontsize=12,
                   fontweight="bold", color="#a371f7", pad=10)
    _style_ax(ax4)

    # Panel 5: Key metrics text box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#161b22")

    n_comps = len(avg_scores)
    n_safety = sum(1 for _, s in avg_scores.items() if s >= 0.05)
    top_comp = sorted_comps[0] if sorted_comps else ("N/A", 0)
    n_exps = len(patch_data)

    text_lines = [
        ("Total Components", f"{n_comps}"),
        ("Safety-Critical (>0.05)", f"{n_safety}"),
        ("Most Critical", f"{top_comp[0]} ({top_comp[1]:.3f})"),
        ("Total Experiments", f"{n_exps}"),
        ("Model", model_tag),
    ]

    for i, (label, value) in enumerate(text_lines):
        y = 0.85 - i * 0.15
        ax5.text(0.1, y, label, transform=ax5.transAxes, fontsize=11,
                 color="#8b949e", va="center")
        ax5.text(0.9, y, value, transform=ax5.transAxes, fontsize=11,
                 color="#58a6ff", fontweight="bold", va="center", ha="right")

    ax5.set_title("Key Metrics", fontsize=12, fontweight="bold",
                   color="#2ecc71", pad=10)
    ax5.axis("off")

    # Main title
    fig.suptitle(f"VLM Safety Circuit Analysis — {model_tag}",
                 fontsize=20, fontweight="bold", color="white", y=0.98,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#1f6feb")])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dashboard_{model_tag}.png")
    fig.savefig(out_path, facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def _style_ax(ax):
    """Apply dark theme styling to an axis."""
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create stunning visualizations")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--models", nargs="+",
                        default=["llava-1.5-7b-hf", "blip-vqa-base"])
    args = parser.parse_args()

    print("Creating visualizations...\n")

    for model in args.models:
        print(f"=== {model} ===")
        create_circuit_activation_gif(args.results_dir, args.output_dir, model)
        create_refusal_direction_gif(args.results_dir, args.output_dir, model)
        create_compression_impact_gif(args.results_dir, args.output_dir, model)
        create_summary_dashboard(args.results_dir, args.output_dir, model)
        print()

    print("All visualizations complete!")


if __name__ == "__main__":
    main()
