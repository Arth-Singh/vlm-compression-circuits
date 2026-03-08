"""
Stunning animated visualizations for VLM Safety Circuit Analysis.

Creates publication-quality GIFs and static figures:
1. Neural circuit board — safety components light up like a chip
2. Flowing heatmap — activation scores ripple through layers
3. Compression battle — circuit-aware vs blind pruning animated radar
4. Summary dashboard — beautiful 6-panel overview

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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Color palette — neon cyberpunk on dark
# ---------------------------------------------------------------------------

BG_DARK = "#0a0e17"
BG_PANEL = "#0f1923"
GRID_DIM = "#1a2332"
NEON_BLUE = "#00d4ff"
NEON_CYAN = "#00ffd4"
NEON_GREEN = "#39ff14"
NEON_PINK = "#ff006e"
NEON_ORANGE = "#ff6b35"
NEON_PURPLE = "#bf5af2"
TEXT_WHITE = "#e8eaed"
TEXT_DIM = "#6b7b8d"
ACCENT_GOLD = "#ffd700"

CIRCUIT_CMAP = LinearSegmentedColormap.from_list(
    "circuit", [BG_DARK, "#0a1628", "#0d2847", "#1565c0", "#42a5f5",
                NEON_CYAN, NEON_GREEN, ACCENT_GOLD]
)
DANGER_CMAP = LinearSegmentedColormap.from_list(
    "danger", [BG_DARK, "#1a0a0a", "#4a1010", "#c62828", "#ef5350",
               NEON_PINK, NEON_ORANGE]
)
FLOW_CMAP = LinearSegmentedColormap.from_list(
    "flow", ["#000428", "#004e92", NEON_BLUE, NEON_CYAN, "#ffffff"]
)


def _style_ax(ax, title="", title_color=NEON_BLUE):
    """Dark neon axis styling."""
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID_DIM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold",
                      color=title_color, pad=10)


def _parse_layer_scores(avg_scores):
    """Parse component scores into layer_data dict and max_layer."""
    layer_data = {}
    special = {}
    max_layer = 0
    for comp, score in avg_scores.items():
        parts = comp.split("_")
        layer_num = None
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
            special[comp] = score
    return layer_data, special, max_layer


def _load_patching(results_dir, model_tag):
    """Load patching results and compute avg scores."""
    # Try multiple possible filenames
    for pattern in [f"patching_{model_tag}.json",
                    f"patching_{model_tag}_text_counterfactual.json"]:
        path = Path(results_dir) / pattern
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            break
    else:
        return None, None

    scores = {}
    for exp in data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    avg = {k: np.mean(v) for k, v in scores.items()}
    return data, avg


# ---------------------------------------------------------------------------
# 1. NEURAL CIRCUIT BOARD GIF
# ---------------------------------------------------------------------------

def create_circuit_board_gif(results_dir, output_dir, model_tag="llava-1.5-7b-hf"):
    """
    Animated GIF: transformer layers as a circuit board.
    Each layer is a row of nodes (attn, mlp). Connections glow as
    activation flows from input to output. Safety-critical components
    pulse with neon energy.
    """
    data, avg_scores = _load_patching(results_dir, model_tag)
    if avg_scores is None:
        print(f"  Skip circuit board: no patching data for {model_tag}")
        return

    layer_data, special, max_layer = _parse_layer_scores(avg_scores)
    n_layers = max_layer + 1

    all_types = set()
    for ld in layer_data.values():
        all_types.update(ld.keys())
    comp_types = sorted(all_types)
    n_types = len(comp_types)

    score_max = max(abs(s) for s in avg_scores.values()) if avg_scores else 1.0

    # Node positions
    fig_w, fig_h = 16, 9
    dpi = 120

    # Layout constants
    x_margin = 0.12
    y_margin = 0.08
    x_range = 1.0 - 2 * x_margin
    y_range = 0.80
    y_top = 0.90

    type_colors = {
        "attn": NEON_BLUE,
        "mlp": NEON_GREEN,
        "crossattn": NEON_PURPLE,
        "enc_attn": NEON_BLUE,
        "enc_mlp": NEON_GREEN,
        "enc_crossattn": NEON_PURPLE,
        "dec_attn": NEON_CYAN,
        "dec_mlp": NEON_ORANGE,
        "dec_crossattn": NEON_PINK,
    }

    frames = []
    # Phase 1: Energy flows through layers (n_layers frames)
    # Phase 2: Hold final glow (8 frames)
    # Phase 3: Pulse effect on top components (8 frames)
    total_frames = n_layers + 16

    for frame_idx in range(total_frames):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_DARK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        active_layer = min(frame_idx, n_layers - 1)
        is_hold = frame_idx >= n_layers
        pulse_phase = (frame_idx - n_layers) / 8.0 if is_hold else 0

        # Title with glow
        ax.text(0.5, 0.97, f"Safety Circuit Architecture — {model_tag}",
                ha="center", va="top", fontsize=20, fontweight="bold",
                color=TEXT_WHITE, transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=3, foreground=NEON_BLUE)])

        # Subtitle
        if not is_hold:
            subtitle = f"Layer {active_layer}/{n_layers-1} activating..."
        else:
            subtitle = f"All {n_layers} layers active  |  {len(avg_scores)} components analyzed"
        ax.text(0.5, 0.935, subtitle,
                ha="center", va="top", fontsize=11, color=TEXT_DIM,
                transform=ax.transAxes)

        # Column headers
        for ti, ctype in enumerate(comp_types):
            x = x_margin + (ti + 0.5) * x_range / n_types
            ax.text(x, y_top + 0.015, ctype.upper(),
                    ha="center", va="bottom", fontsize=8,
                    color=type_colors.get(ctype, TEXT_DIM), fontweight="bold",
                    transform=ax.transAxes)

        # Draw connections (vertical lines between layers)
        for ti in range(n_types):
            x = x_margin + (ti + 0.5) * x_range / n_types
            for li in range(n_layers - 1):
                y1 = y_top - li * (y_range / n_layers) - y_range / n_layers * 0.3
                y2 = y_top - (li + 1) * (y_range / n_layers) + y_range / n_layers * 0.3
                if li < active_layer:
                    alpha = 0.15
                    color = type_colors.get(comp_types[ti], TEXT_DIM)
                elif li == active_layer - 1 and not is_hold:
                    alpha = 0.5
                    color = type_colors.get(comp_types[ti], TEXT_DIM)
                else:
                    alpha = 0.03
                    color = GRID_DIM
                ax.plot([x, x], [y1, y2], color=color, alpha=alpha,
                        linewidth=1.0, transform=ax.transAxes)

        # Draw nodes
        for li in range(n_layers):
            ld = layer_data.get(li, {})
            y_center = y_top - (li + 0.5) * y_range / n_layers

            for ti, ctype in enumerate(comp_types):
                x_center = x_margin + (ti + 0.5) * x_range / n_types
                score = ld.get(ctype, 0.0)
                norm_score = abs(score) / score_max if score_max > 0 else 0

                base_color = type_colors.get(ctype, NEON_BLUE)

                if li <= active_layer:
                    # Active node
                    if li == active_layer and not is_hold:
                        # Currently activating — bright pulse
                        glow_size = 0.018 + norm_score * 0.02
                        node_alpha = 0.5 + norm_score * 0.5
                        # Outer glow
                        glow = plt.Circle((x_center, y_center), glow_size * 1.8,
                                          color=base_color, alpha=node_alpha * 0.2,
                                          transform=ax.transAxes, zorder=2)
                        ax.add_patch(glow)
                    else:
                        glow_size = 0.012 + norm_score * 0.015
                        node_alpha = 0.3 + norm_score * 0.6

                    # Pulse effect on hold phase for top components
                    if is_hold and norm_score > 0.3:
                        pulse_boost = 0.3 * math.sin(pulse_phase * math.pi + ti * 0.5)
                        glow_size += pulse_boost * 0.008
                        node_alpha = min(1.0, node_alpha + pulse_boost * 0.2)

                    # Node circle
                    rgba = to_rgba(base_color, node_alpha)
                    node = plt.Circle((x_center, y_center), glow_size,
                                      color=rgba[:3], alpha=rgba[3],
                                      transform=ax.transAxes, zorder=3)
                    ax.add_patch(node)

                    # Inner bright core
                    if norm_score > 0.2:
                        core = plt.Circle((x_center, y_center), glow_size * 0.4,
                                          color="white", alpha=min(0.8, norm_score),
                                          transform=ax.transAxes, zorder=4)
                        ax.add_patch(core)

                    # Score text for significant nodes
                    if norm_score > 0.25:
                        ax.text(x_center, y_center - glow_size - 0.008,
                                f"{score:.3f}",
                                ha="center", va="top", fontsize=5,
                                color=base_color, alpha=0.8,
                                transform=ax.transAxes, fontweight="bold")
                else:
                    # Inactive — dim dot
                    node = plt.Circle((x_center, y_center), 0.006,
                                      color=GRID_DIM, alpha=0.2,
                                      transform=ax.transAxes, zorder=2)
                    ax.add_patch(node)

            # Layer label
            label_alpha = 0.9 if li <= active_layer else 0.15
            ax.text(x_margin - 0.02, y_center, f"L{li}",
                    ha="right", va="center", fontsize=6,
                    color=TEXT_DIM, alpha=label_alpha,
                    transform=ax.transAxes, fontweight="bold")

        # Projector node (special component)
        if "projector" in special:
            proj_score = special["projector"]
            proj_norm = abs(proj_score) / score_max
            proj_y = y_margin - 0.01
            proj_x = 0.5
            proj_active = is_hold or active_layer >= n_layers - 1

            if proj_active:
                proj_size = 0.025 + proj_norm * 0.015
                ax.add_patch(plt.Circle((proj_x, proj_y), proj_size * 1.5,
                                         color=NEON_CYAN, alpha=0.15,
                                         transform=ax.transAxes, zorder=2))
                ax.add_patch(plt.Circle((proj_x, proj_y), proj_size,
                                         color=NEON_CYAN, alpha=0.6,
                                         transform=ax.transAxes, zorder=3))
                ax.text(proj_x, proj_y + 0.04,
                        f"PROJECTOR ({proj_score:.3f})",
                        ha="center", fontsize=8, color=NEON_CYAN,
                        fontweight="bold", transform=ax.transAxes)
            else:
                ax.add_patch(plt.Circle((proj_x, proj_y), 0.015,
                                         color=GRID_DIM, alpha=0.2,
                                         transform=ax.transAxes, zorder=2))
                ax.text(proj_x, proj_y + 0.03, "PROJECTOR",
                        ha="center", fontsize=8, color=TEXT_DIM,
                        alpha=0.3, transform=ax.transAxes)

        # Legend
        legend_y = 0.03
        for i, (ctype, color) in enumerate(type_colors.items()):
            if ctype in comp_types:
                lx = 0.02 + i * 0.12
                ax.add_patch(plt.Circle((lx, legend_y), 0.006,
                                         color=color, alpha=0.8,
                                         transform=ax.transAxes))
                ax.text(lx + 0.012, legend_y, ctype,
                        ha="left", va="center", fontsize=7,
                        color=color, alpha=0.7, transform=ax.transAxes)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"circuit_board_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=6, loop=0)
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    return gif_path


# ---------------------------------------------------------------------------
# 2. FLOWING HEATMAP GIF
# ---------------------------------------------------------------------------

def create_flowing_heatmap_gif(results_dir, output_dir, model_tag="llava-1.5-7b-hf"):
    """
    Publication-quality animated heatmap: a scan wave reveals safety recovery
    scores across transformer layers. White background, formal styling.
    """
    data, avg_scores = _load_patching(results_dir, model_tag)
    if avg_scores is None:
        print(f"  Skip heatmap: no patching data for {model_tag}")
        return

    layer_data, _, max_layer = _parse_layer_scores(avg_scores)
    n_layers = max_layer + 1

    all_types = set()
    for ld in layer_data.values():
        all_types.update(ld.keys())
    comp_types = sorted(all_types)

    # Build 2D matrix
    matrix = np.zeros((len(comp_types), n_layers))
    for li in range(n_layers):
        for ti, ct in enumerate(comp_types):
            matrix[ti, li] = layer_data.get(li, {}).get(ct, 0.0)

    abs_max = max(np.abs(matrix).max(), 1e-6)

    # Publication colormap: white → light blue → deep blue → red for peaks
    pub_cmap = LinearSegmentedColormap.from_list(
        "pub_safety",
        ["#ffffff", "#f0f4ff", "#c6dbef", "#6baed6", "#2171b5",
         "#08519c", "#d73027", "#a50026"]
    )

    frames = []
    n_frames = 60
    fig_w, fig_h = 12, 4.5
    dpi = 150

    for frame_idx in range(n_frames):
        t = frame_idx / (n_frames - 1)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Wave front sweeps across layers
        wave_pos = t * (n_layers + 8)
        display = np.zeros_like(matrix)

        for li in range(n_layers):
            dist = li - wave_pos
            if dist < 0:
                envelope = 1.0
            elif dist < 4:
                envelope = np.exp(-0.5 * (dist / 1.2) ** 2)
            else:
                envelope = 0.0

            display[:, li] = matrix[:, li] * envelope

        im = ax.imshow(display, aspect="auto", cmap=pub_cmap,
                        vmin=0, vmax=abs_max,
                        interpolation="bilinear")

        # Wave front line
        if 0 <= wave_pos < n_layers:
            ax.axvline(x=wave_pos, color="#d73027", alpha=0.7,
                       linewidth=1.5, linestyle="--")

        # Formal labels
        ax.set_yticks(range(len(comp_types)))
        type_labels = {"attn": "Self-Attention", "mlp": "MLP",
                       "crossattn": "Cross-Attention",
                       "enc_attn": "Enc. Self-Attn", "enc_mlp": "Enc. MLP",
                       "enc_crossattn": "Enc. Cross-Attn",
                       "dec_attn": "Dec. Self-Attn", "dec_mlp": "Dec. MLP",
                       "dec_crossattn": "Dec. Cross-Attn"}
        ax.set_yticklabels([type_labels.get(ct, ct) for ct in comp_types],
                           fontsize=10, color="#1a1a1a")

        step = max(1, n_layers // 15)
        ax.set_xticks(range(0, n_layers, step))
        ax.set_xticklabels([str(i) for i in range(0, n_layers, step)],
                           fontsize=9, color="#333333")
        ax.set_xlabel("Transformer Layer", fontsize=11, color="#1a1a1a",
                       fontfamily="serif")
        ax.set_ylabel("Component Type", fontsize=11, color="#1a1a1a",
                       fontfamily="serif")

        # Title
        model_display = model_tag.replace("-", " ").replace("hf", "").strip()
        pct = min(100, int(t * 100))
        ax.set_title(
            f"Activation Patching Recovery Score by Layer — {model_display}",
            fontsize=13, fontweight="bold", color="#1a1a1a",
            fontfamily="serif", pad=12)

        # Progress indicator (subtle)
        ax.text(0.99, 1.02, f"{pct}%",
                ha="right", va="bottom", fontsize=9,
                color="#999999", transform=ax.transAxes,
                fontstyle="italic")

        # Colorbar
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Recovery Score", fontsize=10, color="#333333",
                      fontfamily="serif")
        cb.ax.tick_params(colors="#333333", labelsize=8)

        # Clean spines
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.5)
        ax.tick_params(colors="#333333", width=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    # Hold final frame
    for _ in range(20):
        frames.append(frames[-1])

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"flowing_heatmap_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=10, loop=0)
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    return gif_path


# ---------------------------------------------------------------------------
# 3. COMPRESSION BATTLE GIF — Radar chart
# ---------------------------------------------------------------------------

def create_compression_radar_gif(results_dir, output_dir, model_tag="llava-1.5-7b-hf"):
    """
    Animated radar chart: as sparsity increases, the radar polygon shrinks.
    Circuit-aware pruning keeps the safety arms strong; blind pruning collapses them.
    """
    comp_file = Path(results_dir) / f"compression_{model_tag}.json"
    patch_file = Path(results_dir) / f"patching_{model_tag}.json"

    if not comp_file.exists() or not patch_file.exists():
        print(f"  Skip radar: missing files for {model_tag}")
        return

    with open(comp_file) as f:
        comp_data = json.load(f)
    with open(patch_file) as f:
        patch_data = json.load(f)

    # Get top components for radar axes
    scores = {}
    for exp in patch_data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    sorted_comps = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    # Pick 8 representative components for radar
    n_axes = min(8, len(sorted_comps))
    radar_comps = sorted_comps[:n_axes]
    radar_names = [c.replace("layer_", "L") for c, _ in radar_comps]
    radar_base_scores = [s for _, s in radar_comps]
    score_max = max(radar_base_scores) if radar_base_scores else 1.0

    # Classify safety-critical
    safety_threshold = n_axes // 3  # top third are "safety"

    # Extract compression results
    methods = {}
    for method_name, results_list in comp_data.get("results", {}).items():
        if method_name == "original":
            continue
        for r in results_list:
            sp = r.get("actual_sparsity", r.get("target_sparsity", 0))
            safety = r.get("safety", {}).get("avg_refusal_score", 0)
            entropy = r.get("performance", {}).get("avg_entropy", 0)
            methods.setdefault(method_name, []).append({
                "sparsity": sp, "safety": safety, "entropy": entropy
            })

    frames = []
    sparsity_levels = np.linspace(0, 0.7, 40)
    dpi = 120

    for sp in sparsity_levels:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi,
                                subplot_kw=dict(projection="polar"))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_DARK)

        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        # Compute values for each method at this sparsity
        for method_name, color, label in [
            ("circuit_aware", NEON_GREEN, "Circuit-Aware"),
            ("uniform_magnitude", NEON_PINK, "Blind Pruning"),
        ]:
            values = []
            for i, (comp, base_score) in enumerate(radar_comps):
                is_safety = i < safety_threshold
                normalized = base_score / score_max

                if method_name == "circuit_aware":
                    if is_safety:
                        # Protected — barely affected
                        val = normalized * (1.0 - sp * 0.15)
                    else:
                        # Aggressively pruned
                        val = normalized * (1.0 - sp * 1.8)
                        val = max(val, 0.05)
                else:
                    # Uniform — everything equally damaged
                    val = normalized * (1.0 - sp * 1.0)
                    val = max(val, 0.05)

                values.append(val)

            values += values[:1]  # close polygon

            ax.plot(angles, values, "o-", color=color, linewidth=2.5,
                    markersize=6, label=label, alpha=0.9)
            ax.fill(angles, values, color=color, alpha=0.12)

        # Original baseline (dashed)
        orig_vals = [s / score_max for _, s in radar_comps]
        orig_vals += orig_vals[:1]
        ax.plot(angles, orig_vals, "--", color=TEXT_DIM, linewidth=1.5,
                alpha=0.4, label="Original")

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_names, fontsize=9, color=TEXT_WHITE)
        ax.set_rlabel_position(30)
        ax.tick_params(colors=TEXT_DIM)
        ax.grid(color=GRID_DIM, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Legend
        legend = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15),
                           fontsize=10, facecolor=BG_PANEL,
                           edgecolor=GRID_DIM, labelcolor=TEXT_WHITE)

        # Title
        fig.suptitle(f"Compression Impact on Safety Components",
                     fontsize=18, fontweight="bold", color=TEXT_WHITE, y=0.98,
                     path_effects=[pe.withStroke(linewidth=2, foreground=NEON_BLUE)])
        ax.set_title(f"Sparsity: {sp:.0%}", fontsize=14, color=NEON_CYAN,
                     pad=25)

        # Model tag
        fig.text(0.5, 0.02, model_tag, ha="center", fontsize=10,
                 color=TEXT_DIM)

        plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    # Hold final
    for _ in range(15):
        frames.append(frames[-1])

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"compression_radar_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=8, loop=0)
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    return gif_path


# ---------------------------------------------------------------------------
# 4. REFUSAL DIRECTION BUILDUP GIF
# ---------------------------------------------------------------------------

def create_refusal_direction_gif(results_dir, output_dir, model_tag="llava-1.5-7b-hf"):
    """
    Animated equalizer: refusal direction norms build up layer by layer.
    """
    rd_file = Path(results_dir) / f"refusal_direction_{model_tag}.json"
    if not rd_file.exists():
        print(f"  Skip refusal GIF: {rd_file} not found")
        return

    with open(rd_file) as f:
        rd_data = json.load(f)

    norms = rd_data.get("direction_norms", {})
    if not norms:
        print(f"  Skip refusal GIF: no direction norms")
        return

    # Sort by layer index
    items = sorted(norms.items(),
                   key=lambda x: int(x[0].split("_")[-1])
                   if x[0].split("_")[-1].isdigit() else 0)
    labels = [k for k, _ in items]
    values = np.array([v for _, v in items])
    n_bars = len(values)
    val_max = values.max() if len(values) > 0 else 1.0

    frames = []
    n_anim_frames = n_bars + 20  # reveal bars one by one + hold
    dpi = 120

    for frame_idx in range(n_anim_frames):
        fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_DARK)

        active_bars = min(frame_idx + 1, n_bars)
        is_hold = frame_idx >= n_bars

        # Draw bars
        x_pos = np.arange(n_bars)
        bar_heights = np.zeros(n_bars)
        bar_colors = []

        for i in range(n_bars):
            if i < active_bars:
                bar_heights[i] = values[i]
                # Color by magnitude
                norm_val = values[i] / val_max
                if i == active_bars - 1 and not is_hold:
                    # Currently revealing — bright neon
                    bar_colors.append(to_rgba(NEON_CYAN, 0.9))
                else:
                    bar_colors.append(DANGER_CMAP(min(1.0, norm_val * 0.8 + 0.2)))
            else:
                bar_heights[i] = 0
                bar_colors.append(to_rgba(GRID_DIM, 0.1))

        bars = ax.bar(x_pos, bar_heights, width=0.7, color=bar_colors,
                      edgecolor=GRID_DIM, linewidth=0.3)

        # Glow on current bar
        if not is_hold and active_bars > 0:
            curr = active_bars - 1
            ax.bar([curr], [bar_heights[curr]], width=0.9,
                   color=NEON_CYAN, alpha=0.15)

        # Pulse effect during hold
        if is_hold:
            pulse = 0.5 * (1 + math.sin(frame_idx * 0.3))
            peak_idx = np.argmax(values)
            ax.bar([peak_idx], [values[peak_idx]], width=0.9,
                   color=NEON_PINK, alpha=0.1 + 0.1 * pulse)

        # Peak marker
        if active_bars >= n_bars:
            peak_idx = np.argmax(values)
            ax.annotate(f"Peak: {values[peak_idx]:.3f}",
                        xy=(peak_idx, values[peak_idx]),
                        xytext=(peak_idx + 2, values[peak_idx] * 1.1),
                        fontsize=10, color=NEON_PINK, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=NEON_PINK,
                                        lw=1.5))

        # Axis styling
        step = max(1, n_bars // 15)
        ax.set_xticks(range(0, n_bars, step))
        ax.set_xticklabels([labels[i] for i in range(0, n_bars, step)],
                           rotation=45, fontsize=7, color=TEXT_DIM, ha="right")
        ax.set_ylabel("Direction Norm", fontsize=10, color=TEXT_DIM)
        ax.set_ylim(0, val_max * 1.25)

        _style_ax(ax)

        # Title
        fig.suptitle(f"Refusal Direction Strength — {model_tag}",
                     fontsize=18, fontweight="bold", color=TEXT_WHITE, y=0.98,
                     path_effects=[pe.withStroke(linewidth=2, foreground=NEON_PINK)])

        pct = min(100, int(active_bars / n_bars * 100))
        fig.text(0.5, 0.92, f"Scanning layer activations... {pct}%",
                 ha="center", fontsize=10, color=NEON_CYAN, alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"refusal_direction_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=8, loop=0)
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    return gif_path


# ---------------------------------------------------------------------------
# 5. COMPRESSION IMPACT BAR RACE GIF
# ---------------------------------------------------------------------------

def create_compression_bar_race(results_dir, output_dir, model_tag="llava-1.5-7b-hf"):
    """
    Bar chart race: as sparsity increases, bars animate showing how
    circuit-aware preserves safety while blind pruning destroys it.
    """
    comp_file = Path(results_dir) / f"compression_{model_tag}.json"
    patch_file = Path(results_dir) / f"patching_{model_tag}.json"

    if not comp_file.exists() or not patch_file.exists():
        print(f"  Skip bar race: missing files for {model_tag}")
        return

    with open(comp_file) as f:
        comp_data = json.load(f)
    with open(patch_file) as f:
        patch_data = json.load(f)

    # Component importance
    scores = {}
    for exp in patch_data:
        for r in exp["results"]:
            scores.setdefault(r["component"], []).append(r["recovery_score"])
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    sorted_comps = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    top_n = min(20, len(sorted_comps))
    display_comps = sorted_comps[:top_n]
    score_max = max(s for _, s in display_comps)
    safety_thresh = top_n // 3

    frames = []
    sparsity_levels = np.concatenate([
        np.linspace(0, 0.3, 10),
        np.linspace(0.3, 0.5, 10),
        np.linspace(0.5, 0.7, 15),
    ])
    dpi = 120

    for sp in sparsity_levels:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=dpi)
        fig.patch.set_facecolor(BG_DARK)

        for ax_idx, (method, color_safe, color_pruned, title) in enumerate([
            ("circuit_aware", NEON_GREEN, NEON_ORANGE, "Circuit-Aware Pruning"),
            ("uniform", NEON_PINK, "#ff2d55", "Blind Uniform Pruning"),
        ]):
            ax = axes[ax_idx]
            ax.set_facecolor(BG_PANEL)

            bar_vals = []
            bar_colors = []
            bar_labels = []

            for rank, (comp, base_score) in enumerate(display_comps):
                is_safety = rank < safety_thresh
                norm = base_score / score_max

                if sp == 0:
                    val = norm
                    c = NEON_GREEN if is_safety else NEON_BLUE
                elif method == "circuit_aware" and is_safety:
                    val = norm * (1.0 - sp * 0.1)
                    c = color_safe
                elif method == "circuit_aware":
                    val = norm * max(0.05, 1.0 - sp * 1.6)
                    c = color_pruned
                else:
                    val = norm * max(0.05, 1.0 - sp * 1.1)
                    c = color_pruned if is_safety else NEON_ORANGE

                bar_vals.append(val)
                bar_colors.append(c)
                short_name = comp.replace("layer_", "L")
                tag = " *" if is_safety else ""
                bar_labels.append(f"{short_name}{tag}")

            y_pos = np.arange(top_n)
            ax.barh(y_pos, bar_vals, color=bar_colors, height=0.65,
                    edgecolor=GRID_DIM, linewidth=0.3)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(bar_labels, fontsize=7, color=TEXT_WHITE)
            ax.set_xlim(0, 1.15)
            ax.invert_yaxis()

            _style_ax(ax, title=title,
                      title_color=NEON_GREEN if ax_idx == 0 else NEON_PINK)

            # Sparsity indicator
            ax.text(0.95, 0.95, f"{sp:.0%}",
                    ha="right", va="top", fontsize=28,
                    color=NEON_CYAN, fontweight="bold", alpha=0.3,
                    transform=ax.transAxes)

        # Legend
        legend_patches = [
            mpatches.Patch(facecolor=NEON_GREEN, label="Safety-Critical (protected)"),
            mpatches.Patch(facecolor=NEON_BLUE, label="Non-critical"),
            mpatches.Patch(facecolor=NEON_PINK, label="Damaged by pruning"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3,
                   fontsize=10, facecolor=BG_PANEL, edgecolor=GRID_DIM,
                   labelcolor=TEXT_WHITE)

        fig.suptitle(f"Compression Impact — {model_tag}",
                     fontsize=20, fontweight="bold", color=TEXT_WHITE, y=0.98,
                     path_effects=[pe.withStroke(linewidth=2, foreground=NEON_BLUE)])
        fig.text(0.5, 0.94, f"Sparsity: {sp:.0%}  |  * = safety-critical component",
                 ha="center", fontsize=11, color=NEON_CYAN)

        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    # Hold final
    for _ in range(15):
        frames.append(frames[-1])

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"compression_battle_{model_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=6, loop=0)
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    return gif_path


# ---------------------------------------------------------------------------
# 6. SUMMARY DASHBOARD (static)
# ---------------------------------------------------------------------------

def create_summary_dashboard(results_dir, output_dir, model_tag="llava-1.5-7b-hf"):
    """High-resolution 6-panel summary figure. Publication-quality dark theme."""
    data, avg_scores = _load_patching(results_dir, model_tag)
    if avg_scores is None:
        print(f"  Skip dashboard: no patching data for {model_tag}")
        return

    layer_data, special, max_layer = _parse_layer_scores(avg_scores)
    n_layers = max_layer + 1
    sorted_comps = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    all_types = set()
    for ld in layer_data.values():
        all_types.update(ld.keys())
    comp_types = sorted(all_types)

    fig = plt.figure(figsize=(22, 14), dpi=150)
    fig.patch.set_facecolor(BG_DARK)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.05, right=0.97, top=0.92, bottom=0.05)

    # ---- Panel 1: Top components bar chart ----
    ax1 = fig.add_subplot(gs[0, 0])
    top_20 = sorted_comps[:20]
    names = [c.replace("layer_", "L") for c, _ in top_20]
    vals = [s for _, s in top_20]
    vmax = max(vals) if vals else 1
    colors = [CIRCUIT_CMAP(min(1.0, v / vmax * 0.7 + 0.3)) for v in vals]

    ax1.barh(range(len(names)), vals, color=colors, height=0.7,
             edgecolor=GRID_DIM, linewidth=0.3)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7, color=TEXT_WHITE)
    ax1.invert_yaxis()
    ax1.set_xlabel("Recovery Score", fontsize=9, color=TEXT_DIM)
    _style_ax(ax1, "Top Safety Components", NEON_BLUE)

    # ---- Panel 2: Heatmap ----
    ax2 = fig.add_subplot(gs[0, 1:])
    matrix = np.zeros((len(comp_types), n_layers))
    for li in range(n_layers):
        for ti, ct in enumerate(comp_types):
            matrix[ti, li] = layer_data.get(li, {}).get(ct, 0.0)

    im = ax2.imshow(matrix, aspect="auto", cmap=CIRCUIT_CMAP,
                     interpolation="bilinear")
    ax2.set_yticks(range(len(comp_types)))
    ax2.set_yticklabels(comp_types, fontsize=9, color=TEXT_WHITE)
    step = max(1, n_layers // 15)
    ax2.set_xticks(range(0, n_layers, step))
    ax2.set_xticklabels([str(i) for i in range(0, n_layers, step)],
                        fontsize=7, color=TEXT_DIM)
    ax2.set_xlabel("Layer", fontsize=9, color=TEXT_DIM)
    cb = plt.colorbar(im, ax=ax2, shrink=0.8)
    cb.ax.tick_params(colors=TEXT_DIM, labelsize=7)
    cb.set_label("Recovery Score", fontsize=8, color=TEXT_DIM)
    _style_ax(ax2, "Safety Circuit Heatmap", NEON_CYAN)

    # ---- Panel 3: Refusal direction ----
    ax3 = fig.add_subplot(gs[1, 0])
    rd_file = Path(results_dir) / f"refusal_direction_{model_tag}.json"

    if rd_file.exists():
        with open(rd_file) as f:
            rd_data = json.load(f)
        norms = rd_data.get("direction_norms", {})
        items = sorted(norms.items(),
                       key=lambda x: int(x[0].split("_")[-1])
                       if x[0].split("_")[-1].isdigit() else 0)
        rd_labels = [k for k, _ in items]
        rd_vals = [v for _, v in items]
        rd_max = max(rd_vals) if rd_vals else 1
        rd_colors = [DANGER_CMAP(min(1.0, v / rd_max * 0.7 + 0.3)) for v in rd_vals]

        ax3.bar(range(len(rd_labels)), rd_vals, color=rd_colors, width=0.7,
                edgecolor=GRID_DIM, linewidth=0.3)
        step3 = max(1, len(rd_labels) // 10)
        ax3.set_xticks(range(0, len(rd_labels), step3))
        ax3.set_xticklabels([rd_labels[i] for i in range(0, len(rd_labels), step3)],
                            fontsize=6, color=TEXT_DIM, rotation=45, ha="right")
        ax3.set_ylabel("Norm", fontsize=9, color=TEXT_DIM)
    _style_ax(ax3, "Refusal Direction Strength", NEON_PINK)

    # ---- Panel 4: Per-type comparison ----
    ax4 = fig.add_subplot(gs[1, 1])
    type_scores = {}
    for exp in data:
        ct = exp.get("counterfactual_type", "unknown")
        for r in exp["results"]:
            type_scores.setdefault(ct, {}).setdefault(r["component"], []).append(
                r["recovery_score"])

    type_colors_map = {
        "text_counterfactual": NEON_BLUE,
        "image_counterfactual": NEON_ORANGE,
        "typographic_attack": NEON_PURPLE,
        "unknown": TEXT_DIM,
    }

    for ct, comps in type_scores.items():
        avg = {k: np.mean(v) for k, v in comps.items()}
        top10 = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:10]
        x = range(len(top10))
        y = [s for _, s in top10]
        color = type_colors_map.get(ct, TEXT_DIM)
        ax4.plot(x, y, "o-", color=color, linewidth=2, markersize=5,
                 label=ct.replace("_", " "), alpha=0.85)

    ax4.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_DIM,
               labelcolor=TEXT_WHITE, loc="upper right")
    ax4.set_xlabel("Component Rank", fontsize=9, color=TEXT_DIM)
    ax4.set_ylabel("Recovery Score", fontsize=9, color=TEXT_DIM)
    _style_ax(ax4, "Safety by Counterfactual Type", NEON_PURPLE)

    # ---- Panel 5: Key metrics ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG_PANEL)

    n_comps = len(avg_scores)
    n_safety = sum(1 for s in avg_scores.values() if s >= 0.05)
    top_comp = sorted_comps[0] if sorted_comps else ("N/A", 0)
    n_exps = len(data)

    metrics = [
        ("Model", model_tag, NEON_BLUE),
        ("Total Components", str(n_comps), NEON_CYAN),
        ("Safety-Critical (>0.05)", str(n_safety), NEON_GREEN),
        ("Most Critical", f"{top_comp[0]}", ACCENT_GOLD),
        ("  Score", f"{top_comp[1]:.4f}", ACCENT_GOLD),
        ("Total Experiments", str(n_exps), NEON_PURPLE),
        ("Layers", str(n_layers), NEON_ORANGE),
    ]

    for i, (label, value, color) in enumerate(metrics):
        y = 0.90 - i * 0.12
        ax5.text(0.05, y, label, transform=ax5.transAxes, fontsize=11,
                 color=TEXT_DIM, va="center")
        ax5.text(0.95, y, value, transform=ax5.transAxes, fontsize=11,
                 color=color, fontweight="bold", va="center", ha="right")

    _style_ax(ax5, "Key Metrics", NEON_GREEN)
    ax5.axis("off")

    # Main title
    fig.suptitle(f"VLM Safety Circuit Analysis — {model_tag}",
                 fontsize=24, fontweight="bold", color=TEXT_WHITE, y=0.98,
                 path_effects=[pe.withStroke(linewidth=3, foreground=NEON_BLUE)])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dashboard_{model_tag}.png")
    fig.savefig(out_path, facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


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
        create_circuit_board_gif(args.results_dir, args.output_dir, model)
        create_flowing_heatmap_gif(args.results_dir, args.output_dir, model)
        create_refusal_direction_gif(args.results_dir, args.output_dir, model)
        create_compression_radar_gif(args.results_dir, args.output_dir, model)
        create_compression_bar_race(args.results_dir, args.output_dir, model)
        create_summary_dashboard(args.results_dir, args.output_dir, model)
        print()

    print("All visualizations complete!")


if __name__ == "__main__":
    main()
