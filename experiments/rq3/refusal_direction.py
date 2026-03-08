"""
Refusal Direction Analysis for VLMs

Extends Chhabra & Khalili (2025) to VLMs: identifies the direction in
residual stream space that mediates refusal / safety-relevant behavior.

Supported models:
  - LLaVA family (LlavaForConditionalGeneration) — hooks language_model layers
  - BLIP-VQA-base (BlipForQuestionAnswering) — hooks text_encoder + text_decoder

For BLIP-VQA (no refusal behavior), we find the direction that maximally
separates harmful vs. benign prompt representations — the "safety-sensitive
direction" — and test if ablating it changes model outputs on harmful inputs.

Usage:
  python refusal_direction.py --model llava-hf/llava-1.5-7b-hf --device cuda
  python refusal_direction.py --model Salesforce/blip-vqa-base --device cpu
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from activation_patching import (
    REFUSAL_PREFIXES,
    compute_refusal_probability,
    compute_output_divergence,
    create_placeholder_image,
    format_prompt,
    load_model,
    load_dataset_entries,
    load_entry_images,
)


# ---------------------------------------------------------------------------
# Residual stream hooks per architecture
# ---------------------------------------------------------------------------

def get_residual_layers(model, model_name: str):
    """
    Returns the list of transformer layers whose output = residual stream.
    Also returns a label prefix for naming.
    """
    if "blip-vqa" in model_name.lower():
        # Return both encoder and decoder layers, labeled separately
        enc_layers = list(model.text_encoder.encoder.layer)
        dec_layers = list(model.text_decoder.bert.encoder.layer)
        labels = ([f"enc_{i}" for i in range(len(enc_layers))] +
                  [f"dec_{i}" for i in range(len(dec_layers))])
        return list(enc_layers) + list(dec_layers), labels
    else:
        # Resolve language model:
        #   Transformers 4.x: model.language_model (LlamaForCausalLM)
        #   Transformers 5.x: model.model.language_model (LlamaModel)
        if hasattr(model, 'language_model'):
            lang_model = model.language_model
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            lang_model = model.model.language_model
        elif hasattr(model, 'model'):
            lang_model = model.model
        else:
            raise ValueError(f"Cannot find language model. Children: {[n for n, _ in model.named_children()]}")

        layers = None
        if hasattr(lang_model, 'model') and hasattr(lang_model.model, 'layers'):
            layers = list(lang_model.model.layers)  # 4.x: LlamaForCausalLM.model.layers
        elif hasattr(lang_model, 'layers'):
            layers = list(lang_model.layers)  # 5.x: LlamaModel.layers
        elif hasattr(lang_model, 'transformer') and hasattr(lang_model.transformer, 'h'):
            layers = list(lang_model.transformer.h)
        else:
            for name, module in lang_model.named_modules():
                if name.endswith('.layers') and hasattr(module, '__len__') and len(module) > 5:
                    layers = list(module)
                    break

        if layers is None:
            raise ValueError(
                f"Cannot find transformer layers. "
                f"lang_model children: {[n for n, _ in lang_model.named_children()]}"
            )
        labels = [f"layer_{i}" for i in range(len(layers))]
        return layers, labels


@torch.no_grad()
def collect_residual_activations(model, processor, text, image, device,
                                  layers, labels, model_name=""):
    """
    Run a forward pass and collect the residual stream output at every layer.
    Returns dict: label -> tensor of shape (hidden_dim,) [mean-pooled over seq].
    """
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # BLIP-VQA forward() requires decoder_input_ids
    if "blip-vqa" in model_name and "decoder_input_ids" not in inputs:
        bos_id = processor.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = processor.tokenizer.cls_token_id or 0
        inputs["decoder_input_ids"] = torch.tensor([[bos_id]], device=device)

    activations = {}
    hooks = []

    for layer, label in zip(layers, labels):
        def make_hook(lbl):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                # Mean-pool over sequence dimension
                activations[lbl] = act[0].mean(dim=0).detach().cpu().float()
            return hook_fn
        h = layer.register_forward_hook(make_hook(label))
        hooks.append(h)

    model(**inputs)

    for h in hooks:
        h.remove()

    return activations


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_refusal_directions(
    model, processor, prompts, default_image, device, model_name,
    dataset_dir=None,
):
    """
    For each layer, compute the mean activation difference between
    harmful and benign prompts. The dominant direction of this difference
    is the refusal/safety-sensitive direction.
    """
    layers, labels = get_residual_layers(model, model_name)

    harmful_acts = {}  # label -> list of (hidden_dim,) tensors
    benign_acts = {}

    if dataset_dir is None:
        dataset_dir = Path(__file__).parent / "data" / "safety_dataset"

    # Known entries that cause unrecoverable CUDA device-side asserts
    skip_entries = {"img_pair_031"}

    for prompt_data in tqdm(prompts, desc="Collecting activations"):
        if prompt_data.get("id") in skip_entries:
            continue
        # Load per-entry images if available, else use default
        harmful_image, benign_image = load_entry_images(prompt_data, dataset_dir)

        harmful_text = format_prompt(prompt_data["harmful"]["text"], model_name)
        h = collect_residual_activations(
            model, processor, harmful_text, harmful_image, device, layers, labels, model_name
        )
        for label, act in h.items():
            harmful_acts.setdefault(label, []).append(act)

        benign_text = format_prompt(prompt_data["benign"]["text"], model_name)
        b = collect_residual_activations(
            model, processor, benign_text, benign_image, device, layers, labels, model_name
        )
        for label, act in b.items():
            benign_acts.setdefault(label, []).append(act)

    # Compute mean difference per layer
    refusal_directions = {}
    direction_norms = {}

    for label in sorted(harmful_acts.keys()):
        h_mean = torch.stack(harmful_acts[label]).mean(dim=0)
        b_mean = torch.stack(benign_acts[label]).mean(dim=0)

        diff = h_mean - b_mean
        norm = diff.norm().item()
        direction = diff / (diff.norm() + 1e-8)

        refusal_directions[label] = direction
        direction_norms[label] = norm

    return refusal_directions, direction_norms


@torch.no_grad()
def test_direction_ablation(
    model, processor, tokenizer, text, image, device, model_name,
    refusal_directions, target_labels=None,
):
    """
    Ablate the refusal direction at specified layers and measure effect.
    For LLaVA: measures change in refusal probability.
    For BLIP-VQA: measures change in output distribution.
    """
    layers, labels = get_residual_layers(model, model_name)
    label_to_layer = dict(zip(labels, layers))

    if target_labels is None:
        target_labels = list(refusal_directions.keys())

    # Baseline forward pass
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # BLIP-VQA forward() requires decoder_input_ids
    if "blip-vqa" in model_name.lower() and "decoder_input_ids" not in inputs:
        bos_id = processor.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = processor.tokenizer.cls_token_id or 0
        inputs["decoder_input_ids"] = torch.tensor([[bos_id]], device=device)

    outputs = model(**inputs)
    baseline_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    if "blip-vqa" in model_name.lower():
        baseline_metric = 0.0  # will compare ablated vs baseline via divergence
    else:
        baseline_metric = compute_refusal_probability(baseline_logits, tokenizer)

    # Ablate refusal direction at target layers
    hooks = []
    for label in target_labels:
        if label not in refusal_directions or label not in label_to_layer:
            continue
        layer_module = label_to_layer[label]
        direction = refusal_directions[label].to(device)

        def make_ablation_hook(d):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                    proj = torch.einsum("...d,d->...", act.float(), d).unsqueeze(-1) * d
                    return (act - proj.to(act.dtype),) + output[1:]
                else:
                    proj = torch.einsum("...d,d->...", output.float(), d).unsqueeze(-1) * d
                    return output - proj.to(output.dtype)
            return hook_fn

        h = layer_module.register_forward_hook(make_ablation_hook(direction))
        hooks.append(h)

    outputs_ablated = model(**inputs)
    ablated_logits = outputs_ablated.logits if hasattr(outputs_ablated, 'logits') else outputs_ablated[0]

    for h in hooks:
        h.remove()

    if "blip-vqa" in model_name.lower():
        ablated_metric = compute_output_divergence(ablated_logits, baseline_logits)
        baseline_metric = 0.0  # baseline vs itself
    else:
        ablated_metric = compute_refusal_probability(ablated_logits, tokenizer)

    return baseline_metric, ablated_metric


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_refusal_directions(direction_norms, ablation_results, output_dir,
                             model_name=""):
    os.makedirs(output_dir, exist_ok=True)
    model_tag = model_name.split("/")[-1] if model_name else "model"

    # Plot 1: Direction norm per layer
    labels_sorted = sorted(direction_norms.keys(),
                            key=lambda x: (x.split("_")[0], int(x.split("_")[-1])
                                           if x.split("_")[-1].isdigit() else 0))
    norms = [direction_norms[l] for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(max(14, len(labels_sorted) * 0.5), 5))
    colors = []
    for l in labels_sorted:
        if l.startswith("enc"):
            colors.append("#3498db")
        elif l.startswith("dec"):
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    ax.bar(range(len(labels_sorted)), norms, color=colors)
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Direction Norm (||mean_harmful - mean_benign||)")
    ax.set_title(f"Safety-Sensitive Direction Strength — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"refusal_direction_norms_{model_tag}.png"), dpi=150)
    plt.close()

    # Plot 2: Ablation effect
    if ablation_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        prompt_ids = [r["prompt_id"] for r in ablation_results]
        baselines = [r["baseline_metric"] for r in ablation_results]
        ablated = [r["ablated_metric"] for r in ablation_results]

        x = np.arange(len(prompt_ids))
        width = 0.35
        ax.bar(x - width/2, baselines, width, label="Baseline", color="#3498db")
        ax.bar(x + width/2, ablated, width, label="Direction Ablated", color="#e74c3c")
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Metric")
        ax.set_title(f"Effect of Ablating Safety Direction — {model_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_ids, rotation=45, ha="right", fontsize=7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"direction_ablation_{model_tag}.png"), dpi=150)
        plt.close()

    print(f"  Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    model, tokenizer, processor = load_model(
        args.model, args.device, args.dtype,
        use_flash_attn=getattr(args, 'flash_attn', False),
        compile_model=getattr(args, 'compile', False),
    )

    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    dataset_path = dataset_dir / "dataset.json"
    prompts = load_dataset_entries(str(dataset_path), num_prompts=args.num_prompts)

    # Default image for entries without images
    default_image = create_placeholder_image()

    # Step 1: Compute refusal/safety directions
    print(f"\n{'='*60}")
    print("Step 1: Computing safety-sensitive directions per layer")
    refusal_directions, direction_norms = compute_refusal_directions(
        model, processor, prompts, default_image, args.device, args.model,
        dataset_dir=dataset_dir,
    )

    print("\nDirection norms (top 10):")
    top_labels = sorted(direction_norms.items(), key=lambda x: x[1], reverse=True)[:10]
    for label, norm in top_labels:
        print(f"  {label}: {norm:.4f}")

    # Step 2: Test ablation
    print(f"\n{'='*60}")
    print("Step 2: Testing direction ablation")

    top_label_ids = [l for l, _ in top_labels[:5]]
    print(f"  Ablating at: {top_label_ids}")

    skip_entries = {"img_pair_031"}
    ablation_results = []
    for prompt_data in tqdm(prompts, desc="Testing ablation"):
        if prompt_data.get("id") in skip_entries:
            continue
        harmful_image, _ = load_entry_images(prompt_data, dataset_dir)
        harmful_text = format_prompt(prompt_data["harmful"]["text"], args.model)
        baseline, ablated = test_direction_ablation(
            model, processor, tokenizer, harmful_text, harmful_image,
            args.device, args.model, refusal_directions,
            target_labels=top_label_ids,
        )
        ablation_results.append({
            "prompt_id": prompt_data["id"],
            "baseline_metric": baseline,
            "ablated_metric": ablated,
            "delta": ablated - baseline,
        })
        print(f"  {prompt_data['id']}: {baseline:.4f} -> {ablated:.4f} "
              f"(delta: {ablated - baseline:+.4f})")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    model_tag = args.model.split("/")[-1]

    output = {
        "model": args.model,
        "direction_norms": {str(k): v for k, v in direction_norms.items()},
        "top_layers": top_label_ids,
        "ablation_results": ablation_results,
    }
    with open(os.path.join(results_dir, f"refusal_direction_{model_tag}.json"), "w") as f:
        json.dump(output, f, indent=2)

    torch.save(
        {k: v for k, v in refusal_directions.items()},
        os.path.join(results_dir, f"refusal_directions_{model_tag}.pt"),
    )

    plot_refusal_directions(direction_norms, ablation_results, results_dir,
                             model_name=args.model)

    print(f"\n{'='*60}")
    print("Refusal/safety direction analysis complete.")
    print(f"  Directions saved to results/refusal_directions_{model_tag}.pt")


def main():
    parser = argparse.ArgumentParser(description="Refusal Direction Analysis for VLMs")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_prompts", type=int, default=-1)
    parser.add_argument("--flash-attn", action="store_true",
                        help="Use Flash Attention 2 (H100/H200)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for graph optimization")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    run_experiment(args)


if __name__ == "__main__":
    main()
