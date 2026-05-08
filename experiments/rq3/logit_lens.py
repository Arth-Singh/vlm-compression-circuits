"""
Experiment 4: Logit Lens Safety Analysis

Projects intermediate residual stream representations through the unembedding
matrix to verify that safety-critical layers encode refusal information.

Following Neo et al. (ICLR 2025) who applied logit lens to LLaVA for visual
token analysis, but adapted for safety/refusal analysis — a novel application.

Method:
  For each harmful/benign input pair:
  1. Forward pass with output_hidden_states=True
  2. At each layer l, project residual stream through final norm + LM head
  3. Measure refusal token probability mass at the last input position
  4. Compare harmful vs benign trajectories across layers

The divergence point (where harmful >> benign) identifies where safety
encoding begins, providing convergent evidence with activation patching.

Technical note (Belrose et al., 2023): The final RMSNorm MUST be applied
before the unembedding projection. Residual stream norms grow exponentially
across layers — without normalization, logits are meaningless.

Usage:
  python logit_lens.py --model llava-hf/llava-1.5-7b-hf --device cuda
  python logit_lens.py --model llava-hf/llava-v1.6-vicuna-13b-hf --device cuda --max-entries 50
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from activation_patching import (
    REFUSAL_STARTERS,
    COMPLIANCE_STARTERS,
    _get_token_ids,
    create_placeholder_image,
    format_prompt,
    load_dataset_entries,
    load_entry_images,
    load_model,
)

SKIP_ENTRIES = {"img_pair_031"}


@torch.inference_mode()
def logit_lens_single(
    model, processor, tokenizer, image, prompt, device, model_name,
    refusal_ids, compliance_ids,
):
    """
    Run logit lens on a single input. Returns per-layer metrics.

    For each layer l:
      h_l = hidden_states[l+1]  (residual stream after layer l)
      h_normed = final_norm(h_l)
      logits_l = lm_head(h_normed)
      → compute refusal/compliance probability mass and logit difference
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states  # tuple of (n_layers + 1) tensors

    # Get the final norm and LM head
    # Transformers 5.x: model.model.language_model.norm + model.lm_head
    # Transformers 4.x: model.language_model.model.norm + model.language_model.lm_head
    final_norm = None
    lm_head = None

    # Try transformers 5.x paths first
    for norm_path, head_path in [
        ("model.language_model.norm", "lm_head"),           # transformers 5.x
        ("language_model.model.norm", "language_model.lm_head"),  # transformers 4.x
        ("language_model.norm", "lm_head"),                 # alternative 5.x
        ("model.language_model.model.norm", "lm_head"),     # alternative
    ]:
        try:
            obj = model
            for part in norm_path.split("."):
                obj = getattr(obj, part)
            final_norm = obj

            obj = model
            for part in head_path.split("."):
                obj = getattr(obj, part)
            lm_head = obj
            break
        except AttributeError:
            final_norm = None
            lm_head = None

    if final_norm is None or lm_head is None:
        raise RuntimeError(f"Cannot find norm/lm_head. Model type: {type(model).__name__}")

    n_layers = len(hidden_states) - 1  # subtract embedding layer
    layer_results = []

    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1]  # +1 because index 0 is embedding output

        # Apply final RMSNorm — REQUIRED before unembedding (Belrose et al., 2023)
        h_normed = final_norm(h)

        # Project through LM head to get logits over vocabulary
        logits = lm_head(h_normed)  # (batch, seq_len, vocab_size)

        # Last input position — predicts first output token
        last_logits = logits[0, -1, :].float()
        probs = F.softmax(last_logits, dim=-1)

        # Refusal token probability mass
        refusal_prob = probs[refusal_ids].sum().item() if refusal_ids else 0.0
        compliance_prob = probs[compliance_ids].sum().item() if compliance_ids else 0.0

        # Logit difference (same metric as activation patching)
        refusal_logit_mean = last_logits[refusal_ids].mean().item() if refusal_ids else 0.0
        compliance_logit_mean = last_logits[compliance_ids].mean().item() if compliance_ids else 0.0
        logit_diff = refusal_logit_mean - compliance_logit_mean

        # Top-5 tokens at this layer
        top_probs, top_ids = torch.topk(probs, 5)
        top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())

        layer_results.append({
            "layer": layer_idx,
            "refusal_prob": refusal_prob,
            "compliance_prob": compliance_prob,
            "logit_difference": logit_diff,
            "top_tokens": top_tokens,
            "top_probs": [round(p, 4) for p in top_probs.tolist()],
        })

    return layer_results


def run_logit_lens(args):
    model, tokenizer, processor = load_model(
        args.model, args.device, args.dtype,
        use_flash_attn=args.flash_attn,
        compile_model=False,
    )

    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    dataset_path = dataset_dir / "dataset.json"
    entries = load_dataset_entries(str(dataset_path), num_prompts=args.max_entries)

    model_tag = args.model.split("/")[-1]
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Precompute token IDs
    refusal_ids = _get_token_ids(tokenizer, REFUSAL_STARTERS, "refusal")
    compliance_ids = _get_token_ids(tokenizer, COMPLIANCE_STARTERS, "compliance")

    print(f"\nLogit Lens Safety Analysis")
    print(f"  Model: {args.model}")
    print(f"  Entries: {len(entries)}")
    print(f"  Refusal token IDs ({len(refusal_ids)}): {refusal_ids}")
    print(f"  Compliance token IDs ({len(compliance_ids)}): {compliance_ids}")

    # Per-entry results
    all_entries = []
    # Aggregated per-layer stats
    harmful_layers = None  # will be [n_layers] arrays
    benign_layers = None

    for entry in tqdm(entries, desc="Logit lens"):
        if entry.get("id") in SKIP_ENTRIES:
            continue

        try:
            harmful_image, benign_image = load_entry_images(entry, dataset_dir)
            harmful_text = format_prompt(entry["harmful"]["text"], args.model)
            benign_text = format_prompt(entry["benign"]["text"], args.model)

            # Harmful input
            harmful_results = logit_lens_single(
                model, processor, tokenizer, harmful_image, harmful_text,
                args.device, args.model, refusal_ids, compliance_ids,
            )

            # Benign input
            benign_results = logit_lens_single(
                model, processor, tokenizer, benign_image, benign_text,
                args.device, args.model, refusal_ids, compliance_ids,
            )

            n_layers = len(harmful_results)
            if harmful_layers is None:
                harmful_layers = {
                    "refusal_prob": [[] for _ in range(n_layers)],
                    "logit_diff": [[] for _ in range(n_layers)],
                }
                benign_layers = {
                    "refusal_prob": [[] for _ in range(n_layers)],
                    "logit_diff": [[] for _ in range(n_layers)],
                }

            for i in range(n_layers):
                harmful_layers["refusal_prob"][i].append(harmful_results[i]["refusal_prob"])
                harmful_layers["logit_diff"][i].append(harmful_results[i]["logit_difference"])
                benign_layers["refusal_prob"][i].append(benign_results[i]["refusal_prob"])
                benign_layers["logit_diff"][i].append(benign_results[i]["logit_difference"])

            all_entries.append({
                "id": entry.get("id", "?"),
                "type": entry.get("type", "?"),
                "category": entry.get("category", "?"),
                "harmful_layers": harmful_results,
                "benign_layers": benign_results,
            })

        except Exception as e:
            print(f"  WARNING: Failed for {entry.get('id', '?')}: {e}")

    # Compute aggregated statistics
    n_layers = len(harmful_layers["refusal_prob"]) if harmful_layers else 0
    layer_summary = []
    for i in range(n_layers):
        h_ref_prob = np.mean(harmful_layers["refusal_prob"][i])
        b_ref_prob = np.mean(benign_layers["refusal_prob"][i])
        h_ld = np.mean(harmful_layers["logit_diff"][i])
        b_ld = np.mean(benign_layers["logit_diff"][i])

        layer_summary.append({
            "layer": i,
            "harmful_refusal_prob": round(h_ref_prob, 6),
            "benign_refusal_prob": round(b_ref_prob, 6),
            "refusal_prob_gap": round(h_ref_prob - b_ref_prob, 6),
            "harmful_logit_diff": round(h_ld, 4),
            "benign_logit_diff": round(b_ld, 4),
            "logit_diff_gap": round(h_ld - b_ld, 4),
        })

    # Find divergence point — layer where gap first exceeds 2× the mean gap of layers 0-5
    early_gaps = [s["refusal_prob_gap"] for s in layer_summary[:6]]
    early_mean = np.mean(early_gaps) if early_gaps else 0
    early_std = np.std(early_gaps) if early_gaps else 0.001
    divergence_threshold = early_mean + 2 * early_std

    divergence_layer = None
    for s in layer_summary:
        if s["refusal_prob_gap"] > divergence_threshold and s["layer"] > 5:
            divergence_layer = s["layer"]
            break

    # Print summary
    print(f"\n{'='*60}")
    print(f"LOGIT LENS SUMMARY — {args.model}")
    print(f"{'='*60}")
    print(f"  Entries analyzed: {len(all_entries)}")
    print(f"  Layers: {n_layers}")
    print(f"  Divergence layer: {divergence_layer}")
    print(f"\n  {'Layer':>5} | {'H.RefProb':>9} | {'B.RefProb':>9} | {'Gap':>8} | {'H.LD':>8} | {'B.LD':>8}")
    print(f"  {'-'*55}")
    for s in layer_summary:
        marker = " <<<" if s["layer"] == divergence_layer else ""
        print(f"  {s['layer']:>5} | {s['harmful_refusal_prob']:>9.4f} | "
              f"{s['benign_refusal_prob']:>9.4f} | {s['refusal_prob_gap']:>8.4f} | "
              f"{s['harmful_logit_diff']:>8.2f} | {s['benign_logit_diff']:>8.2f}{marker}")

    # Save results
    output = {
        "model": args.model,
        "n_entries": len(all_entries),
        "n_layers": n_layers,
        "divergence_layer": divergence_layer,
        "divergence_threshold": round(divergence_threshold, 6),
        "refusal_token_ids": refusal_ids,
        "compliance_token_ids": compliance_ids,
        "layer_summary": layer_summary,
    }

    # Save full per-entry data only if requested (large)
    if args.save_full:
        output["entries"] = all_entries

    output_path = results_dir / f"logit_lens_{model_tag}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Logit Lens Safety Analysis")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-entries", type=int, default=150)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--save-full", action="store_true",
                        help="Save per-entry layer data (large)")

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_logit_lens(args)


if __name__ == "__main__":
    main()
