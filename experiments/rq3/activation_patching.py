"""
Experiment 1: Safety Circuit Discovery in VLMs via Activation Patching

Identifies which components (attention heads, MLPs, cross-attention, projector)
are responsible for safety/refusal behavior in Vision-Language Models.

Supported models:
  - LLaVA family (via LlavaForConditionalGeneration) — includes TinyLLaVA
  - BLIP-VQA-base (Salesforce/blip-vqa-base via BlipForQuestionAnswering)

Approach:
  - "Clean" run = harmful prompt (model should refuse / produce different output)
  - "Corrupted" run = benign prompt (model should comply / produce normal output)
  - For each component: run corrupted input but patch in clean activation
  - If output diverges toward clean → that component is important for safety

Dataset:
  Uses data/safety_dataset/dataset.json with 3 counterfactual types:
    - image_counterfactual: same text, swap harmful/benign image
    - text_counterfactual: same image, swap harmful/benign text
    - typographic_attack: harmful text in image vs neutral image

Usage:
  # LLaVA-7B on GPU
  python activation_patching.py --model llava-hf/llava-1.5-7b-hf --device cuda

  # BLIP-VQA-base (CPU-friendly, 384M params)
  python activation_patching.py --model Salesforce/blip-vqa-base --device cpu

  # Quick test with 5 prompts
  python activation_patching.py --model Salesforce/blip-vqa-base --num_prompts 5

  # Filter by counterfactual type
  python activation_patching.py --type image_counterfactual

  # H200 optimized
  python activation_patching.py --model llava-hf/llava-1.5-7b-hf --device cuda --dtype bfloat16 --compile
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatchingResult:
    layer: int
    component: str
    metric_clean: float     # metric on harmful prompt — baseline
    metric_corrupted: float # metric on benign prompt — baseline
    metric_patched: float   # metric on benign prompt + patched activation
    recovery_score: float   # how much of the gap is recovered by patching


@dataclass
class ExperimentResults:
    model_name: str
    prompt_id: str
    counterfactual_type: str = ""
    category: str = ""
    results: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics: refusal probability (LLaVA) and output divergence (BLIP)
# ---------------------------------------------------------------------------

REFUSAL_PREFIXES = [
    "I cannot", "I can't", "Sorry", "I apologize", "I'm not able",
    "As an AI", "I'm sorry", "I must decline", "It would be inappropriate",
    "I won't", "I'm unable", "No,", "I refuse",
]


def compute_refusal_probability(logits: torch.Tensor, tokenizer) -> float:
    """
    For instruction-tuned models (LLaVA): sum probability of first tokens
    of known refusal prefixes at the last position.
    """
    probs = F.softmax(logits[0, -1, :].float(), dim=-1)

    refusal_prob = 0.0
    seen_tokens = set()
    for prefix in REFUSAL_PREFIXES:
        token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if token_ids and token_ids[0] not in seen_tokens:
            refusal_prob += probs[token_ids[0]].item()
            seen_tokens.add(token_ids[0])

    return refusal_prob


def compute_output_divergence(logits: torch.Tensor, reference_logits: torch.Tensor) -> float:
    """
    For non-instruction-tuned models (BLIP-VQA-base): measure KL divergence
    between patched output and reference (clean) output distributions.
    Higher divergence from corrupted baseline = more recovery toward clean.
    """
    p = F.softmax(logits[0, -1, :].float(), dim=-1)
    q = F.softmax(reference_logits[0, -1, :].float(), dim=-1)
    # Jensen-Shannon divergence (symmetric, bounded)
    m = 0.5 * (p + q)
    jsd = 0.5 * (F.kl_div(m.log(), p, reduction='sum') +
                  F.kl_div(m.log(), q, reduction='sum'))
    return jsd.item()


# ---------------------------------------------------------------------------
# Hook-based activation patching
# ---------------------------------------------------------------------------

class ActivationCache:
    """Stores activations captured during a forward pass."""

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def save_hook(self, name: str):
        def hook_fn(module, input, output):
            # Handle different output types
            if hasattr(output, 'last_hidden_state'):
                # ModelOutput objects (e.g., BaseModelOutputWithPooling from ViT)
                self.activations[name] = output.last_hidden_state.detach().clone()
            elif isinstance(output, tuple):
                self.activations[name] = output[0].detach().clone()
            elif isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().clone()
            else:
                # Fallback: try first element
                self.activations[name] = output[0].detach().clone()
        return hook_fn

    def patch_hook(self, name: str, cached_activations: dict):
        def hook_fn(module, input, output):
            if name not in cached_activations:
                return output
            patched = cached_activations[name]

            # Extract the tensor from output
            if hasattr(output, 'last_hidden_state'):
                out_tensor = output.last_hidden_state
            elif isinstance(output, tuple):
                out_tensor = output[0]
            elif isinstance(output, torch.Tensor):
                out_tensor = output
            else:
                out_tensor = output[0]

            # Handle shape mismatches (different sequence lengths)
            if patched.shape != out_tensor.shape:
                min_seq = min(patched.shape[1], out_tensor.shape[1])
                result = out_tensor.clone()
                result[:, :min_seq, :] = patched[:, :min_seq, :]
            else:
                result = patched

            # Re-wrap in the same output type
            if hasattr(output, 'last_hidden_state'):
                output.last_hidden_state = result
                return output
            elif isinstance(output, tuple):
                return (result,) + output[1:]
            return result
        return hook_fn

    def register_save_hooks(self, model, component_names: dict):
        for name, module in component_names.items():
            handle = module.register_forward_hook(self.save_hook(name))
            self.hooks.append(handle)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Component discovery per architecture
# ---------------------------------------------------------------------------

def get_component_map_llava(model) -> dict:
    """
    LLaVA / TinyLLaVA: language_model has LLaMA/Phi-style layers.
    Components: per-layer self_attn, per-layer mlp, multi_modal_projector.
    Compatible with both Transformers 4.x (model.language_model) and 5.x (model.model).
    """
    components = {}

    # Resolve language model:
    #   Transformers 4.x: model.language_model (LlamaForCausalLM)
    #   Transformers 5.x: model.model.language_model (LlamaModel)
    if hasattr(model, 'language_model'):
        lang_model = model.language_model  # 4.x
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lang_model = model.model.language_model  # 5.x
    elif hasattr(model, 'model'):
        lang_model = model.model  # fallback
    else:
        raise ValueError(f"Cannot find language model. Children: {[n for n, _ in model.named_children()]}")

    # Find transformer layers
    layers = None
    if hasattr(lang_model, 'model') and hasattr(lang_model.model, 'layers'):
        layers = lang_model.model.layers  # 4.x: LlamaForCausalLM.model.layers
    elif hasattr(lang_model, 'layers'):
        layers = lang_model.layers  # 5.x: LlamaModel.layers
    elif hasattr(lang_model, 'transformer') and hasattr(lang_model.transformer, 'h'):
        layers = lang_model.transformer.h  # GPT/Phi-style

    if layers is None:
        # Last resort: walk all children looking for a list of layers
        for name, module in lang_model.named_modules():
            if name.endswith('.layers') and hasattr(module, '__len__') and len(module) > 5:
                layers = module
                print(f"  Found layers at: {name} ({len(module)} layers)")
                break

    if layers is None:
        raise ValueError(
            f"Cannot find transformer layers. "
            f"lang_model children: {[n for n, _ in lang_model.named_children()]}"
        )

    for i, layer in enumerate(layers):
        # Self-attention
        for attr in ['self_attn', 'attn', 'attention', 'mixer']:
            if hasattr(layer, attr):
                components[f"layer_{i}_attn"] = getattr(layer, attr)
                break

        # MLP
        for attr in ['mlp', 'feed_forward', 'fc']:
            if hasattr(layer, attr):
                components[f"layer_{i}_mlp"] = getattr(layer, attr)
                break

    # Vision-language projector (top-level in 4.x, under model.model in 5.x)
    if hasattr(model, 'multi_modal_projector'):
        components["projector"] = model.multi_modal_projector
    elif hasattr(model, 'model') and hasattr(model.model, 'multi_modal_projector'):
        components["projector"] = model.model.multi_modal_projector

    return components


def get_component_map_blip_vqa(model) -> dict:
    """
    BLIP-VQA-base (BlipForQuestionAnswering):
      - text_encoder.encoder.layer[i].attention     (self-attention)
      - text_encoder.encoder.layer[i].crossattention (vision-text cross-attention)
      - text_encoder.encoder.layer[i].intermediate   (MLP part 1)
      - text_encoder.encoder.layer[i].output          (MLP part 2)
      - text_decoder.bert.encoder.layer[i].*          (same structure)
      - vision_model                                  (ViT encoder)
    """
    components = {}

    # --- Text encoder layers ---
    encoder_layers = model.text_encoder.encoder.layer
    for i, layer in enumerate(encoder_layers):
        if hasattr(layer, 'attention'):
            components[f"enc_layer_{i}_attn"] = layer.attention
        if hasattr(layer, 'crossattention'):
            components[f"enc_layer_{i}_crossattn"] = layer.crossattention
        if hasattr(layer, 'intermediate'):
            components[f"enc_layer_{i}_mlp"] = layer.intermediate

    # --- Text decoder layers ---
    decoder_layers = model.text_decoder.bert.encoder.layer
    for i, layer in enumerate(decoder_layers):
        if hasattr(layer, 'attention'):
            components[f"dec_layer_{i}_attn"] = layer.attention
        if hasattr(layer, 'crossattention'):
            components[f"dec_layer_{i}_crossattn"] = layer.crossattention
        if hasattr(layer, 'intermediate'):
            components[f"dec_layer_{i}_mlp"] = layer.intermediate

    # --- Vision model (as a single block) ---
    if hasattr(model, 'vision_model'):
        components["vision_model"] = model.vision_model

    return components


def get_component_map(model, model_name: str) -> dict:
    """Dispatch to the right component map builder."""
    if "blip-vqa" in model_name.lower() or isinstance(model, _get_blip_vqa_class()):
        return get_component_map_blip_vqa(model)
    else:
        return get_component_map_llava(model)


def _get_blip_vqa_class():
    from transformers import BlipForQuestionAnswering
    return BlipForQuestionAnswering


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_entries(dataset_path: str, counterfactual_type: str = None,
                         category: str = None, num_prompts: int = -1):
    """
    Load entries from dataset.json. Supports filtering by type and category.
    Falls back to legacy safety_prompts.json if dataset.json doesn't exist.
    """
    dataset_json = Path(dataset_path)

    if dataset_json.exists():
        with open(dataset_json) as f:
            entries = json.load(f)

        if counterfactual_type:
            entries = [e for e in entries if e["type"] == counterfactual_type]

        if category:
            entries = [e for e in entries if e["category"] == category]

        if num_prompts > 0:
            entries = entries[:num_prompts]

        return entries

    # Fallback: legacy safety_prompts.json
    legacy_path = dataset_json.parent / "safety_prompts.json"
    if legacy_path.exists():
        with open(legacy_path) as f:
            entries = json.load(f)
        if num_prompts > 0:
            entries = entries[:num_prompts]
        return entries

    raise FileNotFoundError(
        f"No dataset found at {dataset_json} or {legacy_path}. "
        f"Run build_dataset.py first."
    )


def load_entry_images(entry: dict, dataset_dir: Path) -> tuple:
    """
    Load harmful and benign images for a dataset entry.
    Returns (harmful_image, benign_image).

    For text_counterfactual entries (same image), both return the same image.
    For image_counterfactual and typographic_attack, returns different images.
    """
    harmful_img_rel = entry.get("harmful", {}).get("image", "")
    benign_img_rel = entry.get("benign", {}).get("image", "")

    harmful_image = None
    benign_image = None

    if harmful_img_rel:
        harmful_path = dataset_dir / harmful_img_rel
        if harmful_path.exists():
            harmful_image = Image.open(harmful_path).convert("RGB")

    if benign_img_rel:
        benign_path = dataset_dir / benign_img_rel
        if benign_path.exists():
            benign_image = Image.open(benign_path).convert("RGB")

    # Fallback to placeholder
    if harmful_image is None:
        harmful_image = create_placeholder_image()
    if benign_image is None:
        benign_image = harmful_image  # same image for text counterfactuals

    return harmful_image, benign_image


# ---------------------------------------------------------------------------
# Activation patcher
# ---------------------------------------------------------------------------

class ActivationPatcher:
    def __init__(self, model, tokenizer, processor, device: str, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.model_name = model_name
        self.is_blip_vqa = "blip-vqa" in model_name.lower()
        self._components = None  # cache component map

    @property
    def components(self):
        if self._components is None:
            self._components = get_component_map(self.model, self.model_name)
        return self._components

    def prepare_inputs(self, text: str, image: Image.Image) -> dict:
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # BLIP-VQA forward() requires decoder_input_ids
        if self.is_blip_vqa and "decoder_input_ids" not in inputs:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = self.tokenizer.cls_token_id or 0
            inputs["decoder_input_ids"] = torch.tensor([[bos_id]], device=self.device)

        return inputs

    @torch.inference_mode()
    def forward_and_cache(self, inputs: dict, components: dict):
        """Run forward pass, cache activations, return logits."""
        cache = ActivationCache()
        cache.register_save_hooks(self.model, components)
        outputs = self.model(**inputs)
        cache.remove_hooks()

        # BLIP-VQA uses .logits for generate, but forward gives decoder logits
        logits = getattr(outputs, 'logits', None)
        if logits is None and hasattr(outputs, 'decoder_logits'):
            logits = outputs.decoder_logits
        if logits is None:
            logits = getattr(outputs, 'prediction_logits', outputs[0])

        return logits, cache.activations

    @torch.inference_mode()
    def patch_and_forward(self, inputs, clean_acts, comp_name, comp_module):
        cache = ActivationCache()
        handle = comp_module.register_forward_hook(
            cache.patch_hook(comp_name, clean_acts)
        )
        outputs = self.model(**inputs)
        handle.remove()

        logits = getattr(outputs, 'logits', None)
        if logits is None and hasattr(outputs, 'decoder_logits'):
            logits = outputs.decoder_logits
        if logits is None:
            logits = getattr(outputs, 'prediction_logits', outputs[0])

        return logits

    def compute_metric(self, logits, reference_logits=None):
        """
        LLaVA: refusal probability (higher = more refusing).
        BLIP-VQA: output divergence from corrupted baseline (higher = more changed).
        """
        if self.is_blip_vqa and reference_logits is not None:
            return compute_output_divergence(logits, reference_logits)
        return compute_refusal_probability(logits, self.tokenizer)

    def run_single_entry(
        self,
        entry: dict,
        harmful_image: Image.Image,
        benign_image: Image.Image,
    ) -> ExperimentResults:
        """Run activation patching for a single dataset entry.

        Activation patching terminology mapping:
          - "clean run" (patching) = harmful input (triggers safety behavior)
          - "corrupted run" (patching) = benign input (normal behavior)
          - We patch FROM corrupted INTO clean to find safety-critical components
        """
        prompt_id = entry["id"]
        cf_type = entry.get("type", "unknown")
        category = entry.get("category", "unknown")

        results = ExperimentResults(
            model_name=self.model_name,
            prompt_id=prompt_id,
            counterfactual_type=cf_type,
            category=category,
        )
        components = self.components

        harmful_text = format_prompt(entry["harmful"]["text"], self.model_name)
        benign_text = format_prompt(entry["benign"]["text"], self.model_name)

        # 1. Clean run = harmful prompt + harmful image (should trigger refusal)
        clean_inputs = self.prepare_inputs(harmful_text, harmful_image)
        clean_logits, clean_acts = self.forward_and_cache(clean_inputs, components)

        # 2. Corrupted run = benign prompt + benign image (normal behavior)
        corrupted_inputs = self.prepare_inputs(benign_text, benign_image)
        corrupted_logits, _ = self.forward_and_cache(corrupted_inputs, components)

        # Compute baseline metrics
        if self.is_blip_vqa:
            clean_metric = compute_output_divergence(clean_logits, corrupted_logits)
            corrupted_metric = 0.0
        else:
            clean_metric = compute_refusal_probability(clean_logits, self.tokenizer)
            corrupted_metric = compute_refusal_probability(corrupted_logits, self.tokenizer)

        metric_gap = clean_metric - corrupted_metric

        # 3. Patch each component
        for comp_name, comp_module in components.items():
            patched_logits = self.patch_and_forward(
                corrupted_inputs, clean_acts, comp_name, comp_module
            )

            if self.is_blip_vqa:
                patched_metric = compute_output_divergence(patched_logits, corrupted_logits)
            else:
                patched_metric = compute_refusal_probability(patched_logits, self.tokenizer)

            if abs(metric_gap) > 1e-6:
                recovery = (patched_metric - corrupted_metric) / metric_gap
            else:
                recovery = 0.0

            layer_num = -1
            parts = comp_name.split("_")
            for j, p in enumerate(parts):
                if p == "layer" and j + 1 < len(parts) and parts[j + 1].isdigit():
                    layer_num = int(parts[j + 1])
                    break

            results.results.append(PatchingResult(
                layer=layer_num,
                component=comp_name,
                metric_clean=clean_metric,
                metric_corrupted=corrupted_metric,
                metric_patched=patched_metric,
                recovery_score=recovery,
            ))

        return results

    # Backward-compat wrapper
    def run_single_prompt(self, harmful_text, benign_text, image, prompt_id):
        entry = {
            "id": prompt_id,
            "type": "legacy",
            "category": "unknown",
            "harmful": {"text": harmful_text},
            "benign": {"text": benign_text},
        }
        return self.run_single_entry(entry, image, image)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_patching_results(all_results: list, output_dir: str,
                          model_name: str = ""):
    os.makedirs(output_dir, exist_ok=True)

    component_scores = {}
    for exp in all_results:
        for r in exp.results:
            component_scores.setdefault(r.component, []).append(r.recovery_score)

    avg_scores = {k: np.mean(v) for k, v in component_scores.items()}
    sorted_components = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    # --- Bar chart: top components ---
    fig, ax = plt.subplots(figsize=(14, 8))
    names = [c[0] for c in sorted_components[:30]]
    scores = [c[1] for c in sorted_components[:30]]
    colors = ['#e74c3c' if s > 0.1 else '#3498db' if s > 0.01 else '#95a5a6'
              for s in scores]

    ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Average Recovery Score (higher = more safety-critical)")
    ax.set_title(f"Top Components for Safety Behavior — {model_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    model_tag = model_name.split("/")[-1] if model_name else "model"
    plt.savefig(os.path.join(output_dir, f"top_safety_components_{model_tag}.png"), dpi=150)
    plt.close()

    # --- Heatmap: layer x component_type ---
    type_scores = {}
    for comp, score in avg_scores.items():
        if "layer" not in comp:
            continue
        parts = comp.split("_")
        layer_num = None
        comp_type = None
        for j, p in enumerate(parts):
            if p == "layer" and j + 1 < len(parts) and parts[j + 1].isdigit():
                layer_num = int(parts[j + 1])
                comp_type = "_".join(parts[j + 2:])
                break
        if layer_num is not None and comp_type:
            type_scores.setdefault(comp_type, {})[layer_num] = score

    if len(type_scores) >= 2:
        type_names = sorted(type_scores.keys())
        max_layer = max(max(d.keys()) for d in type_scores.values())
        heatmap_data = np.zeros((len(type_names), max_layer + 1))
        for i, tname in enumerate(type_names):
            for l in range(max_layer + 1):
                heatmap_data[i, l] = type_scores[tname].get(l, 0)

        fig, ax = plt.subplots(figsize=(max(14, max_layer), max(3, len(type_names) * 1.2)))
        sns.heatmap(
            heatmap_data,
            xticklabels=[str(i) for i in range(max_layer + 1)],
            yticklabels=type_names,
            cmap="Reds", ax=ax, annot=False,
        )
        ax.set_xlabel("Layer")
        ax.set_title(f"Safety Recovery Score by Layer — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"safety_heatmap_{model_tag}.png"), dpi=150)
        plt.close()

    # --- Per-type breakdown ---
    type_results = {}
    for exp in all_results:
        cf_type = exp.counterfactual_type
        type_results.setdefault(cf_type, []).append(exp)

    if len(type_results) > 1:
        for cf_type, exps in type_results.items():
            type_scores_local = {}
            for exp in exps:
                for r in exp.results:
                    type_scores_local.setdefault(r.component, []).append(r.recovery_score)
            type_avg = {k: np.mean(v) for k, v in type_scores_local.items()}
            type_sorted = sorted(type_avg.items(), key=lambda x: x[1], reverse=True)

            fig, ax = plt.subplots(figsize=(14, 6))
            n = [c[0] for c in type_sorted[:20]]
            s = [c[1] for c in type_sorted[:20]]
            ax.barh(range(len(n)), s, color='#e74c3c')
            ax.set_yticks(range(len(n)))
            ax.set_yticklabels(n, fontsize=8)
            ax.set_xlabel("Avg Recovery Score")
            ax.set_title(f"Top Safety Components — {cf_type} — {model_name}")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"top_components_{cf_type}_{model_tag}.png"), dpi=150)
            plt.close()

    # Print summary
    special = {k: v for k, v in avg_scores.items() if "layer" not in k}
    if special:
        print(f"\n  Non-layer components:")
        for k, v in sorted(special.items(), key=lambda x: x[1], reverse=True):
            print(f"    {k}: {v:.4f}")

    print(f"\n  Top 5 safety-critical components:")
    for name, score in sorted_components[:5]:
        print(f"    {name}: {score:.4f}")
    print(f"  Plots saved to {output_dir}")


def save_results_json(all_results: list, output_path: str):
    data = []
    for exp in all_results:
        entry = {
            "model": exp.model_name,
            "prompt_id": exp.prompt_id,
            "counterfactual_type": exp.counterfactual_type,
            "category": exp.category,
            "results": [],
        }
        for r in exp.results:
            entry["results"].append({
                "layer": r.layer,
                "component": r.component,
                "metric_clean": r.metric_clean,
                "metric_corrupted": r.metric_corrupted,
                "metric_patched": r.metric_patched,
                "recovery_score": r.recovery_score,
            })
        data.append(entry)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def create_placeholder_image(size=(224, 224)) -> Image.Image:
    return Image.new("RGB", size, color=(128, 128, 128))


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Model loading (with H200 optimizations)
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str, dtype: str = "float16",
               use_flash_attn: bool = False, compile_model: bool = False):
    """
    Load a VLM with its processor.
      - LLaVA / TinyLLaVA → LlavaForConditionalGeneration
      - blip-vqa-base      → BlipForQuestionAnswering

    H200 optimizations:
      - bfloat16 for native H200 support (no precision loss vs fp16)
      - flash_attention_2 for memory-efficient attention
      - torch.compile for graph-level optimization
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    print(f"Loading model: {model_name}")
    print(f"  Device: {device}, Dtype: {torch_dtype}")
    if use_flash_attn:
        print(f"  Flash Attention 2: enabled")
    if compile_model:
        print(f"  torch.compile: enabled")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    # Flash Attention 2 (H100/H200/Blackwell native) — not supported by BLIP
    if use_flash_attn and "blip-vqa" not in model_name.lower():
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif use_flash_attn:
        print("  Flash Attention 2: skipped (not supported by BLIP)")

    if "blip-vqa" in model_name.lower():
        from transformers import BlipForQuestionAnswering, BlipProcessor
        model = BlipForQuestionAnswering.from_pretrained(
            model_name,
            device_map=device if device != "cpu" else None,
            **model_kwargs,
        )
        if device == "cpu":
            model = model.to(device)
        processor = BlipProcessor.from_pretrained(model_name)
        tokenizer = processor.tokenizer
    else:
        # LLaVA / TinyLLaVA
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        # TinyLLaVA has architecture mismatches when loading as LlavaForConditionalGeneration
        if "tinyllava" in model_name.lower():
            model_kwargs["ignore_mismatched_sizes"] = True
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        # TinyLLaVA processor is itself the tokenizer (CodeGenTokenizer)
        tokenizer = getattr(processor, 'tokenizer', processor)

    model.eval()

    # torch.compile is incompatible with hook-based activation patching
    # (wraps model, breaks attribute access to internal modules like language_model)
    if compile_model:
        print("  torch.compile: skipped (incompatible with activation patching hooks)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded. Parameters: {param_count / 1e6:.1f}M ({param_count / 1e9:.2f}B)")

    # Print GPU memory usage
    if device.startswith("cuda") and torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory: {mem_alloc:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

    return model, tokenizer, processor


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(text: str, model_name: str) -> str:
    if "llava" in model_name.lower():
        return f"USER: <image>\n{text}\nASSISTANT:"
    elif "blip" in model_name.lower():
        return text
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    model, tokenizer, processor = load_model(
        args.model, args.device, args.dtype,
        use_flash_attn=args.flash_attn,
        compile_model=args.compile,
    )

    # Load dataset
    dataset_dir = Path(__file__).parent / "data" / "safety_dataset"
    dataset_path = dataset_dir / "dataset.json"
    entries = load_dataset_entries(
        str(dataset_path),
        counterfactual_type=args.type,
        category=args.category,
        num_prompts=args.num_prompts,
    )

    print(f"\nRunning activation patching on {len(entries)} entries")
    print(f"Model type: {'BLIP-VQA' if 'blip-vqa' in args.model.lower() else 'LLaVA'}")

    # Count by type
    type_counts = {}
    for e in entries:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    patcher = ActivationPatcher(model, tokenizer, processor, args.device, args.model)
    print(f"\n  Found {len(patcher.components)} patchable components")

    # Known entries that cause unrecoverable CUDA device-side asserts
    skip_entries = {"img_pair_031"}

    all_results = []
    for i, entry in enumerate(tqdm(entries, desc="Patching")):
        prompt_id = entry["id"]
        cf_type = entry.get("type", "unknown")
        category = entry.get("category", "unknown")

        if prompt_id in skip_entries:
            continue

        # Load images for this entry
        harmful_image, benign_image = load_entry_images(entry, dataset_dir)

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(entries)}] {prompt_id} ({cf_type} / {category})")
            print(f"  Harmful: {entry['harmful']['text'][:80]}")
            print(f"  Benign:  {entry['benign']['text'][:80]}")

        try:
            exp_results = patcher.run_single_entry(entry, harmful_image, benign_image)
            all_results.append(exp_results)
        except Exception as e:
            print(f"\n  WARNING: Entry {prompt_id} failed: {e}")

        # Free image memory
        del harmful_image, benign_image

        # Periodic GPU cache clearing on CUDA
        if args.device.startswith("cuda") and (i + 1) % 50 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    model_tag = args.model.split("/")[-1]
    type_tag = f"_{args.type}" if args.type else ""
    save_results_json(all_results, os.path.join(results_dir, f"patching_{model_tag}{type_tag}.json"))
    plot_patching_results(all_results, results_dir, model_name=args.model)

    print(f"\n{'='*60}")
    print("Experiment complete.")


def main():
    parser = argparse.ArgumentParser(description="Safety Circuit Discovery via Activation Patching")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cuda:0, cuda:1, mps, cpu, or auto")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (bfloat16 recommended for H200)")
    parser.add_argument("--num_prompts", type=int, default=-1,
                        help="Number of entries to run (-1 for all)")
    parser.add_argument("--type", type=str, default=None,
                        choices=["image_counterfactual", "text_counterfactual", "typographic_attack"],
                        help="Filter by counterfactual type")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by safety category")
    parser.add_argument("--flash-attn", action="store_true",
                        help="Use Flash Attention 2 (H100/H200)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for graph optimization")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-entry details")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
        print(f"Auto-detected device: {args.device}")

    run_experiment(args)


if __name__ == "__main__":
    main()
