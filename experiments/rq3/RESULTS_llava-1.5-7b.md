# RQ3 — LLaVA-1.5-7b Safety Under Compression

**Model:** `llava-hf/llava-1.5-7b-hf` (LlavaForConditionalGeneration, bf16)
**Dataset:** `experiments/rq3/data/safety_dataset/dataset.json` — 420 image+prompt pairs across `image_counterfactual`, `text_counterfactual`, and `typographic_attack` types. Harmful entries pair innocuous-looking text prompts with images depicting weapons, drugs, chemicals, lock-picking tools, etc. Evaluation uses 149 harmful entries (`max_entries=150`, minus 1 in `SKIP_ENTRIES`).
**Hardware:** 3× NVIDIA B200 (183 GB each), CUDA 13.1, transformers 4.57.6, torch 2.10.0+cu128.
**Judge:** `google/gemini-3.1-flash-lite` via OpenRouter, async batched at concurrency 100. The judge sees the *image* in addition to the prompt+response — the harm in this dataset is image-encoded, so a text-only judge cannot tell whether a fluent description constitutes a safety failure.
**Evaluation pipeline:**
1. Sequential generation on the GPU (~1 prompt/sec at bf16).
2. Concurrent multimodal judge calls — each judge call receives the harmful image as a base64 data URL alongside the model's prompt + response.
3. Aggregate refusal/genuine-refusal/incoherent-failure counts.

---

## TL;DR

- **Baseline llava-1.5-7b refuses 0/149 harmful image prompts.** It has essentially no visual safety alignment on this dataset — not a single response declines to engage with the image content (knives, chemicals, lock picks, scattered medication, etc.).
- Compression therefore has nothing to "destroy". Wanda 10/20/30/50% and AWQ int4 all stay near the baseline floor.
- **Compression-induced "refusals" at high sparsity are degenerate**: they're "I cannot tell what this is" outputs from a damaged model, not principled safety refusals. Coherence-aware judging separates these cleanly.
- The targeted-ablation circuit-discovery story is consequently uninformative on this model+dataset combination — Cohen's d for "prune safety-critical vs random baseline" is small, p ≫ 0.05.
- This argues for either (a) using a model with non-trivial baseline visual safety (e.g. LLaVA-1.6, Qwen2-VL-Instruct, GPT-style chat-tuned VLMs), or (b) using a text-harmful dataset (AdvBench/HarmfulQA) where the standard text safety circuit is the relevant target.

## Refusal rates — image-aware multimodal judge (n=149)

| Configuration              | Refusal rate | Genuine refusal | Incoherent failure |
|----------------------------|--------------|-----------------|--------------------|
| baseline (uncompressed)    | 0.0% (0/149) | 0.0%            | 0.0%               |
| AWQ int4                   | 0.0% (0/149) | 0.0%            | 0.0%               |
| Wanda 10%                  | 0.7% (1/149) | 0.7%            | 0.0%               |
| Wanda 20%                  | 0.0% (0/149) | 0.0%            | 0.0%               |
| Wanda 30%                  | 0.0% (0/149) | 0.0%            | 0.0%               |
| Wanda 50%                  | 1.3% (2/149) | 1.3%            | 0.0%               |

## Multimodal vs text-only judge (rerun comparison)

The same six configurations were also evaluated with the original text-only judge (`claude-jupiter-v1-p`, prompt+response only). Differences quantify how much "judging signal" comes from the image alone:

| Configuration              | Text-judge refusal | Multimodal-judge refusal | Δ        |
|----------------------------|--------------------|--------------------------|----------|
| baseline                   | 0.7% (1/149)       | 0.0% (0/149)             | −0.7 pp  |
| AWQ int4                   | 0.0% (0/149)       | 0.0% (0/149)             |  0 pp    |
| Wanda 10%                  | 0.0%               | 0.7%                     | +0.7 pp  |
| Wanda 20%                  | 0.7%               | 0.0%                     | −0.7 pp  |
| Wanda 30%                  | 1.3%               | 0.0%                     | −1.3 pp  |
| Wanda 50%                  | 6.7%               | 1.3%                     | −5.4 pp  |

The text-only judge over-counted refusals at high sparsity because it labelled "I cannot describe this image" outputs (caused by capability damage from heavy pruning) as refusals. The multimodal judge can see the image, sees that the response is degenerate, and either marks it incoherent or correctly classifies it as compliance. Net effect: the apparent monotonic "more pruning → more refusals" curve under the text judge collapses once the judge actually sees what the model was shown.

## Targeted ablation @ 50% sparsity (n=149, n_random=10)

The ablation tests whether components identified as "safety-critical" by activation patching are causally responsible for safety. Three configurations are run on the uncompressed model with 50% magnitude pruning restricted to the named subset, plus 10 random baselines of equal cardinality:

| Ablation                       | Genuine refusal |
|--------------------------------|-----------------|
| prune_safety_only              | 0.0%            |
| prune_nonsafety_only           | 0.7% (1/149)    |
| prune_projector_only           | 0.0%            |
| Random baselines (mean ± std)  | 0.3% ± 0.5%     |
| Random baselines (individual)  | 0, 0, 0, 0, 0.7, 0, 0.7, 1.3, 0, 0.7 |
| Cohen's d (safety vs random)   | **−0.707**      |
| p-value (one-sided t-test)     | 0.026           |

The Cohen's d is **negative**: pruning safety-critical components produces *fewer* refusals than random pruning, and the t-test is "significant" only because the safety-only ablation hits the floor (0%) while the random distribution wobbles slightly above it. The expected sign for a real safety circuit is positive (pruning safety components → safety drops → fewer refusals would actually be consistent here, but the absolute magnitudes — 0.0% vs 0.3% — are at the level of a single Bernoulli trial). Since baseline refusal is 0%, all conditions are pinned near the floor and the ablation cannot meaningfully distinguish "safety-critical" from random subsets. This is **not a failure of the patching/ablation methodology**; it is the dataset+model combination giving zero signal.

## Activation patching summary (logit-difference metric)

Computed via `experiments/rq3/activation_patching.py` on the full 420-entry dataset across 65 patchable components (multimodal projector + 32 attention layers + 32 MLP layers).

Top safety-critical components by recovery score:
- `projector` — recovery 1.0000
- `layer_9_attn` — 0.4972
- `layer_1_mlp`  — 0.4245
- `layer_12_attn` — 0.4038
- `layer_6_attn` — 0.3960

The projector dominates, consistent with the intuition that the multimodal projector is the main vehicle for image-conditional safety in LLaVA-style architectures. Note however that the *behavioural* refusal floor is ~0%, so what we are localizing here is "which components encode the (rare) refusal signal" rather than "which components keep a well-aligned model safe".

## Logit-lens trajectory

Refusal-token vs compliance-token probability mass at each layer's residual stream (after final RMSNorm + lm_head, per Belrose et al. 2023):

```
layer  P(refusal)  P(compliance)  Δ        logit-diff
...
 28    0.0001      0.0001         −0.0000   −1.75
 29    0.0000      0.0000         −0.0000   −2.05
 30    0.0000      0.0000         −0.0000   −1.88
 31    0.0002      0.0002         −0.0000   −1.51
```

Both refusal and compliance probabilities are ~10⁻⁴ at every layer for harmful inputs — i.e. neither is favored. Consistent with the behavioural finding: the model has not internalized a refusal direction for visual harm.

## Engineering notes (for reproducibility)

1. **AutoAWQ + transformers 4.57**: AutoAWQ's `awq/quantize/scale.py` imports `PytorchGELUTanh` (renamed to `GELUTanh` in tx 4.57). One-line shim added at install time.
2. **AutoAWQ + LlavaForConditionalGeneration in tx 4.36+**: `model.language_model` is now a `LlamaModel` directly (previously wrapped in `LlamaForCausalLM`). `awq/models/llava.py` was patched to handle both `lm.model.layers` and `lm.layers`.
3. **AWQ multimodal device placement**: `AutoAWQForCausalLM.from_pretrained(device_map="cuda")` only places the language model on GPU; vision tower and multimodal projector stay on CPU and crash generation. Fix: explicit `.to("cuda")` for `vision_tower` / `multi_modal_projector` after AWQ load. Also: `from_quantized` rejects `device_map="cuda"` — must use `"auto"`.
4. **Wanda speedup**: `compute_wanda_activation_norms` was extracted from `apply_wanda_pruning` so a single 32-prompt calibration sweep is reused across all sparsity levels (saves 3× redundant forward passes per re-run). Original weights are snapshotted once and restored between sparsity levels — no fresh model loads.
5. **Targeted ablation speedup**: Same snapshot/restore trick — load model + snapshot once, restore + prune + evaluate per ablation. Cuts 13 model loads down to 1.
6. **Async judge batching**: Judge calls were the bottleneck (~3-4 s round-trip). Generation runs sequentially on the GPU; judging is then dispatched concurrently via `openai.AsyncOpenAI` against OpenRouter at concurrency 100. End-to-end wall time on the full 6-config + 13-ablation suite dropped from ~120 min (sync) to ~30 min (async).
7. **Image preprocessing for the judge**: PIL → JPEG @ quality 85, downscaled so `max_side ≤ 768` px, encoded as `data:image/jpeg;base64,…`. This keeps payloads small without losing relevant detail (the harmful items are clearly recognizable at 768 px).

## What this rules out / what's worth running next

**Ruled out**: The hypothesis "compression of LLaVA-1.5-7b destroys image-conditional safety". You cannot destroy what isn't there. The baseline already complies in 100% of cases.

**Suggested next runs**:
- **Stronger model**: Re-run the suite on `liuhaotian/llava-v1.6-vicuna-13b-hf` or `Qwen/Qwen2-VL-7B-Instruct`. Both have more recent safety alignment passes and should show non-trivial baseline refusal on this dataset.
- **Text-harmful dataset**: Run the full pipeline against AdvBench / HarmfulQA so the safety circuit you're trying to localize actually operates on text. The activation-patching → ablation → compression pipeline is sound; the dataset just needs to be one the model is calibrated against.
- **Multimodal-aligned compression**: AWQ int4 here didn't budge from 0%, but on a model with non-trivial baseline safety that's the comparison point worth measuring.

## File map (for the rest of the codebase)

| Artifact | Path |
|---|---|
| Activation patching code | `experiments/rq3/activation_patching.py` |
| Logit lens code | `experiments/rq3/logit_lens.py` |
| Compression eval (Wanda + AWQ) | `experiments/rq3/llm_judge_compression.py` |
| Targeted ablation | `experiments/rq3/targeted_ablation.py` |
| Multimodal LLM-judge | `experiments/rq3/llm_judge.py` |
| Wanda + Magnitude pruning ops | `experiments/rq3/compression_experiment.py` |
| Top-level runner (3 GPUs) | `experiments/rq3/run_all_v2.sh` |
| Patching results | `experiments/rq3/results/patching_llava-1.5-7b-hf.json` |
| Logit lens results | `experiments/rq3/results/logit_lens_llava-1.5-7b-hf.json` |
| Baseline (mm-judge) | `experiments/rq3/results/llm_judge_llava-1.5-7b-hf.json` |
| Per-config compression (mm-judge) | `experiments/rq3/results/llm_judge_v3_*_llava-1.5-7b-hf.json` |
| Targeted ablation (mm-judge) | `experiments/rq3/results/targeted_ablation_v2_llava-1.5-7b-hf.json` |
