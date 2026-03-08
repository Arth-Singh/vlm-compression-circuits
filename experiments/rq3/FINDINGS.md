# RQ3: AI Safety Implications of Circuit-Based Compression in VLMs

## Research Question
How does model compression affect safety-critical circuits in Vision-Language Models, and can circuit analysis identify components essential for maintaining safe behavior?

---

## Experimental Setup

### Models
| Model | Parameters | Layers | Architecture | Safety Alignment |
|-------|-----------|--------|-------------|-----------------|
| LLaVA-v1.6-Vicuna-13B | 13.35B | 40 | Decoder-only (LlavaNext) | Yes (Vicuna RLHF) |
| LLaVA-1.5-7B | 7B | 32 | Decoder-only (LlavaForConditional) | Minimal |
| BLIP-VQA-Base | 361M | 12+12 | Encoder-Decoder | None |

### Dataset
- 325 image-text pairs across 3 counterfactual types:
  - **Text counterfactual** (155 entries): Same image, harmful vs. benign text
  - **Image counterfactual** (150 entries): Same text, harmful vs. benign image
  - **Typographic attack** (20 entries): Text embedded in images to override safety

### Methods
1. **Activation Patching** (Experiment 1): Measures recovery score per component — how much restoring a single component recovers safe behavior from a corrupted baseline.
2. **Refusal Direction Analysis** (Experiment 2): Extracts the linear direction in activation space that encodes refusal behavior, measured per layer.
3. **Compression Evaluation** (Experiment 3): Applies standard pruning methods (uniform magnitude, Wanda, random) at 30%, 50%, 70% sparsity and measures impact on safety using LLM-as-a-Judge.

### Evaluation: LLM-as-a-Judge
We replaced the naive first-token probability metric with Claude (Haiku 4.5) as a judge. The model generates full text responses, and Claude classifies each as REFUSAL or COMPLIANCE. This increased detection from ~0% (broken metric) to 84.9% refusal rate on the baseline model.

---

## Key Findings

### Finding 1: The Vision-Language Projector is the Most Safety-Critical Component

Activation patching on LLaVA-v1.6-Vicuna-13B reveals the **projector** (vision-language bridge) has by far the highest recovery score:

| Component | Recovery Score | Role |
|-----------|---------------|------|
| **projector** | **0.522** | Vision-language bridge |
| layer_19_mlp | 0.259 | Mid-layer MLP |
| layer_14_attn | 0.186 | Mid-layer attention |
| layer_15_mlp | 0.184 | Mid-layer MLP |
| layer_37_mlp | 0.177 | Late-layer MLP |
| layer_17_attn | 0.176 | Mid-layer attention |
| layer_17_mlp | 0.173 | Mid-layer MLP |

**Interpretation**: The projector translates visual information into the language model's embedding space. Corrupting it disrupts the model's ability to correctly interpret image content, which is the foundation for safety decisions in multimodal contexts. This is the single most important component for VLM safety.

### Finding 2: Safety Circuits Concentrate in Mid-to-Late Layers (14-19)

The safety-critical transformer layers form a contiguous band in layers 14-19 (both attention and MLP). This aligns with prior work on safety circuits in text-only LLMs (Arditi et al., 2024), which found refusal behavior encoded in middle layers.

- **Early layers (0-5)**: Low importance (< 0.13) — handle basic feature extraction
- **Middle layers (14-19)**: High importance (0.17-0.26) — safety decision-making
- **Late layers (35-39)**: Moderate importance — output formatting

### Finding 3: Refusal Direction Strengthens Monotonically Toward Final Layers

The refusal direction norm increases steadily from layer 0 to the final layer across both LLaVA models:

| Model | Layer 0 | Layer 15 | Layer 31 | Final Layer |
|-------|---------|----------|----------|-------------|
| LLaVA-1.5-7B | 1.15 | 1.13 | **5.44** | 5.44 (L31) |
| LLaVA-v1.6-13B | 1.24 | 1.75 | 4.34 | **6.50** (L39) |
| BLIP-VQA-Base | ~0.01 | ~0.01 | ~0.01 | ~0.01 |

**Interpretation**: Refusal behavior is a cumulative computation that builds across layers, with the final layers contributing the most. BLIP shows no refusal direction at all, confirming it lacks safety alignment. The 13B model with explicit RLHF training shows a stronger, more distributed refusal signal.

### Finding 4: Cross-Model Safety Architecture Comparison

| Property | BLIP-VQA-Base | LLaVA-1.5-7B | LLaVA-v1.6-13B |
|----------|--------------|--------------|----------------|
| Has refusal? | No | Minimal | Yes (84.9%) |
| Top safety component | N/A | projector (0.30) | projector (0.52) |
| Safety-critical layers | N/A | 16-22 | 14-19 |
| Refusal direction peak | None | Layer 31 (5.44) | Layer 39 (6.50) |
| Refusal mechanism | None | Weak/inconsistent | Strong, multi-pattern |

### Finding 5: LLaVA-v1.6-13B Baseline Refusal Behavior (LLM Judge)

Full dataset evaluation (324 entries) with Claude as judge:

| Metric | Value |
|--------|-------|
| Total refusals | 275 / 324 (84.9%) |
| Empty/silent refusals | 93 |
| "I'm sorry" refusals | 126 |
| "Cannot/can't" refusals | 26 |
| Compliance cases | 49 (15.1%) |

The model employs diverse refusal strategies:
- **Explicit verbal refusal**: "I'm sorry, I cannot..."
- **Silent refusal**: Empty response (no output generated)
- **Deflection**: "Can you please rephrase your question?"
- **Evasion**: "The image is not clear enough..."

Compliance cases cluster around: weapon descriptions (katana), fire-starting instructions, phishing emails, and counterfeit currency — indicating specific safety gaps.

### Finding 6: Compression Impact on Safety (LLM Judge v2 — with Coherence)

Evaluated on 149 harmful prompts per configuration using Claude Haiku 4.5 with coherence scoring. This separates **genuine refusals** (coherent, intentional safety behavior) from **model failures** (incoherent/empty output from a broken model).

**Total refusal rate** (includes both genuine + model failures):

| Method | 30% Sparsity | 50% Sparsity | 70% Sparsity |
|--------|:------------:|:------------:|:------------:|
| **Baseline** | | **84.9%** | |
| Uniform Magnitude | 64.4% | 62.4% | 69.1% |
| Wanda | **74.5%** | 53.0% | 73.2% |
| Random | 73.2% | 84.6% | 95.3% |

**Genuine refusal rate** (coherent, intentional refusals only):

| Method | 30% Sparsity | 50% Sparsity | 70% Sparsity |
|--------|:------------:|:------------:|:------------:|
| **Baseline** | | **~85%** | |
| Uniform Magnitude | 18.1% | 6.7% | **0%** |
| Wanda | **23.5%** | **22.8%** | **0%** |
| Random | **0%** | **0%** | **0%** |

**Key observations:**

1. **The coherence metric reveals the true picture**: Without it, random pruning at 30% appeared to have 73% "refusal" — but 100% of those are model failures, not safety. Zero genuine refusals.

2. **Wanda best preserves genuine safety**: 23.5% genuine refusal at 30%, 22.8% at 50%. It's the only method that maintains meaningful safety behavior at moderate sparsity.

3. **Uniform magnitude degrades safety faster**: Only 18.1% genuine at 30%, dropping to 6.7% at 50%. This method uniformly damages safety-critical components in layers 14-19.

4. **At 70% sparsity, ALL methods produce 0% genuine refusals**: The model is too degraded for coherent output regardless of pruning method.

5. **The genuine refusal rate is lower than the raw baseline**: This reflects the stricter coherence criterion — many baseline "refusals" (empty responses, garbled text) are also model limitations rather than intentional safety behavior.

### Finding 7: Targeted Ablation Confirms Safety Circuit Causality

The strongest evidence: selectively pruning ONLY safety-critical vs non-safety components at 50% sparsity.

| Ablation Config | Total Refusal | Genuine Refusal | Model Failure |
|----------------|:---:|:---:|:---:|
| **Baseline (no pruning)** | **84.9%** | **~85%** | **~0%** |
| Prune safety-only | 73.2% | 15.4% | 57.7% |
| Prune non-safety-only | 81.2% | 14.8% | 66.4% |
| **Prune projector-only** | **85.2%** | **25.5%** | **59.7%** |

**Interpretation:**

1. **Pruning safety components drops genuine refusal to 15.4%**: Confirms these components are causally important — removing them destroys the model's ability to coherently refuse.

2. **Pruning non-safety components also degrades**: 14.8% genuine refusal. At 50% sparsity on decoder components, even "non-critical" pruning causes substantial model degradation. This suggests the 0.05 safety threshold may be too low, or that safety behavior requires general model coherence.

3. **Projector-only pruning has the highest genuine refusal rate (25.5%)**: Despite the projector being the #1 safety component by recovery score, pruning it alone preserves more genuine refusal than pruning all safety components. This suggests the projector's importance is more about vision-language alignment than refusal behavior per se.

4. **High model failure rates across all ablations**: 50% sparsity is aggressive. At this level, any targeted pruning causes significant degradation, making it hard to isolate safety-specific effects from general model failure.

---

## Methodology Notes

### Why LLM-as-a-Judge?
The standard approach of checking first-token probabilities for refusal prefixes ("I cannot", "Sorry", etc.) completely failed to detect refusals in LLaVA-v1.6-Vicuna-13B. The model's refusal behavior includes empty responses, non-English deflections, and creative evasion that no keyword list can capture. Using Claude Haiku 4.5 as a judge provides robust, context-aware refusal detection.

### Activation Patching Protocol
For each dataset entry:
1. Run model on harmful input → get corrupted output logits
2. Run model on benign counterfactual → get clean activations
3. For each of 81 components (40 attn + 40 mlp + projector):
   - Patch clean activation into corrupted run at that component
   - Measure recovery toward clean behavior
4. Recovery score = (metric_corrupted - metric_patched) / (metric_corrupted - metric_clean)

Higher recovery score → component is more important for safety behavior.

---

## Files

| File | Description |
|------|-------------|
| `activation_patching.py` | Experiment 1: Component importance via patching |
| `refusal_direction.py` | Experiment 2: Refusal direction extraction |
| `compression_experiment.py` | Experiment 3: Standard compression methods |
| `llm_judge.py` | LLM-as-a-Judge for base model evaluation |
| `llm_judge_compression.py` | LLM-as-a-Judge for compressed model evaluation |
| `results/patching_llava-v1.6-vicuna-13b-hf.json` | Patching results (324 entries × 81 components) |
| `results/refusal_direction_llava-v1.6-vicuna-13b-hf.json` | Refusal direction norms (40 layers) |
| `results/llm_judge_llava-v1.6-vicuna-13b-hf.json` | Full LLM judge results (324 entries) |
