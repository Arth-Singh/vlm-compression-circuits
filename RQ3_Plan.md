# RQ3: What Are the Implications of Our Method for AI Safety?

---

## 1. The Big Picture (Full Project Context)

### What we're building
A **task-specific compression method for VLMs** using circuit analysis (mechanistic interpretability). Instead of blindly compressing the whole model (GPTQ, AWQ, Wanda), we:

1. **RQ1**: Discover which internal circuits (attention heads, MLPs, projector components) are necessary and sufficient for specific tasks (Image Captioning, VQA) in VLMs. Then study how standard compression methods damage these circuits.
2. **RQ2**: Use those findings to build a compression method that selectively removes only components OUTSIDE the task-critical circuit — yielding smaller models that perform as well or better on the target task.
3. **RQ3**: Show that this approach also **preserves safety alignment** better than blind compression, and analyze the implications for AI safety.

### Why RQ3 matters
Safety alignment in models depends on sparse, brittle mechanisms — refusal heads, safety-relevant MLPs, specific directions in the residual stream. Blind compression (pruning/quantization) can accidentally destroy these, creating models that still perform well on benchmarks but have lost their safety guardrails. Our task-specific approach, by being selective about what to remove, has the potential to preserve these safety mechanisms — or can be explicitly designed to include them.

### Models
The team agreed on **two model families**:
- **LLaVA** (linear projector architecture) — start with TinyLLaVA for RQ1 prototyping, scale to LLaVA-7B/13B
- **BLIP-2** (cross-attention / Q-Former architecture) — different modality bridging mechanism

**Note on model choice for RQ3**: For safety experiments specifically, we should prioritize **LLaVA-7B or 13B** (backed by Vicuna/LLaMA with safety training) over TinyLLaVA. Tiny models typically lack meaningful safety alignment, so measuring safety degradation on them would be uninformative. BLIP-2 with Flan-T5 backend is also worth testing since it has different safety characteristics. We have an H200 available, so running 7B/13B is not a constraint.

---

## 2. What Already Exists (Related Work We MUST Know)

### Directly competing / overlapping work

| Paper | Key Idea | How we differ |
|-------|----------|---------------|
| **HSR** (ACL 2025 Findings) [arxiv 2505.16104] | Restores safety in pruned LVLMs by identifying safety-critical attention heads and selectively restoring neurons post-pruning. Recovers 14-27% of lost safety. | HSR is **reactive** (fix after pruning). Our approach is **proactive** (preserve during compression by design). We also connect safety to task circuits, which HSR doesn't. |
| **Chhabra & Khalili 2025** [arxiv 2504.04215] | Found that quantized models preserve refusal directions while pruned models don't. Proposed AIRD to restore refusal in compressed models. | They work on **text-only LLMs**, not VLMs. We extend to multimodal. Also, AIRD is post-hoc; we aim to preserve safety during compression. |
| **COSMIC** (ACL 2025 Findings) | Generalized refusal direction identification across model architectures. | Complementary — we can use their methods to identify refusal directions in VLMs. |
| **NOTICE** (NAACL 2025) [arxiv 2406.16320] | MI pipeline for VLMs. Found "universal attention heads" in BLIP and LLaVA. Cross-attention in BLIP does object detection/suppression; LLaVA self-attention does outlier suppression. | We build on their findings for RQ1, but they don't study safety or compression. |

### Safety evaluation tools we can use

| Benchmark | What it measures | Notes |
|-----------|-----------------|-------|
| **MMJ-Bench** (AAAI 2025) | Unified jailbreak attack + defense evaluation for VLMs | Has code on GitHub, includes multiple attack types |
| **MM-SafetyBench** (ECCV 2024) | Safety evaluation for multimodal LLMs | Includes typographic attacks (text-in-image) |
| **VLGuard / SafetyBench** | General safety benchmarks | For baseline safety scores |

### Key insight from literature
- In text LLMs, refusal is mediated by a **single direction** in the residual stream
- Quantization preserves this direction; **pruning destroys it** (Chhabra & Khalili)
- **Nobody has mapped refusal/safety circuits in VLMs yet** — this is our gap and contribution
- HSR is the closest work but is reactive and doesn't connect safety to task-specific circuits

---

## 3. RQ3 Experimental Plan

### Boundary with RQ1/RQ2 (Avoiding Double Work)

| What | Who does it | RQ3 role |
| ---- | ----------- | -------- |
| Task circuit discovery (IC/VQA circuits) | RQ1 team | We **consume** their circuit maps — don't redo |
| Compressed model baselines (Wanda, GPTQ, AWQ) | RQ1 team / baseline task | We **consume** their models — don't redo |
| Task-specific compressed models | RQ2 team | We **consume** their models — don't redo |
| **Safety circuit discovery** | **RQ3 (us)** | Our unique contribution — same tools (activation patching), but applied to safety/refusal prompts, not task prompts |
| **Safety evaluation** | **RQ3 (us)** | Our unique contribution — safety benchmarks, jailbreak testing |
| **Overlap analysis** (task circuits vs. safety circuits) | **RQ3 (us)** | Bridges RQ1 findings with our safety findings |

**In short**: RQ1 gives us task circuits + compressed models. RQ2 gives us task-specific compressed models. We only run NEW experiments on safety-specific questions. The activation patching tooling is the same, but the prompts, metrics, and research question are different.

### Experiment 1: Safety Circuit Discovery in VLMs (NOVEL — no one has done this)

**Goal**: Map which components (attention heads, MLPs, projector layers) are responsible for safety/refusal behavior in VLMs.

**What makes this different from RQ1**: RQ1 patches with task-relevant prompts (e.g., correct vs. wrong caption) to find task circuits. We patch with safety-relevant prompts (harmful vs. benign) to find safety circuits. Same method, different question.

**Method**:

1. Construct paired prompts:
   - (image, benign question) vs. (image, harmful question) — same image, different text
   - (benign image, question) vs. (adversarial image, same question) — same text, different image
   - Typographic attacks: (image with harmful text embedded, benign question)
2. Run **activation patching** on LLaVA-7B and BLIP-2:
   - Patch each attention head, MLP layer, and projector component
   - Measure change in refusal probability
   - Identify components where patching flips refusal -> compliance (these are safety-critical)
3. Also identify **refusal directions** in the residual stream (following Chhabra & Khalili's approach, but for VLMs)

**Output**: A map of safety-critical components per model and per modality (text path vs. vision path vs. cross-modal).

**Dependency**: None — can start immediately on uncompressed models.

### Experiment 2: Circuit Overlap Analysis

**Goal**: Quantify the relationship between task circuits (from RQ1) and safety circuits (from Exp 1).

**Input from RQ1**: Task circuit maps (which components are necessary for IC and VQA).

**Method**:

1. Take task circuits from RQ1 (components necessary for IC and VQA)
2. Compute overlap with safety circuits from Experiment 1
3. Analyze per-layer and per-module-type (attention vs. MLP vs. projector)

**Possible findings (all publishable)**:

- **High overlap** → Task-specific compression naturally preserves safety (good news story)
- **Low overlap** → Task-specific compression is just as dangerous as blind compression for safety; need to explicitly protect safety circuits (motivates our method extension)
- **Asymmetric** → Some tasks (e.g., VQA) overlap more with safety than others (e.g., IC) — interesting finding about task-safety relationships

**Dependency**: Needs RQ1 task circuits. Experiment 1 can run in parallel with RQ1.

### Experiment 3: Safety Evaluation of Compressed Models

**Goal**: Empirically demonstrate safety differences across compression methods.

**Input from RQ1/RQ2**: Compressed model checkpoints (we don't compress ourselves).

**Compare 4 (or 5) model variants**:

1. **Full model** (uncompressed) — upper bound
2. **Blind-pruned** (Wanda / SparseGPT) — from RQ1/baseline task
3. **Blind-quantized** (GPTQ / AWQ) — from RQ1/baseline task
4. **Task-specific compressed** (from RQ2 — preserves task circuit only)
5. **(Optional) Task+Safety compressed** (preserves union of task + safety circuits)

**Evaluation metrics**:

- **Refusal rate** on harmful prompts (% of times model refuses)
- **Attack success rate (ASR)** using MMJ-Bench jailbreak attacks:
  - Text-based jailbreaks (prompt injection, role-playing)
  - Vision-based jailbreaks (typographic attacks, adversarial images)
  - Combined multimodal attacks
- **Toxicity score** on adversarial inputs
- **Task performance** simultaneously (to show safety doesn't cost accuracy)

**Dependency**: Needs compressed models from RQ1/RQ2 teams.

---

## 4. What We Can Start NOW (Before RQ1/RQ2 Deliver)

| Task | Description | Owner | Blocked? |
|------|-------------|-------|----------|
| Safety prompt dataset | Curate paired safe/harmful prompts for VLMs (text + image pairs) | RQ3 team | No |
| Experiment 1 | Run safety circuit discovery on uncompressed LLaVA-7B / BLIP-2 | RQ3 team | No |
| Evaluation harness | Set up MMJ-Bench, MM-SafetyBench, toxicity scoring pipeline | RQ3 team | No |
| Literature framing | Write the Section 4 intro (safety risks of compression, gap in VLM safety analysis) | RQ3 team | No |

**NOT our job** (consume from other teams):

- Baseline compression (Wanda, GPTQ, AWQ) — RQ1 / baseline task
- Task circuit discovery — RQ1
- Task-specific compression method — RQ2

---

## 5. Our Differentiation (Why This Is Not HSR or Chhabra & Khalili)

1. **Proactive, not reactive**: We preserve safety circuits DURING compression, not restore them after. HSR is a band-aid; we're prevention.
2. **VLM-specific**: Chhabra & Khalili only studied text LLMs. We map safety circuits across vision encoder, projector, and language decoder — revealing which modality pathway carries safety.
3. **Task-safety connection**: We're the first to study the relationship between task-specific circuits and safety circuits. This yields insight into whether task-specific compression is inherently safer or not.
4. **Multimodal attack surface**: We evaluate with VLM-specific attacks (typographic, adversarial images), not just text-based safety metrics.

---

## 6. Expected Paper Contribution (Section 4)

By the end, Section 4 should contain:
- First mapping of safety/refusal circuits in VLMs (Experiment 1)
- Analysis of task-safety circuit overlap (Experiment 2)
- Empirical safety comparison across compression methods (Experiment 3)
- Argument for why task-specific compression (optionally augmented with safety circuit preservation) is the right paradigm for safe model deployment
