# Methodology

## 1. Overview

This work presents an empirical study of adversarial robustness under post-training quantization (PTQ) for Vision Transformers (ViTs) deployed at the edge. The central research question is whether adversarial defenses remain effective after model compression, and whether this effectiveness degrades as compression becomes more aggressive.

Every experimental condition is constructed by applying the same three operations to the pretrained model, always in this fixed order:

1. **Compress the model.** The pretrained FP32 DeiT-Small backbone is quantized via post-training quantization to the target precision (INT8 or INT4; Section 3.3). The compressed model *is* the edge-deployed artifact, and all subsequent operations act on it.
2. **Defend the model.** A defense is applied to the *already-compressed* model by adversarially fine-tuning its trainable parameters — either Adversarial Training (AT; Section 4) or Adversarial Training with Knowledge Distillation (AT+KD; Section 4) — while the quantized weights stay frozen. The undefended baseline omits this operation.
3. **Attack the model.** The resulting (compressed, optionally defended) model is attacked white-box under the L∞ threat model (FGSM, PGD, Patch, or the combined attack; Section 5), and its clean and robust accuracy are measured (Section 6).

These operations are organised into evaluation phases:

1. **Phase 1** — Baseline evaluation of an undefended model across all three precision levels (FP32, INT8, INT4) and three attack types.
2. **Phase 2a** — Adversarial Training (AT) applied to each compressed model, followed by re-evaluation under all attacks.
3. **Phase 2b** — Adversarial Training with Knowledge Distillation (AT+KD) applied to each compressed model, followed by re-evaluation under all attacks.
4. **Phase 3** — The combined attack — *sequential perturbation composition* of FGSM → PGD → Patch (Section 5.4) — applied against the undefended, AT, and AT+KD models at each compression level.

The pipeline strictly enforces the order **compress first, then defend, then attack**. Compressing before defending reflects the realistic edge deployment scenario in which a model is quantized for resource constraints *before* any robustness fine-tuning is applied, so that the defended artifact is itself the compressed model that will be deployed.

---

## 2. Dataset

### 2.1 Dataset Description

All experiments use **Imagenette2-320**, a 10-class subset of ImageNet-1K introduced by fast.ai, containing the following classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, and parachute. Images are sourced at 320-pixel resolution and resized during preprocessing.

| Split | Images Used | Selection Method |
|-------|-------------|-----------------|
| Validation | 3,925 (entire val split) | Deterministic `torch.randperm` with seed 42 |
| Training (AT/AT+KD) | 9,469 (entire train split) | Deterministic `torch.randperm` with seed 42 |
| Patch evaluation | 500 | Strict prefix of the 3,925-image validation subset |

The configured subset sizes (`val_subset_size = 3925`, `train_subset_size = 9469`) are capped to the available image counts, so in practice the *entire* Imagenette validation and training splits are used. The patch evaluation subset is intentionally smaller (500 images) because the adversarial patch attack requires 150 optimisation steps per batch, making it approximately 7.5× more expensive than FGSM or PGD. Using a 500-image prefix of the full validation subset preserves comparability of clean accuracy denominators across attacks.

### 2.2 Label Remapping

Since Imagenette2-320 contains only 10 classes, its integer class indices (0–9) do not correspond to ImageNet-1K indices (0–999). The remapping table assigns each synset to its ImageNet-1K position (e.g. tench → 0, English springer → 217, church → 497, parachute → 701). All experiment scripts apply a deterministic remapping from ImageFolder synset identifiers to their corresponding ImageNet-1K indices before computing accuracy, ensuring that pretrained model logits are evaluated against the correct class positions.

### 2.3 Preprocessing

The following preprocessing pipeline is applied consistently across all splits and phases:

```
Resize(256) → CenterCrop(224) → ToTensor() → Normalize(mean, std)
```

For training (AT/AT+KD fine-tuning), data augmentation is applied:

```
RandomResizedCrop(224) → RandomHorizontalFlip() → ToTensor() → Normalize(mean, std)
```

ImageNet normalisation statistics are used throughout:

- **Mean**: [0.485, 0.456, 0.406]
- **Std**: [0.229, 0.224, 0.225]

### 2.4 Rationale

Imagenette2-320 is selected as a computationally tractable proxy for full ImageNet-1K evaluation. Its 10 classes are visually distinct, reducing label ambiguity, while the use of an ImageNet-pretrained model means that the 10-class distribution is a strict subset of the model's training distribution — clean accuracy figures are directly comparable to full ImageNet benchmarks.

---

## 3. Model Architecture

### 3.1 Base Architecture

All experiments use **DeiT-Small** (Data-efficient Image Transformer, Small variant), a Vision Transformer architecture proposed by Touvron et al. (2021). DeiT-Small processes images as sequences of non-overlapping 16×16 patches with a class token prepended for classification.

| Property | Value |
|----------|-------|
| Architecture | Vision Transformer (ViT) |
| Variant | DeiT-Small |
| timm identifier | `deit_small_patch16_224` |
| HuggingFace identifier | `facebook/deit-small-patch16-224` |
| Parameters | ~22M |
| Input resolution | 224 × 224 |
| Patch size | 16 × 16 |
| Transformer blocks | 12 |
| Number of classes | 1,000 |

### 3.2 Pre-training

The model is loaded with ImageNet-1K pretrained weights via `timm.create_model(..., pretrained=True)` for **all** precision levels. The quantized variants are not loaded from a separate checkpoint or via the HuggingFace `transformers` API; instead, the FP32 `timm` model is instantiated and its `nn.Linear` layers are replaced **in place** with `bitsandbytes` quantized layers (Section 3.3). Model outputs are routed through a thin `LogitsWrapper` that returns a plain $(N, C)$ logits tensor, providing a uniform interface across all precision levels for the attack library and the evaluation loop. No fine-tuning is performed on the undefended baseline — Phase 1 evaluates the (compressed) pretrained weights directly.

### 3.3 Compression Levels

Three precision levels are evaluated — **FP32**, **INT8**, and **INT4** — declared as `compression.levels = ["fp32", "int8", "int4"]` and evaluated in every phase. FP32 is both the full-precision baseline (clean-accuracy and robustness upper bound) *and* the frozen teacher in AT+KD (Section 4). Only the **linear (weight) layers** are quantized; LayerNorm, embeddings, and the patch-projection convolution remain in their native precision. Quantization is **data-free** — no calibration set is used, as the schemes quantize the pretrained weights directly.

| Level | Method | Backend | Implementation |
|-------|--------|---------|---------------|
| FP32 | None (baseline) | timm | Full-precision pretrained weights |
| INT8 | Post-training quantization | bitsandbytes ≥0.41.0 | `nn.Linear` → `bnb.nn.Linear8bitLt` (`has_fp16_weights=False`), true int8 weight storage, cuBLAS-LT 8-bit kernel; requires CUDA sm_70+ (Tesla T4 is sm_75) |
| INT4 | Post-training quantization (NF4) | bitsandbytes ≥0.41.0 | `nn.Linear` → `bnb.nn.Linear4bit` (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16`), weights stored as `Params4bit` in NF4 |

INT4 uses NormalFloat-4 (NF4) quantization, which maps weights to the 16 values of a normal distribution. NF4 is information-theoretically optimal for normally distributed weights and is used as the compression scheme in QLoRA (Dettmers et al., 2023).

**Weight-only quantization (scope of the edge claim).** Both INT8 and INT4 are *weight-only* PTQ: the integer-stored weights are dequantized to the compute dtype (FP16 for INT4; the int8 path uses bitsandbytes' mixed-precision cuBLAS-LT kernel) on the fly during each matrix multiply, and activations are not quantized. The compression therefore reduces the model's **memory footprint** — the primary edge constraint — but does not constitute full integer-only inference; reported robustness reflects this weight-only regime.

### 3.4 Justification

DeiT-Small is selected as the primary model for the following reasons:

- **Parameter efficiency**: At ~22M parameters, it fits comfortably within the 16 GB VRAM of the Tesla T4 GPUs used for experiments, alongside the FP32 teacher required for AT+KD.
- **ViT relevance**: Vision Transformers are increasingly deployed at the edge; studying their robustness under quantization is of direct practical relevance.
- **Compression sensitivity**: The attention mechanism and layer normalisation in ViTs are known to be sensitive to quantization, making INT4 degradation a measurable and scientifically interesting phenomenon.

---

## 4. Training Configuration

### 4.1 Adversarial Fine-tuning (AT and AT+KD)

Both defenses fine-tune the already-compressed model. Training configuration is shared across AT and AT+KD except where noted.

**Adversarial example generation during training.** Both defenses use **single-step FGSM** to generate the adversarial inputs on which the model is trained (i.e. *FGSM-AT*), at the training budget $\varepsilon_{\text{train}} = 8/255$ matching the evaluation budget. This is **not** PGD-AT (Madry et al., 2018) or TRADES (Zhang et al., 2019). FGSM-AT is chosen because the experiments run under a constrained Kaggle dual-T4 compute budget, and multi-step inner maximisation across three compression levels and seven epochs would exceed it. We note that single-step FGSM-AT is known to be weaker than PGD-AT and susceptible to *catastrophic overfitting*, where robustness to multi-step attacks collapses abruptly during training (Andriushchenko & Flammarion, 2020); the layer-freezing and small per-compression learning rates (Sections 4.2–4.3) mitigate, but do not eliminate, this risk, and it is a stated limitation of the defense configuration.

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Epochs | 7 |
| Batch size | 16 |
| Warmup epochs | 1 (epoch 1 uses lr/10) |
| Training adversary | FGSM, single-step |
| AT epsilon ($\varepsilon_{\text{train}}$) | 8/255 ≈ 0.03137 |
| Checkpoint frequency | Every epoch |
| Loss function | Cross-entropy (AT); weighted CE + KL divergence (AT+KD), see below |

**AT loss (Phase 2a).** Standard cross-entropy on the FGSM-perturbed inputs:

$$\mathcal{L}_{\text{AT}} = \mathcal{L}_{\text{CE}}\!\big(f_\theta(\mathbf{x}_{\text{adv}}),\, y\big), \qquad \mathbf{x}_{\text{adv}} = \text{FGSM}(f_\theta, \mathbf{x}, y;\, \varepsilon_{\text{train}}).$$

**AT+KD loss (Phase 2b).** Cross-entropy on hard labels plus a temperature-scaled KL-divergence distillation term from a frozen FP32 teacher:

$$\mathcal{L}_{\text{AT+KD}} = \alpha \cdot \mathcal{L}_{\text{CE}}\!\big(f_S(\mathbf{x}_{\text{adv}}),\, y\big) + (1-\alpha)\cdot \tau^2 \cdot \text{KL}\!\Big(\sigma\big(\tfrac{\mathbf{z}_S}{\tau}\big)\,\big\|\,\sigma\big(\tfrac{\mathbf{z}_T}{\tau}\big)\Big),$$

where $\mathbf{z}_S, \mathbf{z}_T$ are the student and teacher logits, $\sigma$ is the softmax, the temperature is $\tau = 4.0$, and the cross-entropy weight is $\alpha = 0.5$ (so the KD weight is $1-\alpha = 0.5$). The $\tau^2$ factor restores the gradient magnitude attenuated by temperature scaling (Hinton et al., 2015). **The teacher is queried on the same FGSM adversarial inputs $\mathbf{x}_{\text{adv}}$ as the student — not on clean inputs** — so the soft targets describe the teacher's behaviour under attack. The teacher is the frozen, full-precision (FP32) pretrained DeiT-Small (it is *not* itself adversarially trained); it is held in `eval()` mode with every forward pass wrapped in `torch.no_grad()`, and is never updated.

### 4.2 Per-Compression Learning Rates

Per-compression learning rates are used because quantized weights are significantly more fragile than full-precision weights under gradient updates. INT4 NF4 weights cannot absorb large gradient steps without catastrophic weight corruption.

| Compression | Learning Rate |
|-------------|--------------|
| FP32 | 1 × 10⁻⁵ |
| INT8 | 5 × 10⁻⁶ |
| INT4 | 1 × 10⁻⁶ |

### 4.3 Layer Freezing

To reduce the risk of catastrophic forgetting under adversarial fine-tuning, the backbone is partially frozen. Specifically:

- All parameters are frozen initially.
- The **last 4 transformer blocks** and the **classifier head** are unfrozen and updated during training.
- Quantized (integer) parameters from bitsandbytes are excluded from gradient computation regardless of freezing status.

This yields approximately 7.48M trainable parameters out of 22.05M total (33.9%) in the unquantized architecture; for INT8/INT4 students the trainable count is lower, since quantized integer weights in the unfrozen blocks are excluded from gradient updates.

### 4.4 Hardware

All experiments are conducted on **Kaggle Notebooks** using a **dual NVIDIA Tesla T4** accelerator (2× T4, 16 GB VRAM each) under Kaggle's free-tier weekly GPU-hour budget. Evaluation DataLoaders use `num_workers=2` (matching the 4-core Kaggle CPU allocation), while the small per-epoch clean-accuracy probe loaders inside the defense routines use `num_workers=0` to avoid multiprocessing overhead.

---

## 5. Adversarial Attack Methods

All attacks operate under the **L∞ threat model** with perturbation bound ε = 8/255. Input images are ImageNet-normalised; all attacks internally un-normalise to [0, 1] pixel space, apply perturbations, clamp to [0, 1], and re-normalise before returning adversarial examples. This ensures that the measured L∞ perturbation in pixel space exactly matches the configured ε.

### 5.1 FGSM (Fast Gradient Sign Method)

FGSM (Goodfellow et al., 2014) computes a single-step gradient-sign perturbation:

$$\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon \cdot \text{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}, y)\right)$$

where $\mathcal{L}$ is the cross-entropy loss, $\theta$ are the model parameters, and $\varepsilon$ is the perturbation bound.

| Parameter | Value |
|-----------|-------|
| ε | 8/255 ≈ 0.03137 |
| Norm | L∞ |
| Steps | 1 |
| Targeted | No |
| Implementation | `torchattacks.FGSM` |

FGSM is used both as an evaluation attack and as the adversarial training attack in the AT and AT+KD defenses, ensuring consistency between the training threat model and the evaluation threat model.

### 5.2 PGD (Projected Gradient Descent)

PGD (Madry et al., 2018) iteratively refines the adversarial perturbation through multiple gradient steps, projecting back onto the L∞ ball after each step:

$$\mathbf{x}^{(t+1)} = \Pi_{\mathcal{B}_\infty(\mathbf{x}, \varepsilon)}\left(\mathbf{x}^{(t)} + \alpha \cdot \text{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}^{(t)}, y)\right)\right)$$

where $\Pi_{\mathcal{B}_\infty(\mathbf{x}, \varepsilon)}$ denotes projection onto the L∞ ball of radius ε centred at the clean input $\mathbf{x}$, and $\alpha$ is the step size.

| Parameter | Value |
|-----------|-------|
| ε | 8/255 ≈ 0.03137 |
| α (step size) | 2/255 ≈ 0.00784 |
| Steps | 20 |
| Norm | L∞ |
| Targeted | No |
| Implementation | `torchattacks.PGD` |

PGD with 20 steps and α = 2/255 is the standard configuration in the adversarial robustness literature and is sufficient to find strong adversarial examples under the L∞ constraint.

### 5.3 Adversarial Patch Attack

The adversarial patch attack (Brown et al., 2017) optimises a localised, visually conspicuous patch that causes misclassification when applied to any image. Unlike FGSM and PGD, the patch is not constrained to be imperceptible — it is physically realisable and can be printed and placed on real-world objects.

**Patch optimisation:** A 32×32 pixel patch is initialised uniformly and optimised by maximising the cross-entropy loss via sign-based gradient ascent:

$$\mathbf{p}^{(t+1)} = \text{clip}_{[0,1]}\left(\mathbf{p}^{(t)} + \eta \cdot \text{sign}\left(\nabla_{\mathbf{p}} \mathcal{L}(\theta, \mathbf{x} \oplus \mathbf{p}, y)\right)\right)$$

where $\mathbf{p}$ is the patch, $\eta$ is the patch learning rate, and $\mathbf{x} \oplus \mathbf{p}$ denotes the image with the patch applied at a uniformly random location.

| Parameter | Value |
|-----------|-------|
| Patch size | 32 × 32 pixels |
| Image size | 224 × 224 pixels |
| Patch area | ~2% of image area |
| Optimisation steps | 150 |
| Patch learning rate (η) | 0.05 |
| Placement | Uniformly random per batch |
| Norm constraint | Clamped to [0, 1] per step |
| Loss | Cross-entropy (maximised) |

The patch is evaluated on a 500-image subset of the validation set rather than the full 3,925-image subset, as 150 optimisation steps per batch makes the patch attack approximately 7.5× more expensive than FGSM or PGD per image.

### 5.4 Combined Attack (Phase 3)

The Phase 3 combined attack is a **sequential perturbation composition** of the three preceding attacks, applied in the fixed order FGSM → PGD → Patch:

$$\mathbf{x}_{\text{adv}} = \text{Patch}\Big(\text{PGD}\big(\text{FGSM}(\mathbf{x},\, y),\, y\big),\, y\Big).$$

The adversarial *output* of each stage becomes the *input* to the next, and the same ground-truth labels $y$ are used at every stage. It is **not** a survivor-filtering cascade (FGSM, then re-attack only the still-correct images with PGD, etc.) and **not** an ensemble that selects the strongest single perturbation or sums per-attack losses — every image passes through all three stages in sequence, accumulating an FGSM step, then 20 PGD steps, then a 150-step optimised patch. Because each stage operates in ImageNet-normalised space and internally un-normalises/re-normalises (Section 5), the composition is well-defined end-to-end. The combined attack is the strongest perturbation considered and, like the patch attack, is evaluated on the 500-image subset.

| Parameter | Value |
|-----------|-------|
| Composition | Sequential: FGSM → PGD → Patch |
| Per-stage parameters | As in Sections 5.1–5.3 (ε = 8/255; PGD 20 steps, α = 2/255; patch 32×32, 150 steps) |
| Labels | Ground-truth $y$, shared across all stages |
| Evaluation subset | 500 images (prefix of the validation subset) |

---

## 6. Evaluation Metrics

Four metrics are recorded for every (model, compression, defense, attack) combination:

| Metric | Definition |
|--------|-----------|
| **Clean accuracy** | $\text{Acc}_\text{clean} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$ on clean inputs |
| **Robust accuracy** | $\text{Acc}_\text{robust} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i^{\text{adv}} = y_i]$ on adversarial inputs |
| **Attack success rate (ASR)** | $\text{ASR} = 1 - \text{Acc}_\text{robust}$ |
| **Robustness gap** | $\Delta = \text{Acc}_\text{clean} - \text{Acc}_\text{robust}$ |

All metrics are computed over the same deterministic validation subset (seed=42) across all phases, ensuring direct comparability. For the patch attack, clean accuracy is re-computed on the 500-image subset so that the denominator matches the robust accuracy denominator.

Results are written to CSV immediately after each (compression, attack) evaluation, enabling incremental recovery from interruption without recomputation.

---

## 7. Experimental Setup

### 7.1 Evaluation Protocol

- **Validation subset**: 3,925 images (the entire validation split) drawn deterministically from the Imagenette2-320 validation split using `torch.randperm(seed=42)`.
- **Patch evaluation subset**: 500 images as a strict prefix of the 3,925-image validation subset.
- **Training subset**: 9,469 images (the entire training split) drawn deterministically from the training split using `torch.randperm(seed=42)`.
- **Batch size (eval)**: 32.
- **Batch size (training)**: 16.
- No cross-validation is performed — the validation subset is fixed across all phases to ensure metric comparability.

### 7.2 Resumability

All experiment scripts implement CSV-based resumability: completed (model, compression, defense, attack) tuples are read from the results CSV at startup, and any already-completed combinations are skipped. This ensures that partial runs due to hardware interruption can be resumed without duplicating results.

### 7.3 Random Seeds

All randomness is seeded at the start of every script:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 7.4 Statistical Reporting

Each (compression, defense, attack) cell is evaluated once on the fixed 3,925-image (or 500-image for patch and combined) validation subset. No repeated trials are performed. Variance across seeds is not reported; the deterministic subset selection ensures that all reported numbers are reproducible exactly given the same model weights and environment.

---

## 8. Ablation Studies

### 8.1 Compression Level Sensitivity

The primary ablation is the compression level sweep (INT8 → INT4). This directly quantifies how aggressively PTQ degrades both clean accuracy and adversarial robustness, and whether AT and AT+KD can recover robustness at each level.

### 8.2 Defense Comparison

AT vs. AT+KD at each compression level constitutes a controlled ablation of the knowledge distillation component. Both defenses share identical training configurations; the only difference is the addition of the KL divergence term and the frozen FP32 teacher in AT+KD.

---

## 9. Reproducibility

### 9.1 Code

All code is available at: **https://github.com/Jmanav/ADVC**

The repository includes:
- `configs/base.yaml` — single source of truth for all hyperparameters
- `attacks/` — FGSM, PGD, and Patch attack implementations
- `defenses/` — AT and AT+KD fine-tuning implementations
- `experiments/` — Phase 1, 2a, and 2b evaluation scripts
- `notebooks/run_phases_local.ipynb` — end-to-end notebook for reproducing all results

### 9.2 Library Versions

| Library | Minimum Version |
|---------|----------------|
| PyTorch | ≥ 2.1.0 |
| torchvision | ≥ 0.16.0 |
| timm | ≥ 0.9.0 |
| torchattacks | ≥ 3.4.0 |
| bitsandbytes | ≥ 0.41.0 |
| transformers | ≥ 4.35.0 (optional; only the `bitsandbytes` integration is used — models load via `timm`) |
| accelerate | ≥ 0.24.0 (optional; pulled in transitively by `bitsandbytes`) |
| PyYAML | ≥ 6.0 |
| tqdm | ≥ 4.0.0 |

### 9.3 Computational Requirements

| Resource | Specification |
|----------|--------------|
| GPU | 2× NVIDIA Tesla T4 (16 GB VRAM each), Kaggle Notebooks |
| CPU | 4 cores (Kaggle allocation) |
| RAM | ~16 GB (Kaggle allocation) |
| Phase 1 (baseline eval) | ~30–45 min |
| Phase 2a (AT, 2 compression levels) | ~1.5–2 hr |
| Phase 2b (AT+KD, 2 compression levels) | ~1.5–2 hr |
| **Total** | **~4–5 hr** |

## References

- Andriushchenko, M., & Flammarion, N. (2020). *Understanding and improving fast adversarial training*. NeurIPS 2020.
- Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). *Adversarial patch*. arXiv:1712.09665.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. NeurIPS 2023.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv:1412.6572.
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. arXiv:1503.02531.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR 2018.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). *Training data-efficient image transformers & distillation through attention*. ICML 2021.
- Zhang, H., Yu, Y., Jiao, J., Xing, E. P., El Ghaoui, L., & Jordan, M. I. (2019). *Theoretically principled trade-off between robustness and accuracy (TRADES)*. ICML 2019.