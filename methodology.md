# Methodology

## 1. Overview

This work presents an empirical study of adversarial robustness under post-training quantization (PTQ) for Vision Transformers (ViTs) deployed at the edge. The central research question is whether adversarial defenses remain effective after model compression, and whether this effectiveness degrades as compression becomes more aggressive.

The experimental design follows a three-phase pipeline:

1. **Phase 1** — Baseline evaluation of an undefended model across two compression levels (INT8, INT4) and three attack types.
2. **Phase 2a** — Adversarial Training (AT) applied to each compressed model, followed by re-evaluation.
3. **Phase 2b** — Adversarial Training with Knowledge Distillation (AT+KD) applied to each compressed model, followed by re-evaluation.

The compression pipeline strictly enforces the order: **compress first, then defend**. This reflects the realistic edge deployment scenario in which a model is quantized for resource constraints before any robustness fine-tuning is applied.

---

## 2. Dataset

### 2.1 Dataset Description

All experiments use **Imagenette2-320**, a 10-class subset of ImageNet-1K introduced by fast.ai, containing the following classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, and parachute. Images are sourced at 320-pixel resolution and resized during preprocessing.

| Split | Images Used | Selection Method |
|-------|-------------|-----------------|
| Validation | 5,000 | Deterministic `torch.randperm` with seed 42 |
| Training (AT/AT+KD) | 10,000 | Deterministic `torch.randperm` with seed 42 |
| Patch evaluation | 500 | Strict prefix of the 5,000-image validation subset |

The patch evaluation subset is intentionally smaller (500 images) because the adversarial patch attack requires 150 optimisation steps per batch, making it approximately 7.5× more expensive than FGSM or PGD. Using a 500-image prefix of the full validation subset preserves comparability of clean accuracy denominators across attacks.

### 2.2 Label Remapping

Since Imagenette2-320 contains only 10 classes, its integer class indices (0–9) do not correspond to ImageNet-1K indices (0–999). All experiment scripts apply a deterministic remapping from ImageFolder synset identifiers to their corresponding ImageNet-1K indices before computing accuracy, ensuring that pretrained model logits are evaluated against the correct class positions.

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

The model is loaded with ImageNet-1K pretrained weights via `timm.create_model(..., pretrained=True)` for FP32, and via `transformers.AutoModelForImageClassification.from_pretrained` for INT8 and INT4 quantised variants. No fine-tuning is performed on the undefended baseline — Phase 1 evaluates the pretrained weights directly.

### 3.3 Compression Levels

Two compression configurations are evaluated. FP32 is not evaluated as a compression level; it serves only as the pre-quantization base model and as the frozen teacher in AT+KD (Section 4).

| Level | Method | Backend | Configuration |
|-------|--------|---------|---------------|
| INT8 | Post-training quantization | bitsandbytes ≥0.41.0 | `load_in_8bit=True` |
| INT4 | Post-training quantization (NF4) | bitsandbytes ≥0.41.0 | `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16` |

INT4 uses NormalFloat-4 (NF4) quantization, which maps weights to the 16 values of a normal distribution. NF4 is information-theoretically optimal for normally distributed weights and is used as the compression scheme in QLoRA (Dettmers et al., 2023). Compute operations are performed in float16 to maintain numerical stability.

### 3.4 Justification

DeiT-Small is selected as the primary model for the following reasons:

- **Parameter efficiency**: At ~22M parameters, it fits within the 8.6 GB VRAM constraint of the H100 slice used for experiments.
- **ViT relevance**: Vision Transformers are increasingly deployed at the edge; studying their robustness under quantization is of direct practical relevance.
- **Compression sensitivity**: The attention mechanism and layer normalisation in ViTs are known to be sensitive to quantization, making INT4 degradation a measurable and scientifically interesting phenomenon.

---

## 4. Training Configuration

### 4.1 Adversarial Fine-tuning (AT and AT+KD)

Both defenses fine-tune the already-compressed model. Training configuration is shared across AT and AT+KD except where noted.

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Epochs | 7 |
| Batch size | 16 |
| Warmup epochs | 1 (epoch 1 uses lr/10) |
| AT epsilon | 8/255 ≈ 0.03137 |
| Checkpoint frequency | Every epoch |
| Loss function | Cross-entropy (AT); weighted CE + KL divergence (AT+KD) |

### 4.2 Per-Compression Learning Rates

Per-compression learning rates are used because quantized weights are significantly more fragile than full-precision weights under gradient updates. INT4 NF4 weights cannot absorb large gradient steps without catastrophic weight corruption.

| Compression | Learning Rate |
|-------------|--------------|
| INT8 | 5 × 10⁻⁶ |
| INT4 | 1 × 10⁻⁶ |

### 4.3 Layer Freezing

To reduce the risk of catastrophic forgetting under adversarial fine-tuning, the backbone is partially frozen. Specifically:

- All parameters are frozen initially.
- The **last 4 transformer blocks** and the **classifier head** are unfrozen and updated during training.
- Quantized (integer) parameters from bitsandbytes are excluded from gradient computation regardless of freezing status.

This yields approximately 7.48M trainable parameters out of 22.05M total (33.9%) in the unquantized architecture; for INT8/INT4 students the trainable count is lower, since quantized integer weights in the unfrozen blocks are excluded from gradient updates.

### 4.4 Hardware

All experiments are conducted on a **NVIDIA H100 GPU** (8.6 GB VRAM slice) with 1 Intel Xeon Platinum 8480+ CPU core and 28 GB RAM. DataLoader `num_workers=0` is used throughout to avoid multiprocessing deadlocks with a single CPU core.

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

The patch is evaluated on a 500-image subset of the validation set rather than the full 5,000-image subset, as 150 optimisation steps per batch makes the patch attack approximately 7.5× more expensive than FGSM or PGD per image.

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

- **Validation subset**: 5,000 images drawn deterministically from the Imagenette2-320 validation split using `torch.randperm(seed=42)`.
- **Patch evaluation subset**: 500 images as a strict prefix of the 5,000-image validation subset.
- **Training subset**: 10,000 images drawn deterministically from the training split using `torch.randperm(seed=42)`.
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

Each (compression, defense, attack) cell is evaluated once on the fixed 5,000-image (or 500-image for patch) validation subset. No repeated trials are performed. Variance across seeds is not reported; the deterministic subset selection ensures that all reported numbers are reproducible exactly given the same model weights and environment.

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
| transformers | ≥ 4.35.0 |
| accelerate | ≥ 0.24.0 |
| PyYAML | ≥ 6.0 |
| tqdm | ≥ 4.0.0 |

### 9.3 Computational Requirements

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA H100 (8.6 GB VRAM slice) |
| CPU | 1 core Intel Xeon Platinum 8480+ |
| RAM | 28 GB |
| Phase 1 (baseline eval) | ~30–45 min |
| Phase 2a (AT, 2 compression levels) | ~1.5–2 hr |
| Phase 2b (AT+KD, 2 compression levels) | ~1.5–2 hr |
| **Total** | **~4–5 hr** |

### 9.4 Environment Setup

To reproduce all results:

```bash
git clone https://github.com/Jmanav/ADVC.git
cd ADVC
pip install -r requirements.txt
# Edit configs/base.yaml to set dataset.val_dir and dataset.train_dir
python experiments/eval_phase1.py --model deit_small
python experiments/eval_phase2_at.py --model deit_small
python experiments/eval_phase2_atkd.py --model deit_small
```

All scripts are fully resumable — they can be interrupted and restarted without recomputing completed rows.

---

## References

- Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). *Adversarial patch*. arXiv:1712.09665.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. NeurIPS 2023.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv:1412.6572.
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. arXiv:1503.02531.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR 2018.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). *Training data-efficient image transformers & distillation through attention*. ICML 2021.
