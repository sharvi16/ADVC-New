# 3. Methodology

This study evaluates whether adversarial defenses remain effective after
post-training quantization (PTQ) of a Vision Transformer (ViT) intended for edge
deployment, and whether their effectiveness degrades as quantization becomes more
aggressive. The experimental design enforces a strict *compress-first, defend-second*
pipeline that mirrors a realistic edge workflow. All hyperparameters are read from a
single configuration file (`configs/base.yaml`), which is the authoritative source
for every value reported below; no hyperparameter is hardcoded in the experiment
scripts.

## 3.1 Dataset and Preprocessing

All experiments use **ImageNette** (`imagenette2`), a ten-class subset of ImageNet-1k
consisting of the synsets *tench, English springer, cassette player, chain saw,
church, French horn, garbage truck, gas pump, golf ball,* and *parachute*. The
dataset provides **training** and **validation** splits only; there is no separate
held-out test split, and the validation split is used for all robustness reporting.
The full splits are used: the validation split contains $\approx 3{,}925$ images
(`val_subset_size` $= 3925$, capped to the available count) and the training split
$\approx 9{,}469$ images (`train_subset_size` $= 9469$). Images are read via
`torchvision.datasets.ImageFolder`.

Because the backbone is pretrained on the full ImageNet-1k label space, each
ImageNette synset is remapped at load time from its ten-way `ImageFolder` index to
its corresponding ImageNet-1k class index (e.g. *tench* $\rightarrow 0$, *English
springer* $\rightarrow 217$, *church* $\rightarrow 497$, *parachute* $\rightarrow
701$). This allows the $1000$-way pretrained classifier to be evaluated directly,
without replacing the classification head.

**Input resolution and normalization.** All inputs are $224 \times 224$ and are
normalized with the standard ImageNet statistics, per channel
$\mu = (0.485, 0.456, 0.406)$ and $\sigma = (0.229, 0.224, 0.225)$.

**Augmentation per phase.**
- *Clean evaluation (Phases 1–3):* `Resize(256)` $\rightarrow$ `CenterCrop(224)`
  $\rightarrow$ `ToTensor` $\rightarrow$ `Normalize`. No stochastic augmentation.
- *Defense fine-tuning (AT and AT+KD):* `RandomResizedCrop(224)` $\rightarrow$
  `RandomHorizontalFlip` $\rightarrow$ `ToTensor` $\rightarrow$ `Normalize`. AT and
  AT+KD use identical augmentation.

**Determinism and subsets.** A global seed of $42$ is set for `random`, `numpy`, and
`torch` (all CUDA devices) at the start of every script; subsets are drawn with a
seeded `torch.randperm`, making image ordering reproducible across runs and phases.
The patch and combined attacks — which require many gradient-ascent steps per batch
($\approx 7.5\times$ the cost of FGSM/PGD) — are evaluated on a fixed prefix of the
first $500$ validation images from the same permutation, i.e. a strict subset of the
full evaluation set, with clean accuracy recomputed on that subset.

## 3.2 Model Architecture

The model under study is **DeiT-S** (`deit_small_patch16_224`), a data-efficient
Vision Transformer. It is instantiated from `timm` with ImageNet-1k pretrained
weights and placed in evaluation mode prior to inference.

| Specification | Value |
|---|---|
| Variant | DeiT-S (`deit_small_patch16_224`) |
| Pretraining | ImageNet-1k (via `timm`, `pretrained=True`) |
| Input resolution | $224 \times 224$ |
| Patch size | $16 \times 16$ |
| Transformer blocks | $12$ |
| Parameters | $\approx 22\text{M}$ |
| Output classes | $1000$ |

All compression levels are built from this single `timm` model: the quantized
variants are produced by replacing the model's linear layers in place (Section 3.3),
not by loading a separate checkpoint. Model outputs are routed through a thin
`LogitsWrapper` that returns a plain $(N, C)$ logits tensor regardless of whether the
underlying module returns a raw tensor or a dataclass, providing a uniform interface
for the attack library and the evaluation loop. DeiT-B is excluded to remain within
the GPU compute budget; DeiT-S is used uniformly as attacker, defender, and teacher.

## 3.3 Post-Training Quantization Setup

Three compression levels are studied — **FP32**, **INT8**, and **INT4** — all
declared as compression levels in the configuration and evaluated in every phase. All
compression is *post-training* (no quantization-aware training), and only the
**linear (weight) layers** are quantized; activations are not quantized, and
LayerNorm, embeddings, and the patch-projection convolution are left in their native
precision. Quantization is **data-free**: no calibration dataset is used, since the
NF4 scheme quantizes pretrained weights directly via block-wise normalization. The
quantization backend is `bitsandbytes`.

- **FP32 (baseline).** Full-precision pretrained DeiT-S; establishes the
  clean-accuracy and robustness upper bound and serves as the frozen teacher in
  AT+KD.

- **INT8.** Genuine 8-bit weight quantization via `bitsandbytes` `Linear8bitLt`. Every
  `nn.Linear` layer is replaced in place with a `bnb.nn.Linear8bitLt` layer
  (`has_fp16_weights=False`); weights are stored as true 8-bit integers and matrix
  multiplications use the cuBLAS-LT 8-bit kernel. The replacement is performed on CPU
  before the model is moved to CUDA, which triggers bitsandbytes to quantize the
  weights on the first `.cuda()` call. This path requires CUDA compute capability
  $\geq 7.0$ (`sm_70`); the NVIDIA Tesla T4 is `sm_75` and fully supports it.

- **INT4 (NF4).** Genuine 4-bit **NormalFloat-4 (NF4)** quantization via
  `bitsandbytes`. Every `nn.Linear` layer is replaced in place with a
  `bnb.nn.Linear4bit` layer storing weights as `Params4bit` in NF4 format with FP16
  compute dtype (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16`). NF4
  is the quantization scheme popularized by QLoRA (Dettmers et al., 2023). All
  quantized tensors have gradients disabled.

**Compression ratio.** Relative to FP32, weight precision is reduced by $\approx
2\times$ for INT8 (32-bit $\rightarrow$ 8-bit) and $\approx 8\times$ for NF4 INT4
(32-bit $\rightarrow$ 4-bit). The repository measures parameter footprint via
`get_model_size_mb()` but does not log a stored on-disk compression ratio.
[EXACT ON-DISK COMPRESSION RATIO NOT FOUND IN CODE]

## 3.4 Threat Model

All attacks are **untargeted, white-box** attacks: the adversary has full access to
the model under evaluation (including compressed and/or defended weights) and
backpropagates through it. The white-box setting is adopted as a worst-case adversary
that yields a conservative (lower-bound) estimate of robustness; in Phase 2 the
attack is constructed against the defended model itself, making the evaluation
adaptive with respect to the defended weights.

All perturbations are realized in **pixel space**. Inputs are ImageNet-normalized, so
each attack is configured with `set_normalization_used(mean, std)` (for `torchattacks`)
or performs explicit un-normalization (for the custom patch attack): the attack
un-normalizes to $[0, 1]$, applies and clips the perturbation, and re-normalizes
before the forward pass, guaranteeing the realized $\ell_\infty$ budget equals
$\epsilon$ in $[0, 1]$ space. A pre-training sanity check asserts the measured
pixel-space $\ell_\infty$ is within $\pm 10\%$ of $\epsilon$.

- **FGSM.** Single-step $\ell_\infty$ attack (`torchattacks.FGSM`):
  $x_{\text{adv}} = \mathrm{clip}_{[0,1]}\!\big(x + \epsilon\,\mathrm{sign}(\nabla_x \mathcal{L}(f(x), y))\big)$,
  with $\epsilon = 8/255 \approx 0.03137$.

- **PGD.** Iterative $\ell_\infty$ attack (`torchattacks.PGD`):
  $x^{(t+1)} = \Pi_{\mathcal{B}_\infty(x,\epsilon)}\!\big(x^{(t)} + \alpha\,\mathrm{sign}(\nabla_x \mathcal{L}(f(x^{(t)}), y))\big)$,
  with $\epsilon = 8/255 \approx 0.03137$, step size $\alpha = 2/255 \approx 0.00784$,
  and $20$ steps.

- **Adversarial Patch.** A custom localized attack optimizing a $32 \times 32$ pixel
  patch ($\approx 2\%$ of image area) at a uniformly random location, via $150$
  PGD-style sign-gradient ascent steps with step size $\eta = 0.05$, maximizing
  cross-entropy. The patch is unbounded within its support (clipped only to $[0,1]$).

- **AutoAttack.** **Not implemented in this codebase.** The requested AutoAttack
  benchmark is absent; robustness is reported under FGSM, PGD, the patch attack, and
  the combined attack below. [AUTOATTACK NOT FOUND IN CODE]

**Phase 3 combined attack.** The combined attack chains the three attacks
**sequentially** — FGSM $\rightarrow$ PGD $\rightarrow$ Patch — with the adversarial
output of each stage feeding the next and all stages sharing the ground-truth labels.
It is the strongest perturbation considered and is evaluated on the $500$-image subset.

## 3.5 Adversarial Training — Phase 2a

Phase 2a applies **Adversarial Training (AT)** to the already-compressed model. The
defense is explicitly applied **after** quantization (compress-first-then-defend),
and the quantized weights are **frozen** throughout: only floating-point parameters
are trainable, so the packed integer NF4 weights cannot receive gradient updates.

The AT variant used is **single-step (FGSM-based) adversarial training** — not
PGD-AT or TRADES. At each step, adversarial inputs are generated with FGSM at the
training budget $\epsilon_{\text{train}} = 8/255 \approx 0.03137$ (matching the
evaluation budget) and the model is optimized under a standard cross-entropy loss:
$$
\mathcal{L}_{\text{AT}} = \mathrm{CE}\!\big(f_\theta(x_{\text{adv}}),\, y\big),
\qquad x_{\text{adv}} = \mathrm{FGSM}(f_\theta, x, y;\, \epsilon_{\text{train}}).
$$

To protect fragile quantized weights and stay within budget, fine-tuning is
**parameter-efficient**: the backbone is frozen and only the **last four transformer
blocks** and the **classifier head** are updated. Optimization settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay $0.01$ |
| Epochs | $7$ |
| Batch size | $16$ |
| LR schedule | linear warmup: epoch 1 at $\mathrm{lr}/10$, full LR from epoch 2 |
| Base LR (FP32) | $1 \times 10^{-5}$ |
| Base LR (INT8) | $5 \times 10^{-6}$ |
| Base LR (INT4) | $1 \times 10^{-6}$ |
| Training $\epsilon$ | $8/255 \approx 0.03137$ |
| Checkpointing | every epoch |

Per-compression learning rates are used because AdamW with layer freezing requires
smaller steps than full fine-tuning, and more aggressively quantized weights are more
fragile.

## 3.6 Adversarial Training with Knowledge Distillation — Phase 2b

Phase 2b augments AT with **knowledge distillation (KD)** from a teacher. The teacher
is the **full-precision (FP32) pretrained DeiT-S**, i.e. it is *not* adversarially
trained; it is the clean, uncompressed model. The teacher is held in `eval()` mode
with gradients disabled, every teacher forward pass is wrapped in `torch.no_grad()`,
and the frozen state is re-asserted each epoch, so the teacher is never updated.

Both teacher and student are queried on the **same adversarial inputs** ($x_{\text{adv}}$
generated with FGSM against the student) — the teacher is *not* queried on clean
inputs. The loss combines hard-label cross-entropy with a temperature-scaled KL
divergence (Hinton et al., 2015):
$$
\mathcal{L}_{\text{AT+KD}}
= \alpha \cdot \mathrm{CE}\!\big(f_S(x_{\text{adv}}),\, y\big)
+ (1-\alpha)\cdot \tau^2 \cdot \mathrm{KL}\!\Big(\sigma\!\big(\tfrac{z_S}{\tau}\big) \,\big\|\, \sigma\!\big(\tfrac{z_T}{\tau}\big)\Big),
$$
where $z_S, z_T$ are the student and teacher logits on $x_{\text{adv}}$,
$\sigma$ is the softmax, the temperature is $\tau = 4.0$, and the cross-entropy
weight is $\alpha = 0.5$ (KL weight $1-\alpha = 0.5$). The $\tau^2$ factor restores
the gradient magnitude attenuated by temperature scaling. All optimizer, schedule,
epoch, batch-size, learning-rate, layer-freezing, and checkpointing settings are
**identical to Phase 2a** (Section 3.5).

## 3.7 Experimental Pipeline and Design Rationale

The pipeline is fixed as:
$$
\text{Pretrained DeiT-S (FP32)} \;\rightarrow\; \text{Compress (PTQ)} \;\rightarrow\; \text{Defend (AT / AT+KD)} \;\rightarrow\; \text{Attack} \;\rightarrow\; \text{Measure}.
$$
We deliberately **compress first, then defend**. This reflects edge deployment, where
a model is quantized to satisfy on-device memory and latency constraints *before* any
robustness hardening, and where the defended artifact must itself be the compressed
model. The contrasting *defend-then-compress* ordering — e.g. quantization-aware
adversarial training, where robust weights are learned and then quantized — is
ill-suited to this scenario: it presumes access to the full training pipeline at the
target precision and risks the quantization step degrading the robustness that
training installed, so the deployed (compressed) model is no longer the one whose
robustness was validated. Studying robustness recovery *on the already-compressed
model* therefore matches the operational constraint.

The full experimental grid is $\{\text{FP32}, \text{INT8}, \text{INT4}\} \times
\{\text{Undefended}, \text{AT}, \text{AT+KD}\}$, each evaluated under multiple
attacks:

| Compression \ Defense | Undefended (Phase 1) | AT (Phase 2a) | AT+KD (Phase 2b) |
|---|---|---|---|
| FP32 | FGSM, PGD, Patch | FGSM, PGD, Patch | FGSM, PGD, Patch |
| INT8 | FGSM, PGD, Patch | FGSM, PGD, Patch | FGSM, PGD, Patch |
| INT4 | FGSM, PGD, Patch | FGSM, PGD, Patch | FGSM, PGD, Patch |

Phase 3 additionally evaluates the **combined attack** against all nine
$\{\text{compression}\} \times \{\text{none}, \text{AT}, \text{AT+KD}\}$ cells. In
total the study comprises $36$ evaluated configurations.

## 3.8 Evaluation Protocol and Metrics

For every configuration we record four metrics, all in $[0,1]$ and computed from
top-1 `argmax` predictions:

| Metric | Definition |
|---|---|
| Clean accuracy ($\mathrm{acc}_{\text{clean}}$) | top-1 accuracy on clean inputs |
| Robust accuracy ($\mathrm{acc}_{\text{rob}}$) | top-1 accuracy on adversarial inputs |
| Attack success rate (ASR) | $1 - \mathrm{acc}_{\text{rob}}$ (untargeted) |
| Robustness gap | $\mathrm{acc}_{\text{clean}} - \mathrm{acc}_{\text{rob}}$ |

Accuracy degradation relative to the FP32 baseline is reported as
$\Delta = \mathrm{acc}^{\text{FP32}} - \mathrm{acc}^{\text{quant}}$ for the
corresponding (defense, attack) cell, capturing the clean- and robust-accuracy cost
of compression. Clean accuracy is computed once per (compression, defense) model and
reused across attacks on the full validation set; for the patch and combined attacks
it is recomputed on the $500$-image subset so that all metrics share a common
denominator. Each result row also records a UTC timestamp, model, compression,
defense, attack, and phase; rows are written incrementally and each script skips
already-completed configurations, making every phase resumable.

**AutoAttack as trustworthy benchmark.** AutoAttack is *not* used in this study (it
is not implemented); the strongest evaluation is the sequential combined attack.
**Variance / seeds.** Each configuration is evaluated **once** on the fixed,
seed-$42$ validation subset; no multi-seed runs are performed and no variance is
reported. Determinism makes results exactly reproducible given identical weights and
environment. [MULTI-SEED VARIANCE NOT FOUND IN CODE]

## 3.9 Implementation Details

The implementation uses **PyTorch** (target build `torch==2.3.1+cu121`,
`torchvision==0.18.1+cu121`) on **CUDA 12.1**. Models are instantiated with
`timm` ($\geq 0.9.0$); attacks use `torchattacks` ($\geq 3.4.0$) for FGSM and PGD,
with the patch and combined attacks implemented in-repository; quantization uses
`bitsandbytes` ($\geq 0.41.0$) for NF4 INT4; configuration is loaded from YAML via
`PyYAML`. Evaluation uses a batch size of $32$ with two DataLoader workers; defense
fine-tuning uses a batch size of $16$.

The experiments were run on **Kaggle Notebooks** using a **dual NVIDIA Tesla T4
($2\times$ T4, 16 GB VRAM each)** accelerator under Kaggle's free-tier weekly
GPU-hour budget. Both INT8 (`Linear8bitLt`) and INT4 (NF4 `Linear4bit`) use genuine
`bitsandbytes` quantization. All randomness is seeded with $42$
(`random`, `numpy`, `torch`, and all CUDA devices). The code is available at
**https://github.com/Jmanav/ADVC**.

## References

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015. arXiv:1412.6572.
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). *Training Data-Efficient Image Transformers & Distillation Through Attention*. ICML 2021.
