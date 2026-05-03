"""
defenses/at_kd.py

Adversarial Training + Knowledge Distillation (AT+KD) — fine-tunes an
already-compressed student model using FGSM adversarial inputs combined with
soft-label supervision from a frozen FP32 teacher.

Loss formula (Hinton et al., 2015 + adversarial training):
    adv_images   = FGSM(student, images, labels, eps=at_eps)
    teacher_soft = softmax(teacher(adv_images) / T)   # no gradients ever
    student_soft = softmax(student(adv_images) / T)

    loss = alpha       * CrossEntropy(student(adv_images), true_labels)
         + (1 - alpha) * T² * KLDiv(log(student_soft), teacher_soft)

The T² factor (Hinton 2015) preserves gradient magnitude under temperature
scaling and is standard practice in KD.

All parameters come from configs/base.yaml — never hardcode values here.
Compression must be applied before calling this module.
Teacher must never be fine-tuned — it stays frozen for the entire run.

Usage:
    from defenses.at_kd import at_kd_train
    hardened_student = at_kd_train(student, teacher, train_loader, config)
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Resolve project root so sibling packages import cleanly whether this module
# is imported from the project root or from defenses/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torchattacks
from models.loader import load_config  # noqa: F401 — re-exported for convenience


class _LogitsWrapper(nn.Module):
    """Unwrap HuggingFace ImageClassifierOutput to a plain (N, C) tensor.

    torchattacks expects model(x) to return a plain tensor.  INT8/INT4 models
    loaded via HuggingFace return a dataclass with a .logits attribute.  This
    thin wrapper makes both cases identical so FGSM can compute gradients.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Perturbation sanity check
# ---------------------------------------------------------------------------

def _check_fgsm_perturbation(
    fgsm: torchattacks.FGSM,
    train_loader: torch.utils.data.DataLoader,
    at_eps: float,
    model_device: str,
    mean: list,
    std: list,
) -> None:
    """Assert that the FGSM L-inf perturbation in pixel space is within 10% of at_eps.

    Grabs one batch from train_loader, generates adversarial examples, then
    un-normalises both clean and adversarial images to [0, 1] pixel space.
    Asserts that the L-inf of (adv − clean) lies in [at_eps * 0.9, at_eps * 1.1].

    This check fires before any training epoch.  If it fails a ValueError is
    raised immediately so no compute is wasted on a misconfigured run.

    Args:
        fgsm:         torchattacks.FGSM instance with set_normalization_used
                      already called.
        train_loader: DataLoader yielding ImageNet-normalised (mean/std) images.
        at_eps:       Configured epsilon (defense.at_eps from base.yaml).
        model_device: Device string for moving tensors to match the model.
        mean:         ImageNet normalisation mean — 3-element list.
        std:          ImageNet normalisation std  — 3-element list.

    Raises:
        ValueError: If the measured L-inf is outside at_eps ± 10%.
    """
    print("[AT+KD] Running FGSM perturbation sanity check …")

    images, labels = next(iter(train_loader))
    images = images.to(model_device)
    labels = labels.to(model_device)

    adv_images = fgsm(images, labels)

    # Un-normalise to [0, 1] pixel space for measurement.
    # In normalised space the perturbation is scaled by 1/std per channel, so
    # the L-inf of (adv_norm − clean_norm) is NOT eps.  Measuring in pixel
    # space gives the true L-inf that must equal eps.
    mean_t = torch.tensor(mean, dtype=images.dtype, device=model_device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=images.dtype, device=model_device).view(1, 3, 1, 1)
    images_px = (images     * std_t + mean_t).clamp(0.0, 1.0)
    adv_px    = (adv_images * std_t + mean_t).clamp(0.0, 1.0)

    linf = (adv_px - images_px).abs().max().item()

    lo = at_eps * 0.9
    hi = at_eps * 1.1
    print(
        f"[AT+KD] Perturbation L-inf (pixel space) : {linf:.5f}  "
        f"(expected {at_eps:.5f} ± 10%  →  [{lo:.5f}, {hi:.5f}])"
    )

    if not (lo <= linf <= hi):
        raise ValueError(
            f"FGSM perturbation sanity check FAILED — training aborted.\n"
            f"  Measured L-inf (pixel space) : {linf:.5f}\n"
            f"  Expected range               : [{lo:.5f}, {hi:.5f}]\n"
            f"  Configured at_eps            : {at_eps:.5f}  "
            f"({round(at_eps * 255)}/255)\n"
            "  Likely cause: set_normalization_used() was not called on the\n"
            "  FGSM attack, so perturbations were applied in normalised space\n"
            "  (~[-2.1, 2.6]) instead of pixel space ([0, 1]).  Ensure\n"
            "  fgsm.set_normalization_used(mean, std) is called after building\n"
            "  the attack."
        )

    print("[AT+KD] Perturbation sanity check PASSED.\n")


# ---------------------------------------------------------------------------
# Clean accuracy helper
# ---------------------------------------------------------------------------

def _measure_clean_acc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """Measure clean accuracy on the given loader.

    Switches model to eval mode for measurement, then restores to train mode.

    Args:
        model:  The model to evaluate.
        loader: DataLoader yielding (images, labels) batches.
        device: Device string to move tensors to.

    Returns:
        Accuracy in [0, 1].
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if hasattr(logits, "logits"):
                logits = logits.logits
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if was_training:
        model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    compression: str,
    checkpoint_dir: str,
) -> str:
    """Save student model checkpoint after a training epoch.

    INT4 models (bitsandbytes NF4) embed absmax, quant_map, and quant_state
    metadata directly in their parameter tensors.  Saving only the state_dict
    and reloading it into a freshly-quantised model causes a state conflict
    because the quantisation metadata is regenerated on load.  To avoid this,
    INT4 checkpoints save the full model object.

    FP32 and INT8 models are saved as plain state_dicts (lighter, portable).

    Filename convention:
      INT4  →  atkd_{compression}_epoch{epoch:02d}_full_model.pt
      other →  atkd_{compression}_epoch{epoch:02d}.pt

    Args:
        model:          The fine-tuned student model.
        epoch:          1-based epoch index.
        compression:    Compression level string, e.g. "fp32", "int8", "int4".
        checkpoint_dir: Directory path for AT+KD checkpoints (from base.yaml).

    Returns:
        Absolute path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if compression == "int4":
        filename = f"atkd_{compression}_epoch{epoch:02d}_full_model.pt"
        ckpt_path = os.path.join(checkpoint_dir, filename)
        torch.save(model, ckpt_path)
    else:
        filename = f"atkd_{compression}_epoch{epoch:02d}.pt"
        ckpt_path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), ckpt_path)
    return os.path.abspath(ckpt_path)


# ---------------------------------------------------------------------------
# Layer-freeze helper
# ---------------------------------------------------------------------------

def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers, then unfreeze the last 4 transformer blocks and head.

    Handles HuggingFace ViT models (model.vit.encoder.layer / model.classifier),
    HuggingFace models with model.encoder.layer, and timm FP32 models
    (model.blocks / model.head).

    Only float parameters (fp32, fp16, bf16) have requires_grad set — integer
    quantised weights from bitsandbytes cannot carry gradients and are skipped.

    Args:
        model: The compressed student model whose backbone should be frozen.

    Raises:
        ValueError: If the model architecture cannot be detected.
    """
    # Detect architecture and get the list of transformer blocks.
    # Try HuggingFace ViT structure first (INT8/INT4 via bitsandbytes).
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        blocks = model.vit.encoder.layer
    # Fall back to HuggingFace models with top-level encoder.
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        blocks = model.encoder.layer
    # timm DeiT-S FP32: model.blocks is a Sequential of 12 transformer blocks.
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        raise ValueError(
            "[AT+KD] Cannot detect model architecture for layer freezing.  "
            "Expected model.vit.encoder.layer, model.encoder.layer, or model.blocks."
        )

    # Freeze all float params first — integer quantised params are left alone.
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad = False

    # Unfreeze last 4 blocks — float params only.
    for block in list(blocks)[-4:]:
        for param in block.parameters():
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                param.requires_grad = True

    # Unfreeze classifier head — float params only.
    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[AT+KD] Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def at_kd_train(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict,
    compression: str = "fp32",
) -> nn.Module:
    """Fine-tune a compressed student with FGSM adversarial inputs and KD loss.

    Training protocol:
      - Backbone frozen; only last 4 transformer blocks + classifier head trained.
      - Optimizer : AdamW (weight_decay from config["defense"])
      - LR schedule: linear warmup — lr/10 for epoch 1, full lr from epoch 2.
      - Loss       : alpha * CE(hard labels) + (1-alpha) * T² * KL(soft labels).
      - Epochs     : config["defense"]["epochs"]  (7)
      - Checkpoint saved after every epoch to config["paths"]["checkpoints_atkd_dir"]
      - Clean-acc drop > 15% triggers a printed WARNING; training continues.

    The teacher is a frozen FP32 DeiT-S that provides soft-label supervision.
    It must already be loaded and moved to the appropriate device before this
    call.  This function enforces teacher.eval() and wraps every teacher
    forward pass in torch.no_grad() — the teacher is NEVER updated.

    Args:
        student:      Already-compressed nn.Module to fine-tune.
                      Must be on the correct device before this call.
        teacher:      Frozen FP32 nn.Module used for soft-label targets.
                      Must be on the correct device before this call.
        train_loader: DataLoader over the training subset (batch_size=32).
        config:       Parsed base.yaml config dict.
        compression:  Compression level of the student; used for checkpoint names
                      and per-compression LR lookup.  One of "fp32", "int8", "int4".

    Returns:
        student: The same nn.Module, fine-tuned in place, returned in eval mode.
    """
    _set_seeds(config["seed"])

    defense_cfg = config["defense"]
    at_kd_cfg   = config["at_kd"]
    ds_cfg      = config["dataset"]
    ckpt_dir = config["paths"]["checkpoints_atkd_dir"]

    epochs: int            = defense_cfg["epochs"]
    weight_decay: float    = defense_cfg["weight_decay"]
    at_eps: float          = defense_cfg["at_eps"]
    warmup_epochs: int     = int(defense_cfg.get("warmup_epochs", 1))
    save_every_epoch: bool = defense_cfg.get("save_every_epoch", True)
    mean: list             = ds_cfg["mean"]
    std: list              = ds_cfg["std"]

    temperature: float = at_kd_cfg["temperature"]
    alpha: float       = at_kd_cfg["alpha"]       # weight for CE loss
    kd_weight: float   = 1.0 - alpha              # weight for KL loss

    # Per-compression learning rate — same config key as AT.
    # AdamW + layer freezing requires much lower LRs than full SGD fine-tune.
    # INT8/INT4 quantised weights are especially fragile.
    lr_cfg = defense_cfg["lr"]
    if isinstance(lr_cfg, dict):
        if compression not in lr_cfg:
            raise KeyError(
                f"[AT+KD] No LR configured for compression='{compression}'.  "
                f"Add a '{compression}' key under defense.lr in base.yaml.  "
                f"Available keys: {list(lr_cfg.keys())}"
            )
        lr: float = float(lr_cfg[compression])
    else:
        lr = float(lr_cfg)

    # Enforce teacher is frozen — eval mode, no gradient accumulation.
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device

    # Build FGSM attack bound to the student.
    # set_normalization_used() is mandatory: training images are
    # ImageNet-normalised (range ≈ [-2.1, 2.6]).  Without it torchattacks
    # clamps normalised values to [0, 1], producing effective perturbations
    # of ~2.1 in pixel space instead of the intended 8/255 ≈ 0.031.
    # With it, torchattacks un-normalises internally, applies eps in [0,1]
    # pixel space, then re-normalises before returning.
    #
    # Use _LogitsWrapper so INT8/INT4 HuggingFace models (which return a
    # dataclass) expose a plain tensor interface to torchattacks.
    fgsm = torchattacks.FGSM(_LogitsWrapper(student), eps=at_eps)
    fgsm.set_normalization_used(mean=mean, std=std)

    # Freeze backbone — only last 4 blocks + head will receive gradient updates.
    _freeze_backbone(student)

    # Build optimizer over trainable params only.
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Linear warmup: epoch 1 runs at lr/10, full lr from epoch 2 onward.
    def _warmup_lambda(epoch_idx: int) -> float:
        return 0.1 if epoch_idx < warmup_epochs else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    ce_criterion = nn.CrossEntropyLoss()
    # KLDivLoss expects log-probabilities as input, probabilities as target.
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    print(f"[AT+KD] Starting adversarial training with KD — {epochs} epoch(s)")
    print(f"[AT+KD] compression    : {compression}")
    print(f"[AT+KD] student device : {student_device}")
    print(f"[AT+KD] teacher device : {teacher_device}")
    print(f"[AT+KD] optimizer      : AdamW  lr={lr}  weight_decay={weight_decay}")
    print(f"[AT+KD] warmup_epochs  : {warmup_epochs}  (epoch 1 uses lr={lr * 0.1:.2e})")
    print(f"[AT+KD] at_eps         : {at_eps:.5f}  ({round(at_eps * 255)}/255)")
    print(f"[AT+KD] temperature    : {temperature}")
    print(f"[AT+KD] alpha (CE)     : {alpha}   kd_weight (KL) : {kd_weight}")
    print(f"[AT+KD] checkpoints    : {ckpt_dir}")
    print()

    # Verify FGSM produces correct perturbations BEFORE any epoch runs.
    # Raises ValueError immediately if L-inf is outside at_eps ± 10%.
    _check_fgsm_perturbation(fgsm, train_loader, at_eps, str(student_device), mean, std)

    # Measure baseline clean accuracy before any weight updates.
    # Use a fixed 500-image subset — the full train loader (10 000 images) hangs
    # for 10+ minutes; 500 images give a reliable estimate in a few seconds.
    print("[AT+KD] Measuring baseline clean accuracy (500-image subset) …")
    baseline_loader = DataLoader(
        Subset(train_loader.dataset, range(500)),
        batch_size=64,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )
    baseline_clean_acc = _measure_clean_acc(student, baseline_loader, str(student_device))
    print(f"[AT+KD] Baseline clean_acc : {baseline_clean_acc:.4f}\n")

    _first_batch_checked = False

    for epoch in range(1, epochs + 1):
        student.train()
        # Re-enforce teacher frozen state at the start of every epoch.
        teacher.eval()

        # Show effective LR for this epoch (after LambdaLR scaling).
        current_lr = scheduler.get_last_lr()[0] if epoch > 1 else lr * _warmup_lambda(0)
        print(f"[AT+KD] Epoch {epoch}/{epochs} — effective lr={current_lr:.2e}")

        running_loss = 0.0
        running_ce   = 0.0
        running_kl   = 0.0
        correct = 0
        total = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for images, labels in loop:
            images = images.to(student_device)
            labels = labels.to(student_device)

            # Image range check fires exactly once, before any gradient step.
            if not _first_batch_checked:
                _first_batch_checked = True
                print(
                    f"\n[AT+KD] Image range confirmed: "
                    f"min={images.min():.3f} max={images.max():.3f} "
                    f"(ImageNet normalized — expected)\n"
                )

            # Generate adversarial examples using the student.
            # torchattacks temporarily sets model.eval() internally, then
            # restores training mode via model.train() after generation.
            adv_images = fgsm(images, labels)

            # ── Teacher soft targets (no gradients, ever) ─────────────────────
            teacher_images = adv_images.to(teacher_device)
            with torch.no_grad():
                teacher_out = teacher(teacher_images)
                teacher_logits = (
                    teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out
                ).to(student_device)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

            # ── Student forward pass ──────────────────────────────────────────
            optimizer.zero_grad()
            student_out = student(adv_images)
            student_logits = student_out.logits if hasattr(student_out, "logits") else student_out

            # ── CE loss on hard labels ────────────────────────────────────────
            ce_loss = ce_criterion(student_logits, labels)

            # ── KL divergence loss on soft targets ────────────────────────────
            # Scale by T² to preserve gradient magnitude (Hinton et al. 2015).
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
            kl_loss = kl_criterion(student_log_soft, teacher_soft) * (temperature ** 2)

            # ── Combined loss ─────────────────────────────────────────────────
            loss = alpha * ce_loss + kd_weight * kl_loss
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_ce   += ce_loss.item() * batch_size
            running_kl   += kl_loss.item() * batch_size
            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix(
                loss=f"{running_loss / total:.4f}",
                ce=f"{running_ce / total:.4f}",
                kl=f"{running_kl / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

        # Step scheduler at end of each epoch (drives warmup → full lr).
        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        print(
            f"[AT+KD] Epoch {epoch}/{epochs} — "
            f"loss={epoch_loss:.4f}  "
            f"ce={running_ce / total:.4f}  "
            f"kl={running_kl / total:.4f}  "
            f"train_adv_acc={epoch_acc:.4f}"
        )

        # Measure clean accuracy after the epoch and compare to baseline.
        epoch_clean_acc = _measure_clean_acc(student, train_loader, str(student_device))
        clean_drop = baseline_clean_acc - epoch_clean_acc
        print(
            f"[AT+KD] Epoch {epoch} clean_acc={epoch_clean_acc:.4f}  "
            f"(baseline={baseline_clean_acc:.4f}  drop={clean_drop:+.4f})"
        )

        if clean_drop > 0.15:
            print(
                f"\n[AT+KD] *** WARNING: clean_acc dropped {clean_drop:.4f} "
                f"(> 0.15 threshold) after epoch {epoch}. ***\n"
                f"[AT+KD] Continuing training — saving checkpoint for all epochs.\n"
                f"[AT+KD] Current lr={current_lr:.2e}  compression={compression}"
            )

        if save_every_epoch:
            ckpt_path = save_checkpoint(student, epoch, compression, ckpt_dir)
            print(f"[AT+KD] Checkpoint saved → {ckpt_path}")

    student.eval()
    print(f"\n[AT+KD] Training complete.  Student returned in eval mode.")
    return student


# ---------------------------------------------------------------------------
# Sanity check — run directly to verify the training loop executes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from models.loader import load_model

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Sanity check: AT+KD on DeiT-S FP32 student (2-epoch smoke test) ===\n")

    ds_cfg = cfg["dataset"]
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])

    train_dataset = ImageFolder(
        root=str(_ROOT / ds_cfg["train_dir"]),
        transform=transform,
    )
    # Tiny 64-sample subset so the smoke test finishes quickly.
    subset = Subset(train_dataset, list(range(64)))
    loader = DataLoader(
        subset,
        batch_size=cfg["defense"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Load FP32 model as both teacher and student (smoke test only).
    teacher = load_model("deit_small", "fp32", cfg, device=device)
    student = load_model("deit_small", "fp32", cfg, device=device)

    # Override epochs to 2 without mutating the shared config dict.
    smoke_cfg = {**cfg, "defense": {**cfg["defense"], "epochs": 2}}

    student = at_kd_train(student, teacher, loader, smoke_cfg, compression="fp32")
    print("\n[AT+KD] Sanity check passed.")
