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
from tqdm import tqdm

# Resolve project root so sibling packages import cleanly.
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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    compression: str,
    checkpoint_dir: str,
) -> str:
    """Save student model state dict after a training epoch.

    Args:
        model:          The fine-tuned student model.
        epoch:          1-based epoch index.
        compression:    Compression level string, e.g. "fp32", "int8", "int4".
        checkpoint_dir: Directory path for AT+KD checkpoints (from base.yaml).

    Returns:
        Absolute path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    filename = f"atkd_{compression}_epoch{epoch:02d}.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), ckpt_path)
    return os.path.abspath(ckpt_path)


# ---------------------------------------------------------------------------
# Logits extraction helper
# ---------------------------------------------------------------------------

def _get_logits(output) -> torch.Tensor:
    """Extract a plain (N, C) logits tensor from model output.

    timm models return a plain tensor; HuggingFace models (quantized via
    transformers) return a dataclass with a .logits attribute.
    """
    if hasattr(output, "logits"):
        return output.logits
    return output


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
        compression:  Compression level of the student; used only for naming
                      checkpoint files.  One of "fp32", "int8", "int4".

    Returns:
        student: The same nn.Module, fine-tuned in place, returned in eval mode.
    """
    _set_seeds(config["seed"])

    defense_cfg = config["defense"]
    at_kd_cfg = config["at_kd"]
    ckpt_dir = config["paths"]["checkpoints_atkd_dir"]

    epochs: int = defense_cfg["epochs"]
    lr: float = defense_cfg["lr"]
    momentum: float = defense_cfg["momentum"]
    weight_decay: float = defense_cfg["weight_decay"]
    at_eps: float = defense_cfg["at_eps"]
    save_every_epoch: bool = defense_cfg.get("save_every_epoch", True)

    temperature: float = at_kd_cfg["temperature"]
    alpha: float = at_kd_cfg["alpha"]          # weight for CE loss
    kd_weight: float = 1.0 - alpha             # weight for KL loss

    # Enforce teacher is frozen — eval mode, no gradient accumulation.
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device

    # FGSM attack is bound to the student (which will be in train mode during
    # the loop; torchattacks handles the eval/train mode switching internally).
    # Use _LogitsWrapper so INT8/INT4 HuggingFace models (which return a
    # dataclass) expose a plain tensor interface to torchattacks.
    fgsm = torchattacks.FGSM(_LogitsWrapper(student), eps=at_eps)

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    ce_criterion = nn.CrossEntropyLoss()
    # KLDivLoss expects log-probabilities as input, probabilities as target.
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    print(f"[AT+KD] Starting adversarial training with KD — {epochs} epoch(s)")
    print(f"[AT+KD] compression    : {compression}")
    print(f"[AT+KD] student device : {student_device}")
    print(f"[AT+KD] teacher device : {teacher_device}")
    print(f"[AT+KD] lr={lr}  momentum={momentum}  weight_decay={weight_decay}")
    print(f"[AT+KD] at_eps         : {at_eps:.5f}  ({round(at_eps * 255)}/255)")
    print(f"[AT+KD] temperature    : {temperature}")
    print(f"[AT+KD] alpha (CE)     : {alpha}   kd_weight (KL) : {kd_weight}")
    print(f"[AT+KD] checkpoints    : {ckpt_dir}")
    print()

    for epoch in range(1, epochs + 1):
        student.train()
        # Re-enforce teacher frozen state at the start of every epoch.
        teacher.eval()

        running_loss = 0.0
        running_ce = 0.0
        running_kl = 0.0
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

            # ── Generate adversarial examples using the student ───────────────
            # torchattacks handles eval/train toggling internally.
            adv_images = fgsm(images, labels)

            # ── Teacher soft targets (no gradients, ever) ─────────────────────
            teacher_images = adv_images.to(teacher_device)
            with torch.no_grad():
                teacher_out = teacher(teacher_images)
                teacher_logits = _get_logits(teacher_out).to(student_device)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

            # ── Student forward pass ──────────────────────────────────────────
            optimizer.zero_grad()
            student_out = student(adv_images)
            student_logits = _get_logits(student_out)

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
            running_ce += ce_loss.item() * batch_size
            running_kl += kl_loss.item() * batch_size
            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix(
                loss=f"{running_loss / total:.4f}",
                ce=f"{running_ce / total:.4f}",
                kl=f"{running_kl / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"[AT+KD] Epoch {epoch}/{epochs} — "
            f"loss={epoch_loss:.4f}  "
            f"ce={running_ce / total:.4f}  "
            f"kl={running_kl / total:.4f}  "
            f"train_acc={epoch_acc:.4f}"
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
