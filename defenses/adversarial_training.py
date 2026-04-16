"""
defenses/adversarial_training.py

Adversarial Training (AT) — fine-tunes an already-compressed model on FGSM
adversarial inputs for a fixed number of epochs.

All parameters come from configs/base.yaml — never hardcode values here.
Compression must be applied before calling this module.

Usage:
    from defenses.adversarial_training import adversarial_train
    hardened_model = adversarial_train(model, train_loader, config)
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    print("[AT] Running FGSM perturbation sanity check …")

    images, labels = next(iter(train_loader))
    images = images.to(model_device)
    labels = labels.to(model_device)

    # Generate adversarial examples (torchattacks handles grad internally).
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
        f"[AT] Perturbation L-inf (pixel space) : {linf:.5f}  "
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

    print("[AT] Perturbation sanity check PASSED.\n")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    compression: str,
    checkpoint_dir: str,
) -> str:
    """Save model checkpoint after a training epoch.

    INT4 models (bitsandbytes NF4) embed absmax, quant_map, and quant_state
    metadata directly in their parameter tensors.  Saving only the state_dict
    and reloading it into a freshly-quantised model causes a state conflict
    because the quantisation metadata is regenerated on load.  To avoid this,
    INT4 checkpoints save the full model object.

    FP32 and INT8 models are saved as plain state_dicts (lighter, portable).

    Filename convention:
      INT4  →  at_{compression}_epoch{epoch:02d}_full_model.pt
      other →  at_{compression}_epoch{epoch:02d}.pt

    Args:
        model:          The fine-tuned model.
        epoch:          1-based epoch index.
        compression:    Compression level string, e.g. "fp32", "int8", "int4".
        checkpoint_dir: Directory path for AT checkpoints (from base.yaml).

    Returns:
        Absolute path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if compression == "int4":
        filename = f"at_{compression}_epoch{epoch:02d}_full_model.pt"
        ckpt_path = os.path.join(checkpoint_dir, filename)
        torch.save(model, ckpt_path)
    else:
        filename = f"at_{compression}_epoch{epoch:02d}.pt"
        ckpt_path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), ckpt_path)
    return os.path.abspath(ckpt_path)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def adversarial_train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict,
    compression: str = "fp32",
) -> nn.Module:
    """Fine-tune an already-compressed model using FGSM adversarial inputs.

    Training follows the AT protocol from CLAUDE.md:
      - Optimizer: SGD with params from config["defense"]
      - Loss:      CrossEntropy on FGSM-perturbed inputs
      - Epochs:    config["defense"]["epochs"]  (7)
      - Checkpoint saved after every epoch to config["paths"]["checkpoints_at_dir"]

    Args:
        model:        A torch.nn.Module that has already been compressed.
                      Must be on the correct device before this call.
        train_loader: DataLoader over the training subset (batch_size=32 expected).
        config:       Parsed base.yaml config dict.
        compression:  Compression level string used to name checkpoint files.
                      One of "fp32", "int8", "int4".

    Returns:
        model: The same nn.Module, fine-tuned in place, returned in eval mode.
    """
    _set_seeds(config["seed"])

    defense_cfg = config["defense"]
    ds_cfg      = config["dataset"]
    ckpt_dir = config["paths"]["checkpoints_at_dir"]
    epochs: int       = defense_cfg["epochs"]
    lr: float         = defense_cfg["lr"]
    momentum: float   = defense_cfg["momentum"]
    weight_decay: float = defense_cfg["weight_decay"]
    at_eps: float     = defense_cfg["at_eps"]
    save_every_epoch: bool = defense_cfg.get("save_every_epoch", True)
    mean: list        = ds_cfg["mean"]
    std: list         = ds_cfg["std"]

    # Infer device from the first model parameter.
    model_device = next(model.parameters()).device

    # Build FGSM attack bound to this model.
    # set_normalization_used() is mandatory: training images are
    # ImageNet-normalised (range ≈ [-2.1, 2.6]).  Without it torchattacks
    # clamps normalised values to [0, 1], producing effective perturbations
    # of ~2.1 in pixel space instead of the intended 8/255 ≈ 0.031.
    # With it, torchattacks un-normalises internally, applies eps in [0,1]
    # pixel space, then re-normalises before returning.
    #
    # Use _LogitsWrapper so INT8/INT4 HuggingFace models (which return a
    # dataclass) expose a plain tensor interface to torchattacks.
    fgsm = torchattacks.FGSM(_LogitsWrapper(model), eps=at_eps)
    fgsm.set_normalization_used(mean=mean, std=std)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"[AT] Starting adversarial training — {epochs} epoch(s)")
    print(f"[AT] compression : {compression}")
    print(f"[AT] device      : {model_device}")
    print(f"[AT] lr={lr}  momentum={momentum}  weight_decay={weight_decay}")
    print(f"[AT] at_eps      : {at_eps:.5f}  ({round(at_eps * 255)}/255)")
    print(f"[AT] checkpoints : {ckpt_dir}")
    print()

    # Verify FGSM produces correct perturbations BEFORE any epoch runs.
    # Raises ValueError immediately if L-inf is outside at_eps ± 10%.
    _check_fgsm_perturbation(fgsm, train_loader, at_eps, model_device, mean, std)

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for images, labels in loop:
            images = images.to(model_device)
            labels = labels.to(model_device)

            # Generate FGSM adversarial examples.
            # torchattacks temporarily sets model.eval() internally, then
            # restores training mode via model.train() after generation.
            adv_images = fgsm(images, labels)

            optimizer.zero_grad()
            logits = model(adv_images)

            # Unwrap HuggingFace dataclass output (INT8 / INT4 models).
            if hasattr(logits, "logits"):
                logits = logits.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"[AT] Epoch {epoch}/{epochs} — "
            f"loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}"
        )

        if save_every_epoch:
            ckpt_path = save_checkpoint(model, epoch, compression, ckpt_dir)
            print(f"[AT] Checkpoint saved → {ckpt_path}")

    model.eval()
    print(f"\n[AT] Training complete.  Model returned in eval mode.")
    return model


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

    print("=== Sanity check: AT on DeiT-S FP32 (2-epoch smoke test) ===\n")

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
    # Use a tiny 64-sample subset so the smoke test finishes quickly.
    subset = Subset(train_dataset, list(range(64)))
    loader = DataLoader(
        subset,
        batch_size=cfg["defense"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Smoke test: override epochs to 2 without mutating the config dict.
    smoke_cfg = {**cfg, "defense": {**cfg["defense"], "epochs": 2}}

    model = load_model("deit_small", "fp32", cfg, device=device)
    model = adversarial_train(model, loader, smoke_cfg, compression="fp32")
    print("\n[AT] Sanity check passed.")
