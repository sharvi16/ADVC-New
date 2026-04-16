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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    compression: str,
    checkpoint_dir: str,
) -> str:
    """Save model state dict after a training epoch.

    Args:
        model:          The fine-tuned model.
        epoch:          1-based epoch index.
        compression:    Compression level string, e.g. "fp32", "int8", "int4".
        checkpoint_dir: Directory path for AT checkpoints (from base.yaml).

    Returns:
        Absolute path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
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
    ckpt_dir = config["paths"]["checkpoints_at_dir"]
    epochs: int = defense_cfg["epochs"]
    lr: float = defense_cfg["lr"]
    momentum: float = defense_cfg["momentum"]
    weight_decay: float = defense_cfg["weight_decay"]
    at_eps: float = defense_cfg["at_eps"]
    save_every_epoch: bool = defense_cfg.get("save_every_epoch", True)

    # Infer device from the first model parameter.
    model_device = next(model.parameters()).device

    # Build FGSM attack bound to this model.  The model is temporarily put into
    # train mode below; torchattacks will call model(x) to get gradients.
    # Use _LogitsWrapper so INT8/INT4 HuggingFace models (which return a
    # dataclass) expose a plain tensor interface to torchattacks.
    fgsm = torchattacks.FGSM(_LogitsWrapper(model), eps=at_eps)

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
