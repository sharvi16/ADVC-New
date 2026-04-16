"""
experiments/diagnose_fgsm.py

Diagnostic script: verify FGSM generates meaningful perturbations on AT-hardened
INT8 and INT4 models.

For each compression level (int8, int4) this script:
  1. Loads the AT checkpoint into the compressed model.
  2. Grabs the first 10 images from the validation set.
  3. Prints clean predictions and top-1 confidence scores.
  4. Generates FGSM adversarial images.
  5. Prints the L-inf norm of the perturbation (should be ≈ 8/255 = 0.03137).
  6. Prints adversarial predictions and top-1 confidence scores.
  7. Reports how many of the 10 images flipped prediction.
  8. Saves a perturbation visualisation to results/figures/fgsm_perturbation_{compression}.png

Usage:
    python experiments/diagnose_fgsm.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model

# ── Matplotlib — use non-interactive Agg backend so it works in Colab / headless ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Logits wrapper (same as in eval_phase2_at.py) ────────────────────────────

class LogitsWrapper(nn.Module):
    """Unwrap HuggingFace model output to a plain (N, C) logits tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


# ── ImageNette → ImageNet-1k label remapping ─────────────────────────────────

_IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def _remap_subset_labels(dataset: ImageFolder) -> ImageFolder:
    if len(dataset.classes) >= 1000:
        return dataset
    new_samples = []
    for path, lbl in dataset.samples:
        synset = dataset.classes[lbl]
        new_lbl = _IMAGENETTE_TO_IMAGENET.get(synset, lbl)
        new_samples.append((path, new_lbl))
    dataset.samples = new_samples
    dataset.targets = [lbl for _, lbl in new_samples]
    return dataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_at_checkpoint(model: nn.Module, compression: str, cfg: dict) -> nn.Module:
    """Load the final AT checkpoint (epoch 7) into an already-loaded compressed model."""
    ckpt_dir = cfg["paths"]["checkpoints_at_dir"]
    epochs: int = cfg["defense"]["epochs"]
    filename = f"at_{compression}_epoch{epochs:02d}.pt"
    ckpt_path = _ROOT / ckpt_dir / filename

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run eval_phase2_at.py first to generate it."
        )

    print(f"  Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def unnormalize(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """Undo ImageNet normalisation and return a (H, W, 3) uint8 numpy array."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = tensor.cpu() * s + m          # (3, H, W)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()   # (H, W, 3)
    return (img * 255).astype(np.uint8)


def save_perturbation_figure(
    clean_img: torch.Tensor,
    adv_img: torch.Tensor,
    compression: str,
    figures_dir: str,
    mean: list,
    std: list,
) -> str:
    """Save a 3-panel figure: clean | adversarial | perturbation×10.

    Args:
        clean_img:    (3, H, W) normalised tensor for one image.
        adv_img:      (3, H, W) normalised tensor for the same image (adversarial).
        compression:  "int8" or "int4" — used in the filename.
        figures_dir:  Directory to save the PNG.
        mean / std:   ImageNet normalisation parameters.

    Returns:
        Absolute path to the saved PNG.
    """
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(figures_dir) / f"fgsm_perturbation_{compression}.png")

    clean_np  = unnormalize(clean_img, mean, std)
    adv_np    = unnormalize(adv_img, mean, std)

    # Perturbation amplified ×10 for visibility, shifted to [0, 255]
    diff = adv_img.cpu() - clean_img.cpu()             # (3, H, W) float
    diff_vis = (diff * 10 + 0.5).clamp(0, 1)           # scale & centre on 0.5
    diff_vis = (diff_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    linf = diff.abs().max().item()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Clean", "Adversarial (FGSM)", f"Perturbation ×10\nL∞={linf:.5f}"]
    imgs   = [clean_np, adv_np, diff_vis]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle(f"FGSM perturbation — {compression} (AT checkpoint)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(Path(out_path).resolve())


# ── Main diagnostic routine ───────────────────────────────────────────────────

def diagnose(compression: str, cfg: dict, device: str) -> None:
    """Run the full diagnostic for one compression level."""
    print(f"\n{'='*60}")
    print(f"  DIAGNOSING: {compression.upper()}")
    print(f"{'='*60}")

    ds_cfg  = cfg["dataset"]
    mean    = ds_cfg["mean"]
    std     = ds_cfg["std"]
    eps     = cfg["fgsm"]["eps"]

    # ── Load model + AT checkpoint ────────────────────────────────────────────
    print(f"\n[1] Loading {compression} model …")
    raw_model = load_model("deit_small", compression, cfg, device=device)
    raw_model = load_at_checkpoint(raw_model, compression, cfg)

    model = LogitsWrapper(raw_model)
    model.eval()

    # Infer device (bitsandbytes models use device_map="auto")
    model_device = str(next(raw_model.parameters()).device)
    print(f"  Model on: {model_device}")

    # ── Build data loader — grab first 10 images ──────────────────────────────
    print(f"\n[2] Loading first 10 validation images …")
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    full_dataset = ImageFolder(root=str(_ROOT / ds_cfg["val_dir"]), transform=transform)
    full_dataset = _remap_subset_labels(full_dataset)

    # Take the first 10 samples (deterministic — no shuffle)
    subset  = Subset(full_dataset, list(range(10)))
    loader  = DataLoader(subset, batch_size=10, shuffle=False, num_workers=0)

    images, labels = next(iter(loader))
    images = images.to(model_device)
    labels = labels.to(model_device)
    print(f"  Images shape : {images.shape}")
    print(f"  Labels       : {labels.tolist()}")

    # ── Clean inference ───────────────────────────────────────────────────────
    print(f"\n[3] Clean predictions:")
    with torch.no_grad():
        clean_logits = model(images)                        # (10, 1000)
    clean_probs   = F.softmax(clean_logits, dim=1)
    clean_conf, clean_preds = clean_probs.max(dim=1)

    for i in range(len(labels)):
        marker = "✓" if clean_preds[i].item() == labels[i].item() else "✗"
        print(
            f"  [{marker}] img {i:02d}  "
            f"true={labels[i].item():4d}  "
            f"pred={clean_preds[i].item():4d}  "
            f"conf={clean_conf[i].item():.4f}"
        )

    # ── Generate FGSM adversarial examples ───────────────────────────────────
    print(f"\n[4] Generating FGSM adversarial examples (eps={eps:.5f} = {round(eps*255)}/255) …")
    fgsm = torchattacks.FGSM(model, eps=eps)
    adv_images = fgsm(images, labels)

    # ── L-inf perturbation check ──────────────────────────────────────────────
    print(f"\n[5] Perturbation L-inf norms (per image):")
    diffs = (adv_images - images).abs()
    for i in range(len(labels)):
        linf = diffs[i].max().item()
        print(f"  img {i:02d}  L∞={linf:.6f}  (expected ≈ {eps:.5f})")

    global_linf = diffs.max().item()
    print(f"\n  Global max L∞ = {global_linf:.6f}  (expected ≈ {eps:.5f})")
    if abs(global_linf - eps) < 1e-4:
        print("  ✓  Perturbation magnitude looks correct.")
    else:
        print("  ✗  WARNING: perturbation magnitude deviates from eps — check FGSM setup.")

    # ── Adversarial inference ─────────────────────────────────────────────────
    print(f"\n[6] Adversarial predictions:")
    with torch.no_grad():
        adv_logits = model(adv_images)
    adv_probs = F.softmax(adv_logits, dim=1)
    adv_conf, adv_preds = adv_probs.max(dim=1)

    for i in range(len(labels)):
        changed = adv_preds[i].item() != clean_preds[i].item()
        marker  = "FLIP" if changed else "same"
        print(
            f"  [{marker:<4}] img {i:02d}  "
            f"clean_pred={clean_preds[i].item():4d}  "
            f"adv_pred={adv_preds[i].item():4d}  "
            f"conf={adv_conf[i].item():.4f}"
        )

    # ── Flip summary ──────────────────────────────────────────────────────────
    n_flipped = (adv_preds != clean_preds).sum().item()
    n_fooled  = (adv_preds != labels).sum().item()     # fooled the true label
    print(f"\n[7] Summary:")
    print(f"  Images where prediction CHANGED (clean→adv) : {n_flipped} / {len(labels)}")
    print(f"  Images classified incorrectly on adv inputs : {n_fooled} / {len(labels)}")

    if n_flipped == 0:
        print(
            "  ⚠  No predictions changed — FGSM may not be generating effective "
            "perturbations for this compression level.  Check the model gradient flow."
        )
    else:
        print(f"  ✓  FGSM is generating meaningful perturbations.")

    # ── Save visualisation (uses first image) ─────────────────────────────────
    print(f"\n[8] Saving perturbation figure …")
    figures_dir = str(_ROOT / cfg["paths"]["figures_dir"])
    fig_path = save_perturbation_figure(
        images[0].cpu(),
        adv_images[0].cpu(),
        compression,
        figures_dir,
        mean,
        std,
    )
    print(f"  Saved → {fig_path}")

    # ── Free GPU memory ───────────────────────────────────────────────────────
    del raw_model, model, images, labels, adv_images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg    = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device : {device}")
    print(f"FGSM eps : {cfg['fgsm']['eps']:.5f}  ({round(cfg['fgsm']['eps'] * 255)}/255)")

    for compression in ["int8", "int4"]:
        try:
            diagnose(compression, cfg, device)
        except FileNotFoundError as exc:
            print(f"\n[SKIP] {compression}: {exc}")
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {compression}: {exc}")
            traceback.print_exc()

    print("\n\nDiagnosis complete.")
