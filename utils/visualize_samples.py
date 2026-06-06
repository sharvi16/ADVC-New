"""
utils/visualize_samples.py

Generates paper-quality side-by-side visualisation panels comparing:
  1. Clean vs adversarial images (per attack)
  2. Pre-compression vs post-compression images (pixel-level difference)

All panels are saved to results/figures/ as high-DPI PNGs suitable for
inclusion in a research paper.  Never calls plt.show() — Colab-safe.

Usage:
    python utils/visualize_samples.py                        # all combos
    python utils/visualize_samples.py --compression fp32     # one level
    python utils/visualize_samples.py --attack fgsm          # one attack
    python utils/visualize_samples.py --n-samples 4          # fewer images

Output files (all in results/figures/): 
    attack_samples_{compression}_{attack}.png
        Grid: clean | adv | perturbation×10  — N rows
    compression_samples_{compression}.png
        Grid: fp32_clean | compressed_clean | pixel_diff×10  — N rows
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model, resolve_data_path
import attacks.fgsm as fgsm_mod
import attacks.pgd as pgd_mod
import attacks.patch as patch_mod

# ── Label remapping (mirrors eval scripts) ────────────────────────────────────

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


def _remap_labels(dataset: ImageFolder) -> ImageFolder:
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


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _denorm(t: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """Convert a (C, H, W) normalised tensor to a (H, W, 3) uint8 numpy array."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    img = (t.cpu().float() * std_t + mean_t).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _perturbation_vis(clean: torch.Tensor, adv: torch.Tensor, amplify: int = 10) -> np.ndarray:
    """Return an amplified absolute-difference image as uint8 (H, W, 3)."""
    diff = (adv.cpu().float() - clean.cpu().float()).abs()
    diff_amp = (diff * amplify).clamp(0.0, 1.0)
    return (diff_amp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── Loader ────────────────────────────────────────────────────────────────────

def _build_loader(cfg: dict, n: int, device: str) -> DataLoader:
    ds_cfg = cfg["dataset"]
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])
    ds = ImageFolder(root=str(resolve_data_path(_ROOT, ds_cfg["val_dir"])), transform=transform)
    ds = _remap_labels(ds)
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
    subset = Subset(ds, indices)
    return DataLoader(
        subset, batch_size=n, shuffle=False,
        num_workers=cfg["eval"]["num_workers"],
        pin_memory=(device == "cuda"),
    )


def _load_wrapper(model_name: str, compression: str, cfg: dict, device: str):
    """Load model and wrap in a thin logits-unwrap lambda."""
    from torch import nn

    class _W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out.logits if hasattr(out, "logits") else out

    raw = load_model(model_name, compression, cfg, device=device)
    return _W(raw).eval()


# ── Panel builders ────────────────────────────────────────────────────────────

def _save_attack_panel(
    images: torch.Tensor,
    labels: torch.Tensor,
    attack,
    attack_name: str,
    compression: str,
    cfg: dict,
    figures_dir: Path,
    model_device: str,
    amplify: int = 10,
) -> Path:
    """
    Save a panel of N rows × 3 columns:
      col 0 — clean image
      col 1 — adversarial image
      col 2 — perturbation × amplify (amplified absolute difference)

    Args:
        images:       (N, C, H, W) normalised tensor on CPU.
        labels:       (N,) integer labels.
        attack:       Callable attack(images, labels) → adv_images.
        attack_name:  String label for the figure title.
        compression:  Compression level string.
        cfg:          Parsed base.yaml config.
        figures_dir:  Directory to save the PNG.
        model_device: Device string to move tensors to for the attack.
        amplify:      Perturbation amplification factor for visibility.

    Returns:
        Path to the saved PNG.
    """
    mean = cfg["dataset"]["mean"]
    std  = cfg["dataset"]["std"]
    n = images.shape[0]

    adv = attack(images.to(model_device), labels.to(model_device)).cpu().detach()

    fig, axes = plt.subplots(
        n, 3,
        figsize=(9, 3 * n),
        gridspec_kw={"wspace": 0.04, "hspace": 0.06},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean", f"Adversarial ({attack_name.upper()})", f"Perturbation ×{amplify}"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=6, fontweight="bold")

    for row in range(n):
        clean_img = _denorm(images[row], mean, std)
        adv_img   = _denorm(adv[row],    mean, std)
        diff_img  = _perturbation_vis(images[row], adv[row], amplify)

        axes[row, 0].imshow(clean_img)
        axes[row, 1].imshow(adv_img)
        axes[row, 2].imshow(diff_img)
        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(
        f"DeiT-S  |  compression: {compression}  |  attack: {attack_name.upper()}",
        fontsize=12, y=1.01, fontweight="bold",
    )

    out_path = figures_dir / f"attack_samples_{compression}_{attack_name}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")
    return out_path


def _save_compression_panel(
    images_fp32: torch.Tensor,
    images_compressed: torch.Tensor,
    compression: str,
    cfg: dict,
    figures_dir: Path,
    amplify: int = 10,
) -> Path:
    """
    Save a panel of N rows × 3 columns:
      col 0 — FP32 clean image
      col 1 — compressed model clean image (same input, different model output not
               applicable here — this shows quantisation artefacts at input level
               which are zero for PTQ; instead we show FP32 logit-reconstructed vs
               compressed logit-reconstructed via nearest-class heatmap overlay)
      col 2 — pixel difference × amplify

    Since PTQ does not modify the input pixels (only the model weights), the
    pixel difference between fp32_clean and compressed_clean is always zero.
    Instead, col 1 shows the same image annotated with the compressed model's
    top-1 prediction label, and col 0 with the FP32 prediction.  This is the
    meaningful "pre vs post compression" comparison for a paper.

    The difference panel (col 2) is therefore replaced with a colour-coded
    correct/incorrect indicator overlaid on the image.

    Args:
        images_fp32:        (N, C, H, W) normalised tensors, FP32 model outputs.
        images_compressed:  Same images; tuple (logits_fp32, logits_comp, labels).
        compression:        Compression level string.
        cfg:                Parsed base.yaml config.
        figures_dir:        Directory to save the PNG.
        amplify:            Unused; kept for API consistency.

    Returns:
        Path to the saved PNG.
    """
    # Unpack
    images, logits_fp32, logits_comp, labels = images_compressed
    mean = cfg["dataset"]["mean"]
    std  = cfg["dataset"]["std"]
    n = images.shape[0]

    preds_fp32 = logits_fp32.argmax(dim=1)
    preds_comp = logits_comp.argmax(dim=1)

    fig, axes = plt.subplots(
        n, 3,
        figsize=(9, 3 * n),
        gridspec_kw={"wspace": 0.04, "hspace": 0.06},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["FP32 prediction", f"{compression.upper()} prediction", "Difference ×10"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=6, fontweight="bold")

    for row in range(n):
        img_np = _denorm(images[row], mean, std)
        diff   = _perturbation_vis(images[row], images[row], amplify)  # always zero

        gt     = labels[row].item()
        p_fp32 = preds_fp32[row].item()
        p_comp = preds_comp[row].item()

        # Annotate correctness with coloured borders
        def _border(ax, correct: bool):
            colour = "#2ecc71" if correct else "#e74c3c"
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(3)
                spine.set_visible(True)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_xlabel(
            f"pred={p_fp32}  gt={gt}", fontsize=8, labelpad=2,
            color="#2ecc71" if p_fp32 == gt else "#e74c3c",
        )
        _border(axes[row, 0], p_fp32 == gt)

        axes[row, 1].imshow(img_np)
        axes[row, 1].set_xlabel(
            f"pred={p_comp}  gt={gt}", fontsize=8, labelpad=2,
            color="#2ecc71" if p_comp == gt else "#e74c3c",
        )
        _border(axes[row, 1], p_comp == gt)

        # Pixel difference is zero for PTQ — show a grey canvas with annotation
        diff_canvas = np.full_like(img_np, 200)
        changed = "prediction changed" if p_fp32 != p_comp else "prediction same"
        axes[row, 2].imshow(diff_canvas)
        axes[row, 2].text(
            0.5, 0.5, changed, transform=axes[row, 2].transAxes,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#e74c3c" if p_fp32 != p_comp else "#2ecc71",
        )

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(
        f"DeiT-S  |  FP32 vs {compression.upper()} compression  |  "
        f"green border = correct, red = wrong",
        fontsize=12, y=1.01, fontweight="bold",
    )

    out_path = figures_dir / f"compression_samples_{compression}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality visualisation panels."
    )
    parser.add_argument(
        "--model", default="deit_small", choices=["deit_small"],
    )
    parser.add_argument(
        "--compression", choices=["fp32", "int8", "int4"], default=None,
        help="Restrict to one compression level (default: all three).",
    )
    parser.add_argument(
        "--attack", choices=["fgsm", "pgd", "patch"], default=None,
        help="Restrict to one attack (default: all three).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=4,
        help="Number of sample images per panel (default: 4).",
    )
    parser.add_argument(
        "--amplify", type=int, default=10,
        help="Perturbation amplification factor for the diff column (default: 10).",
    )
    args = parser.parse_args()

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    figures_dir = _ROOT / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_compressions: list[str] = cfg["compression"]["levels"]
    compressions = [args.compression] if args.compression else all_compressions
    all_attacks = ["fgsm", "pgd", "patch"]
    attack_names = [args.attack] if args.attack else all_attacks

    n = args.n_samples
    print(f"[viz] device={device}  n_samples={n}  figures_dir={figures_dir}")

    # Load one shared batch of N images (same seed as eval scripts)
    loader = _build_loader(cfg, n, device)
    images_cpu, labels_cpu = next(iter(loader))

    # ── FP32 model loaded once for compression comparison panels ──────────────
    fp32_model = None
    fp32_logits_cpu = None
    if any(c != "fp32" for c in compressions):
        print("\n[viz] Loading FP32 model for compression comparison …")
        fp32_model = _load_wrapper(args.model, "fp32", cfg, device)
        fp32_dev = next(fp32_model.parameters()).device
        with torch.no_grad():
            fp32_logits_cpu = fp32_model(images_cpu.to(str(fp32_dev))).cpu()

    # ── Per-compression loop ──────────────────────────────────────────────────
    for compression in compressions:
        print(f"\n[viz] ── compression: {compression} ──")

        model = _load_wrapper(args.model, compression, cfg, device)
        model_device = str(next(model.parameters()).device)

        # ── Compression panel ─────────────────────────────────────────────────
        with torch.no_grad():
            comp_logits_cpu = model(images_cpu.to(model_device)).cpu()

        if compression == "fp32":
            # FP32 vs FP32 is trivial — skip (no meaningful comparison)
            print(f"[viz] Skipping compression panel for fp32 (baseline = compressed)")
        else:
            _save_compression_panel(
                images_cpu,
                (images_cpu, fp32_logits_cpu, comp_logits_cpu, labels_cpu),
                compression,
                cfg,
                figures_dir,
                amplify=args.amplify,
            )

        # ── Attack panels ─────────────────────────────────────────────────────
        for attack_name in attack_names:
            print(f"[viz] Building {attack_name} attack …")

            if attack_name == "fgsm":
                attack = fgsm_mod.build_attack(model, cfg)
            elif attack_name == "pgd":
                attack = pgd_mod.build_attack(model, cfg)
            elif attack_name == "patch":
                attack = patch_mod.build_attack(model, cfg)
            else:
                raise ValueError(f"Unknown attack: {attack_name!r}")

            _save_attack_panel(
                images_cpu,
                labels_cpu,
                attack,
                attack_name,
                compression,
                cfg,
                figures_dir,
                model_device,
                amplify=args.amplify,
            )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[viz] All panels saved to {figures_dir}")


if __name__ == "__main__":
    main()
