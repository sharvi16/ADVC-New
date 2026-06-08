"""
utils/paper_figures.py

Generates all paper figures and Table 1 from experiment results.

Figures produced (saved to results/figures/):
    fig1_attack_samples_{compression}_{attack}.png
        Side-by-side: original | adversarial | perturbation×10

    fig2_asr_vs_epsilon_{compression}.png
        ASR vs epsilon curve for FGSM and PGD across a sweep of epsilon values.

    fig3_defense_pipeline_{compression}_{attack}.png
        Three-column panel: pre-attack clean | post-attack adversarial | post-defense prediction

    fig4_compression_vs_defense.png
        Grouped bar chart: compression level vs robust_acc, one group per defense.

    table1_quantitative_metrics.csv   (machine-readable)
    table1_quantitative_metrics.txt   (LaTeX-ready)
        Accuracy, PSNR, SSIM, ASR — one row per (compression, defense, attack).

Usage:
    python utils/paper_figures.py                         # all figures
    python utils/paper_figures.py --fig 1                 # only figure 1
    python utils/paper_figures.py --compression int8      # restrict compression
    python utils/paper_figures.py --attack fgsm           # restrict attack
    python utils/paper_figures.py --n-samples 4           # images per panel
    python utils/paper_figures.py --skip-model-figs       # table + fig2/4 only (no GPU needed)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model, resolve_data_path
import attacks.fgsm as fgsm_mod
import attacks.pgd  as pgd_mod
import attacks.patch as patch_mod

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

COLORS = {
    "fp32": "#2c7bb6",
    "int8": "#fdae61",
    "int4": "#d7191c",
    "none": "#aaaaaa",
    "at":   "#1a9641",
    "at_kd":"#7b2d8b",
}

DEFENSE_LABELS = {"none": "No defense", "at": "AT", "at_kd": "AT+KD"}
COMPRESSION_LABELS = {"fp32": "FP32", "int8": "INT8", "int4": "INT4"}
ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "patch": "Patch"}

# ── Label remapping ───────────────────────────────────────────────────────────

_IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764": 0,   "n02102040": 217, "n02979186": 482,
    "n03000684": 491, "n03028079": 497, "n03394916": 566,
    "n03417042": 569, "n03425413": 571, "n03445777": 574,
    "n03888257": 701,
}

def _remap_labels(dataset):
    if len(dataset.classes) >= 1000:
        return dataset
    new_samples = []
    for path, lbl in dataset.samples:
        synset = dataset.classes[lbl]
        new_samples.append((path, _IMAGENETTE_TO_IMAGENET.get(synset, lbl)))
    dataset.samples = new_samples
    dataset.targets = [lbl for _, lbl in new_samples]
    return dataset

# ── Tensor / image helpers ────────────────────────────────────────────────────

def _denorm(t: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """(C,H,W) normalised tensor → (H,W,3) uint8."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = (t.cpu().float() * s + m).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def _perturb_vis(clean: torch.Tensor, adv: torch.Tensor, amp: int = 10) -> np.ndarray:
    """Amplified absolute difference as uint8."""
    diff = (adv.cpu().float() - clean.cpu().float()).abs().clamp(0, 1)
    return ((diff * amp).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def _psnr(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB (computed in [0,1] pixel space)."""
    mse = (clean.float() - adv.float()).pow(2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10 * np.log10(1.0 / mse)

def _ssim_batch(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """Mean SSIM over a batch, computed per-channel then averaged.

    Uses the simplified single-scale formula with window=8 for speed.
    Both tensors should be in [0,1] normalised pixel space.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssims = []
    for i in range(clean.shape[0]):
        c = clean[i].float()      # (C, H, W) in [0,1]
        a = adv[i].float()
        mu_c  = c.mean()
        mu_a  = a.mean()
        sig_c = c.var()
        sig_a = a.var()
        sig_ca = ((c - mu_c) * (a - mu_a)).mean()
        num = (2 * mu_c * mu_a + C1) * (2 * sig_ca + C2)
        den = (mu_c ** 2 + mu_a ** 2 + C1) * (sig_c + sig_a + C2)
        ssims.append((num / den).item())
    return float(np.mean(ssims))

def _unnorm_batch(t: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """(N,C,H,W) normalised → [0,1] pixel space."""
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (t.float() * s + m).clamp(0, 1)

# ── Data loader ───────────────────────────────────────────────────────────────

def _build_loader(cfg: dict, n: int, device: str) -> DataLoader:
    ds_cfg = cfg["dataset"]
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])
    ds = _remap_labels(
        ImageFolder(root=str(resolve_data_path(_ROOT, ds_cfg["val_dir"])), transform=transform)
    )
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
    return DataLoader(
        Subset(ds, indices), batch_size=n, shuffle=False,
        num_workers=cfg["eval"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

# ── Model wrapper ─────────────────────────────────────────────────────────────

import torch.nn as nn

class _W(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        out = self.m(x)
        return out.logits if hasattr(out, "logits") else out

def _load(model_name, compression, cfg, device):
    return _W(load_model(model_name, compression, cfg, device=device)).eval()

def _dev(model):
    return str(next(model.parameters()).device)

def _build_attack(name, model, cfg):
    if name == "fgsm":  return fgsm_mod.build_attack(model, cfg)
    if name == "pgd":   return pgd_mod.build_attack(model, cfg)
    if name == "patch": return patch_mod.build_attack(model, cfg)
    raise ValueError(name)

# ── CSV reader ────────────────────────────────────────────────────────────────

def _load_all_results(cfg: dict) -> list[dict]:
    """Load phase1, phase2_at, phase2_atkd CSVs into one list of dicts."""
    results_dir = _ROOT / cfg["paths"]["results_dir"]
    files = {
        "none":  results_dir / "phase1_results.csv",
        "at":    results_dir / "phase2_at_results.csv",
        "at_kd": results_dir / "phase2_atkd_results.csv",
    }
    rows = []
    for defense, path in files.items():
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["defense"] = row.get("defense", defense)
                rows.append(row)
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Side-by-side: original | adversarial | perturbation×10
# ─────────────────────────────────────────────────────────────────────────────

def fig1_attack_samples(
    model_name: str,
    compression: str,
    attack_name: str,
    cfg: dict,
    figures_dir: Path,
    device: str,
    n: int = 4,
    amp: int = 10,
) -> Path:
    """
    N rows × 3 columns:
      col 0 — Original clean image
      col 1 — Adversarial image
      col 2 — Perturbation (|adv − clean|) amplified ×amp

    Args:
        model_name:   "deit_small"
        compression:  "fp32" | "int8" | "int4"
        attack_name:  "fgsm" | "pgd" | "patch"
        cfg:          Parsed base.yaml config.
        figures_dir:  Output directory.
        device:       "cuda" or "cpu".
        n:            Number of sample images (rows).
        amp:          Perturbation amplification factor.

    Returns:
        Path to saved PNG.
    """
    mean, std = cfg["dataset"]["mean"], cfg["dataset"]["std"]
    loader = _build_loader(cfg, n, device)
    images, labels = next(iter(loader))

    model  = _load(model_name, compression, cfg, device)
    attack = _build_attack(attack_name, model, cfg)
    mdev   = _dev(model)

    adv = attack(images.to(mdev), labels.to(mdev)).cpu().detach()

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.08})
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(
        ["Original", f"Adversarial ({ATTACK_LABELS[attack_name]})", f"Perturbation ×{amp}"]
    ):
        axes[0, col].set_title(title, fontsize=12, pad=7, fontweight="bold")

    for row in range(n):
        axes[row, 0].imshow(_denorm(images[row], mean, std))
        axes[row, 1].imshow(_denorm(adv[row], mean, std))
        axes[row, 2].imshow(_perturb_vis(images[row], adv[row], amp))
        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(
        f"Figure 1 — DeiT-S  |  {COMPRESSION_LABELS[compression]}  |  "
        f"{ATTACK_LABELS[attack_name]} attack  |  ε = {cfg['fgsm']['eps']:.4f}",
        fontsize=13, y=1.01, fontweight="bold",
    )

    out = figures_dir / f"fig1_attack_samples_{compression}_{attack_name}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"[fig1] Saved → {out}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ASR vs epsilon sweep
# ─────────────────────────────────────────────────────────────────────────────

def fig2_asr_vs_epsilon(
    model_name: str,
    compression: str,
    cfg: dict,
    figures_dir: Path,
    device: str,
    n_eval: int = 200,
    epsilons: list | None = None,
) -> Path:
    """
    Line plot of Attack Success Rate vs epsilon for FGSM and PGD.

    The model is loaded once.  For each epsilon value the attack is rebuilt and
    run on a fixed n_eval-image subset.  PGD uses the same alpha/steps from
    config but with the swept epsilon.

    Args:
        model_name:  "deit_small"
        compression: Compression level to evaluate.
        cfg:         Parsed base.yaml config.
        figures_dir: Output directory.
        device:      "cuda" or "cpu".
        n_eval:      Number of images to evaluate per epsilon point.
        epsilons:    List of epsilon values (floats in [0,1]).
                     Default: 9 evenly-spaced values from 1/255 to 16/255.

    Returns:
        Path to saved PNG.
    """
    if epsilons is None:
        epsilons = [k / 255.0 for k in [1, 2, 4, 6, 8, 10, 12, 14, 16]]

    loader = _build_loader(cfg, n_eval, device)
    images, labels = next(iter(loader))

    model = _load(model_name, compression, cfg, device)
    mdev  = _dev(model)
    imgs  = images.to(mdev)
    lbls  = labels.to(mdev)

    import torchattacks
    mean, std = cfg["dataset"]["mean"], cfg["dataset"]["std"]

    asr_fgsm, asr_pgd = [], []

    for eps in epsilons:
        # FGSM — attack needs gradients; only suppress them for the prediction
        atk = torchattacks.FGSM(model, eps=eps)
        atk.set_normalization_used(mean=mean, std=std)
        adv = atk(imgs, lbls)
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        asr_fgsm.append((preds != lbls).float().mean().item())

        # PGD — keep alpha proportional (alpha = eps/4, min 0.5/255)
        alpha = max(eps / 4.0, 0.5 / 255.0)
        atk2 = torchattacks.PGD(
            model, eps=eps, alpha=alpha,
            steps=cfg["pgd"]["steps"],
        )
        atk2.set_normalization_used(mean=mean, std=std)
        adv2 = atk2(imgs, lbls)
        with torch.no_grad():
            preds2 = model(adv2).argmax(dim=1)
        asr_pgd.append((preds2 != lbls).float().mean().item())

    eps_labels = [f"{round(e * 255)}/255" for e in epsilons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(epsilons)), asr_fgsm, "o-",
            color=COLORS["fp32"], linewidth=2, markersize=7, label="FGSM")
    ax.plot(range(len(epsilons)), asr_pgd,  "s--",
            color=COLORS["int4"], linewidth=2, markersize=7, label="PGD")

    # Mark the configured epsilon used in experiments
    cfg_eps = cfg["fgsm"]["eps"]
    if cfg_eps in epsilons:
        idx = epsilons.index(cfg_eps)
        ax.axvline(idx, color="#555555", linestyle=":", linewidth=1.5,
                   label=f"Experiment ε = {round(cfg_eps*255)}/255")

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels(eps_labels, rotation=30, ha="right")
    ax.set_xlabel("Perturbation budget ε (L∞)")
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Figure 2 — ASR vs Epsilon  |  DeiT-S {COMPRESSION_LABELS[compression]}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = figures_dir / f"fig2_asr_vs_epsilon_{compression}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"[fig2] Saved → {out}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Defense pipeline: pre-attack | post-attack | post-defense
# ─────────────────────────────────────────────────────────────────────────────

def fig3_defense_pipeline(
    model_name: str,
    compression: str,
    attack_name: str,
    cfg: dict,
    figures_dir: Path,
    device: str,
    n: int = 4,
    amp: int = 10,
) -> Path:
    """
    N rows × 4 columns:
      col 0 — Clean image + FP32 prediction
      col 1 — Adversarial image + undefended prediction (wrong)
      col 2 — Perturbation ×amp
      col 3 — Same adversarial image + AT-defended prediction

    Requires the AT checkpoint to exist in results/checkpoints/at/.
    Falls back gracefully if the checkpoint is missing (skips col 3).

    Args:
        model_name:   "deit_small"
        compression:  "fp32" | "int8" | "int4"
        attack_name:  "fgsm" | "pgd" | "patch"
        cfg:          Parsed base.yaml config.
        figures_dir:  Output directory.
        device:       "cuda" or "cpu".
        n:            Number of sample images (rows).
        amp:          Perturbation amplification factor.

    Returns:
        Path to saved PNG.
    """
    mean, std = cfg["dataset"]["mean"], cfg["dataset"]["std"]
    loader = _build_loader(cfg, n, device)
    images, labels = next(iter(loader))

    # Undefended model
    model_undef = _load(model_name, compression, cfg, device)
    mdev = _dev(model_undef)
    attack = _build_attack(attack_name, model_undef, cfg)
    adv = attack(images.to(mdev), labels.to(mdev)).cpu().detach()

    with torch.no_grad():
        preds_clean = model_undef(images.to(mdev)).argmax(dim=1).cpu()
        preds_undef = model_undef(adv.to(mdev)).argmax(dim=1).cpu()

    del model_undef
    if device == "cuda":
        torch.cuda.empty_cache()

    # AT-defended model (optional — load checkpoint if available)
    preds_def = None
    ckpt_dir  = _ROOT / cfg["paths"]["checkpoints_at_dir"]
    epochs    = cfg["defense"]["epochs"]
    full_path = ckpt_dir / f"at_{compression}_epoch{epochs:02d}_full_model.pt"
    sd_path   = ckpt_dir / f"at_{compression}_epoch{epochs:02d}.pt"

    if full_path.is_file() or sd_path.is_file():
        try:
            raw_def = load_model(model_name, compression, cfg, device=device)
            if full_path.is_file():
                raw_def = torch.load(str(full_path), map_location="cpu", weights_only=False)
            else:
                state = torch.load(str(sd_path), map_location="cpu", weights_only=True)
                raw_def.load_state_dict(state)
            raw_def.eval()
            model_def = _W(raw_def)
            mdev2 = _dev(raw_def)
            with torch.no_grad():
                preds_def = model_def(adv.to(mdev2)).argmax(dim=1).cpu()
            del model_def
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[fig3] AT checkpoint loaded for {compression}")
        except Exception as e:
            print(f"[fig3] Could not load AT checkpoint ({e}) — col 3 will show N/A")
            preds_def = None

    n_cols = 4
    col_titles = [
        "Pre-attack (clean)",
        f"Post-attack ({ATTACK_LABELS[attack_name]})",
        f"Perturbation ×{amp}",
        "Post-defense (AT)",
    ]

    fig, axes = plt.subplots(n, n_cols, figsize=(13, 3.2 * n),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=7, fontweight="bold")

    def _label_color(pred, gt):
        return "#2ecc71" if pred == gt else "#e74c3c"

    for row in range(n):
        gt = labels[row].item()

        axes[row, 0].imshow(_denorm(images[row], mean, std))
        axes[row, 0].set_xlabel(
            f"pred={preds_clean[row].item()}  gt={gt}", fontsize=8,
            color=_label_color(preds_clean[row].item(), gt), labelpad=2,
        )

        axes[row, 1].imshow(_denorm(adv[row], mean, std))
        axes[row, 1].set_xlabel(
            f"pred={preds_undef[row].item()}  gt={gt}", fontsize=8,
            color=_label_color(preds_undef[row].item(), gt), labelpad=2,
        )

        axes[row, 2].imshow(_perturb_vis(images[row], adv[row], amp))

        if preds_def is not None:
            axes[row, 3].imshow(_denorm(adv[row], mean, std))
            axes[row, 3].set_xlabel(
                f"pred={preds_def[row].item()}  gt={gt}", fontsize=8,
                color=_label_color(preds_def[row].item(), gt), labelpad=2,
            )
        else:
            axes[row, 3].text(
                0.5, 0.5, "No checkpoint", transform=axes[row, 3].transAxes,
                ha="center", va="center", fontsize=9, color="#888888",
            )
            axes[row, 3].set_facecolor("#f5f5f5")

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(
        f"Figure 3 — Defense pipeline  |  DeiT-S {COMPRESSION_LABELS[compression]}  |  "
        f"{ATTACK_LABELS[attack_name]}  |  green=correct  red=wrong",
        fontsize=12, y=1.01, fontweight="bold",
    )

    out = figures_dir / f"fig3_defense_pipeline_{compression}_{attack_name}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"[fig3] Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Compression quality vs defense effectiveness
# ─────────────────────────────────────────────────────────────────────────────

def fig4_compression_vs_defense(
    cfg: dict,
    figures_dir: Path,
    attack_filter: str | None = None,
) -> Path:
    """
    Grouped bar chart: x-axis = compression level, groups = defense,
    y-axis = robust_acc (higher is better).

    One chart per attack (or all three side-by-side if attack_filter is None).
    Data is read from the results CSVs — no model loading needed.

    Args:
        cfg:            Parsed base.yaml config.
        figures_dir:    Output directory.
        attack_filter:  If set, show only this attack.

    Returns:
        Path to saved PNG.
    """
    rows = _load_all_results(cfg)
    if not rows:
        print("[fig4] No results CSVs found — skipping figure 4.")
        return None

    attacks_in_data = sorted({r["attack"] for r in rows})
    if attack_filter:
        attacks_in_data = [a for a in attacks_in_data if a == attack_filter]

    compressions = ["int8", "int4"]
    defenses     = ["none", "at", "at_kd"]
    def_labels   = [DEFENSE_LABELS.get(d, d) for d in defenses]

    n_attacks = len(attacks_in_data)
    fig, axes = plt.subplots(1, n_attacks, figsize=(5.5 * n_attacks, 5), sharey=True)
    if n_attacks == 1:
        axes = [axes]

    bar_w  = 0.22
    x      = np.arange(len(compressions))

    for ax, attack_name in zip(axes, attacks_in_data):
        for di, defense in enumerate(defenses):
            vals = []
            for comp in compressions:
                match = [
                    float(r["robust_acc"])
                    for r in rows
                    if r["compression"] == comp
                    and r["defense"]     == defense
                    and r["attack"]      == attack_name
                ]
                vals.append(match[0] if match else 0.0)

            offset = (di - 1) * bar_w
            bars = ax.bar(
                x + offset, vals, bar_w,
                label=def_labels[di],
                color=COLORS.get(defense, "#999999"),
                edgecolor="white", linewidth=0.6,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([COMPRESSION_LABELS[c] for c in compressions])
        ax.set_xlabel("Compression level")
        ax.set_title(f"{ATTACK_LABELS.get(attack_name, attack_name)} attack", fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Robust accuracy ↑")
            ax.legend(loc="upper right")

    fig.suptitle(
        "Figure 4 — Compression level vs Defense effectiveness (robust accuracy)",
        fontsize=13, y=1.02, fontweight="bold",
    )

    suffix = f"_{attack_filter}" if attack_filter else ""
    out = figures_dir / f"fig4_compression_vs_defense{suffix}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"[fig4] Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Quantitative metrics: accuracy, PSNR, SSIM, ASR
# ─────────────────────────────────────────────────────────────────────────────

def table1_quantitative(
    model_name: str,
    cfg: dict,
    figures_dir: Path,
    device: str,
    n_eval: int = 200,
    compressions: list | None = None,
    attack_names: list | None = None,
) -> tuple[Path, Path]:
    """
    Compute PSNR and SSIM for each (compression, attack) pair on a fixed
    n_eval-image subset, then merge with clean_acc / robust_acc / ASR from
    the results CSVs.  Saves:

      table1_quantitative_metrics.csv  — machine-readable
      table1_quantitative_metrics.txt  — LaTeX tabular (ready to paste)

    Rows with no CSV result get NaN for accuracy metrics but still include
    PSNR and SSIM (which are computed here from scratch).

    Args:
        model_name:   "deit_small"
        cfg:          Parsed base.yaml config.
        figures_dir:  Output directory (same as other figures).
        device:       "cuda" or "cpu".
        n_eval:       Images to evaluate for PSNR/SSIM computation.
        compressions: Subset of levels to include (default: all three).
        attack_names: Subset of attacks to include (default: fgsm, pgd, patch).

    Returns:
        (csv_path, txt_path)
    """
    compressions  = compressions  or ["int8", "int4"]
    attack_names  = attack_names  or ["fgsm", "pgd", "patch"]
    csv_results   = _load_all_results(cfg)

    def _lookup(comp, defense, attack, field):
        for r in csv_results:
            if r["compression"] == comp and r["defense"] == defense and r["attack"] == attack:
                return float(r[field])
        return float("nan")

    mean, std = cfg["dataset"]["mean"], cfg["dataset"]["std"]
    loader = _build_loader(cfg, n_eval, device)
    images_cpu, labels_cpu = next(iter(loader))

    table_rows = []

    for compression in compressions:
        print(f"[table1] Loading {compression} model …")
        model = _load(model_name, compression, cfg, device)
        mdev  = _dev(model)

        for attack_name in attack_names:
            print(f"[table1]   attack={attack_name} …")
            attack = _build_attack(attack_name, model, cfg)
            adv    = attack(images_cpu.to(mdev), labels_cpu.to(mdev)).cpu().detach()

            clean_px = _unnorm_batch(images_cpu, mean, std)
            adv_px   = _unnorm_batch(adv, mean, std)
            psnr_val = _psnr(clean_px, adv_px)
            ssim_val = _ssim_batch(clean_px, adv_px)

            for defense in ["none", "at", "at_kd"]:
                table_rows.append({
                    "compression":    compression,
                    "defense":        defense,
                    "attack":         attack_name,
                    "clean_acc":      _lookup(compression, defense, attack_name, "clean_acc"),
                    "robust_acc":     _lookup(compression, defense, attack_name, "robust_acc"),
                    "asr":            _lookup(compression, defense, attack_name, "asr"),
                    "robustness_gap": _lookup(compression, defense, attack_name, "robustness_gap"),
                    "psnr_db":        round(psnr_val, 2),
                    "ssim":           round(ssim_val, 4),
                })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_fields = [
        "compression", "defense", "attack",
        "clean_acc", "robust_acc", "asr", "robustness_gap",
        "psnr_db", "ssim",
    ]
    csv_path = figures_dir / "table1_quantitative_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(table_rows)
    print(f"[table1] CSV saved → {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative evaluation of DeiT-S under adversarial attacks "
        r"across compression levels and defenses.}",
        r"\label{tab:quantitative}",
        r"\small",
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"Compression & Defense & Attack & Clean Acc & Robust Acc & ASR & Gap & PSNR (dB) \\ \midrule",
    ]
    prev_comp = None
    for r in table_rows:
        if r["compression"] != prev_comp and prev_comp is not None:
            lines.append(r"\midrule")
        prev_comp = r["compression"]
        lines.append(
            f"{COMPRESSION_LABELS[r['compression']]} & "
            f"{DEFENSE_LABELS.get(r['defense'], r['defense'])} & "
            f"{ATTACK_LABELS.get(r['attack'], r['attack'])} & "
            f"{_fmt(r['clean_acc'])} & "
            f"{_fmt(r['robust_acc'])} & "
            f"{_fmt(r['asr'])} & "
            f"{_fmt(r['robustness_gap'])} & "
            f"{_fmt(r['psnr_db'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    txt_path = figures_dir / "table1_quantitative_metrics.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table1] LaTeX saved → {txt_path}")

    return csv_path, txt_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument("--model",       default="deit_small", choices=["deit_small"])
    parser.add_argument("--compression", choices=["int8", "int4"], default=None)
    parser.add_argument("--attack",      choices=["fgsm", "pgd", "patch"], default=None)
    parser.add_argument("--fig",         choices=["1", "2", "3", "4", "table"], default=None,
                        help="Generate only this figure (default: all).")
    parser.add_argument("--n-samples",   type=int, default=4,
                        help="Images per panel for figures 1/3 (default: 4).")
    parser.add_argument("--n-eval",      type=int, default=200,
                        help="Images for fig2 epsilon sweep and table1 PSNR/SSIM (default: 200).")
    parser.add_argument("--amplify",     type=int, default=10,
                        help="Perturbation amplification factor (default: 10).")
    parser.add_argument("--skip-model-figs", action="store_true",
                        help="Skip figures that require model loading (fig1, fig2, fig3, table1).")
    args = parser.parse_args()

    cfg    = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    figures_dir: Path = _ROOT / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_compressions = cfg["compression"]["levels"]
    compressions = [args.compression] if args.compression else all_compressions
    all_attacks  = ["fgsm", "pgd", "patch"]
    attack_names = [args.attack] if args.attack else all_attacks

    want = args.fig  # None → all

    print(f"[paper_figures] device={device}  n_samples={args.n_samples}"
          f"  compressions={compressions}  attacks={attack_names}")
    print(f"[paper_figures] figures_dir={figures_dir}\n")

    # Figure 1
    if not args.skip_model_figs and want in (None, "1"):
        print("── Figure 1: attack samples ─────────────────────────")
        for comp in compressions:
            for atk in attack_names:
                fig1_attack_samples(
                    args.model, comp, atk, cfg, figures_dir, device,
                    n=args.n_samples, amp=args.amplify,
                )

    # Figure 2
    if not args.skip_model_figs and want in (None, "2"):
        print("\n── Figure 2: ASR vs epsilon ─────────────────────────")
        for comp in compressions:
            fig2_asr_vs_epsilon(
                args.model, comp, cfg, figures_dir, device, n_eval=args.n_eval,
            )

    # Figure 3
    if not args.skip_model_figs and want in (None, "3"):
        print("\n── Figure 3: defense pipeline ───────────────────────")
        for comp in compressions:
            for atk in attack_names:
                fig3_defense_pipeline(
                    args.model, comp, atk, cfg, figures_dir, device,
                    n=args.n_samples, amp=args.amplify,
                )

    # Figure 4
    if want in (None, "4"):
        print("\n── Figure 4: compression vs defense ─────────────────")
        fig4_compression_vs_defense(cfg, figures_dir, attack_filter=args.attack)

    # Table 1
    if not args.skip_model_figs and want in (None, "table"):
        print("\n── Table 1: quantitative metrics ────────────────────")
        table1_quantitative(
            args.model, cfg, figures_dir, device,
            n_eval=args.n_eval,
            compressions=compressions,
            attack_names=attack_names,
        )

    print(f"\n[paper_figures] Done. All outputs in {figures_dir}")


if __name__ == "__main__":
    main()
