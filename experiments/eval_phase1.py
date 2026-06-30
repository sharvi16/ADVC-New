"""
experiments/eval_phase1.py

Phase 1: no-defense robustness sweep across compression levels × attacks.

For each (compression, attack) pair the script records:
    model, compression, attack, clean_acc, robust_acc, asr, robustness_gap

Results are written to results/phase1_results.csv immediately after each pair
completes.  Already-written rows are detected on startup and skipped, so the
script is safe to interrupt and re-run (resumable).

Usage:
    python experiments/eval_phase1.py [--model deit_small|deit_base]
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path so sibling packages resolve correctly
# whether the script is run from the project root or from experiments/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model, resolve_data_path
import attacks.fgsm as fgsm_mod
import attacks.pgd as pgd_mod
import attacks.patch as patch_mod
from utils.metrics import (
    clean_accuracy,
    robust_accuracy,
    attack_success_rate,
    robustness_gap,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_FILE = "results/phase1_results.csv"
FIELDNAMES = [
    "timestamp",
    "model",
    "compression",
    "attack",
    "clean_acc",
    "robust_acc",
    "asr",
    "robustness_gap",
]
ATTACK_NAMES = ["fgsm", "pgd", "patch"]


# ── Logits normalisation wrapper ──────────────────────────────────────────────

class LogitsWrapper(nn.Module):
    """Unwrap HuggingFace model output to a plain (N, C) logits tensor.

    timm models (fp32) already return a plain tensor.
    HuggingFace models (int8 / int4 via transformers) return a dataclass with
    a .logits attribute.  This wrapper makes both interfaces identical so that
    torchattacks and the custom PatchAttack work across all compression levels.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if hasattr(out, "logits"):
            return out.logits
        return out


# ── Data loading ──────────────────────────────────────────────────────────────

# When using ImageNette (10-class subset) instead of full ImageNet-1k, the
# ImageFolder class indices (0–9) don't match the pretrained model's output
# indices.  This table remaps each synset to its correct ImageNet-1k position.
_IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _remap_subset_labels(dataset: ImageFolder) -> ImageFolder:
    """Remap ImageFolder targets to ImageNet-1k indices for subset datasets.

    No-op when the dataset already has 1000 classes (full ImageNet).
    """
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


def build_val_loader(cfg: dict, device: str, dataset: str = None) -> DataLoader:
    """Build a deterministic subset loader for the validation set.

    Works with both CIFAR datasets (no remapping) and ImageNette (remapped).

    The subset is drawn with seed=42 via randperm, matching the fixed split
    described in configs/base.yaml so results are reproducible across runs.
    """
    if dataset is None:
        dataset = cfg["dataset"]["name"]

    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["eval"]

    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        transform = T.Compose([
            T.Resize(ds_cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
        ])
        full_dataset = CIFAR10(root="data/cifar", train=False, download=True, transform=transform)
    elif dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        transform = T.Compose([
            T.Resize(ds_cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
        ])
        full_dataset = CIFAR100(root="data/cifar", train=False, download=True, transform=transform)
    else:
        # existing ImageNette path unchanged
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(ds_cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
        ])
        full_dataset = ImageFolder(root=str(resolve_data_path(_ROOT, ds_cfg["val_dir"])), transform=transform)
        full_dataset = _remap_subset_labels(full_dataset)

    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n = min(ds_cfg["val_subset_size"], len(full_dataset))
    indices = torch.randperm(len(full_dataset), generator=rng)[:n].tolist()
    print(f"[phase1] Val subset : {n} images, seed={cfg['seed']}, first 5 indices={indices[:5]}")
    subset = Subset(full_dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )
    return loader


def build_patch_val_loader(cfg: dict, device: str, dataset: str = None) -> DataLoader:
    """Build a smaller validation loader used exclusively for the patch attack.

    The patch attack runs 150 PGD-style optimisation steps per batch, making
    it ~7.5× more expensive than FGSM/PGD.  Using 500 images instead of the
    full val subset keeps patch evaluation tractable while
    still producing a statistically meaningful ASR estimate.
    FGSM and PGD always use the full val_subset_size loader.
    """
    if dataset is None:
        dataset = cfg["dataset"]["name"]

    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["eval"]

    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        transform = T.Compose([
            T.Resize(ds_cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
        ])
        full_dataset = CIFAR10(root="data/cifar", train=False, download=True, transform=transform)
    elif dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        transform = T.Compose([
            T.Resize(ds_cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
        ])
        full_dataset = CIFAR100(root="data/cifar", train=False, download=True, transform=transform)
    else:
        # existing ImageNette path unchanged
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(ds_cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
        ])
        full_dataset = ImageFolder(root=str(resolve_data_path(_ROOT, ds_cfg["val_dir"])), transform=transform)
        full_dataset = _remap_subset_labels(full_dataset)

    # Draw from the same shuffled order as build_val_loader so the 500 images
    # are a strict prefix of the full val subset — results stay comparable.
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n_full = min(ds_cfg["val_subset_size"], len(full_dataset))
    full_indices = torch.randperm(len(full_dataset), generator=rng)[:n_full].tolist()
    patch_indices = full_indices[:500]
    print(f"[phase1] Patch val subset : 500 images (subset of full {n_full}), seed={cfg['seed']}")
    subset = Subset(full_dataset, patch_indices)

    return DataLoader(
        subset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )


# ── Resumability helpers ──────────────────────────────────────────────────────

def load_completed_runs(results_path: str) -> set:
    """Return the set of (model, compression, attack) tuples already in the CSV."""
    completed: set = set()
    if not os.path.isfile(results_path):
        return completed
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["model"], row["compression"], row["attack"]))
    return completed


def append_row(
    results_path: str,
    model_name: str,
    compression: str,
    attack_name: str,
    c_acc: float,
    rob_acc: float,
    asr: float,
    rob_gap: float,
) -> None:
    """Append one result row; write CSV header if the file does not yet exist."""
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model_name,
            "compression": compression,
            "attack": attack_name,
            "clean_acc": round(c_acc, 6),
            "robust_acc": round(rob_acc, 6),
            "asr": round(asr, 6),
            "robustness_gap": round(rob_gap, 6),
        })


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_model_device(model: nn.Module) -> str:
    """Return the device string for the first model parameter found."""
    for p in model.parameters():
        return str(p.device)
    return "cpu"


@torch.no_grad()
def run_clean_eval(
    model: nn.Module,
    loader: DataLoader,
    model_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect clean logits and labels for the full loader.

    Returns:
        all_logits:  (N, C) tensor on CPU.
        all_labels:  (N,)   tensor on CPU.
    """
    logits_list, labels_list = [], []
    for images, labels in loader:
        images = images.to(model_device)
        logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def run_adv_eval(
    attack,
    model: nn.Module,
    loader: DataLoader,
    model_device: str,
) -> torch.Tensor:
    """Run the attack on every batch and collect adversarial logits.

    Gradient context is managed by the attack objects themselves; this function
    does not suppress gradients.

    Returns:
        all_adv_logits: (N, C) tensor on CPU.
    """
    adv_logits_list = []
    for images, labels in loader:
        images = images.to(model_device)
        labels = labels.to(model_device)
        adv_images = attack(images, labels)
        with torch.no_grad():
            adv_logits = model(adv_images)
        adv_logits_list.append(adv_logits.cpu())
    return torch.cat(adv_logits_list, dim=0)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results_path: str, model_name: str) -> None:
    """Print a formatted table of all phase1 rows for the given model."""
    if not os.path.isfile(results_path):
        return
    header = f"\n{'compression':<12} {'attack':<8} {'clean_acc':>10} {'robust_acc':>11} {'asr':>8} {'gap':>8}"
    print(header)
    print("-" * len(header.strip()))
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != model_name:
                continue
            print(
                f"{row['compression']:<12} {row['attack']:<8} "
                f"{float(row['clean_acc']):>10.4f} "
                f"{float(row['robust_acc']):>11.4f} "
                f"{float(row['asr']):>8.4f} "
                f"{float(row['robustness_gap']):>8.4f}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 — no-defense robustness sweep: compression × attack."
    )
    parser.add_argument(
        "--model",
        default="deit_small",
        choices=["deit_small", "deit_base"],
        help="Model to evaluate (default: deit_small)",
    )
    args = parser.parse_args()
    model_name: str = args.model

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[phase1] device      : {device}")
    print(f"[phase1] model       : {model_name}")
    print(f"[phase1] results     : {RESULTS_FILE}")
    print()

    loader = build_val_loader(cfg, device)
    patch_loader = build_patch_val_loader(cfg, device)
    completed = load_completed_runs(RESULTS_FILE)

    if completed:
        print(f"[phase1] Resuming — {len(completed)} combination(s) already done, skipping those.\n")

    compression_levels: list[str] = cfg["compression"]["levels"]

    for compression in compression_levels:
        remaining = [a for a in ATTACK_NAMES if (model_name, compression, a) not in completed]

        if not remaining:
            print(f"[phase1] {compression:<6}: all attacks already done — skipping model load.")
            continue

        # ── Load model (once per compression level) ───────────────────────────
        print(f"[phase1] {compression:<6}: loading {model_name} …")
        try:
            raw_model = load_model(model_name, compression, cfg, device=device)
        except Exception as exc:
            print(f"[phase1] {compression:<6}: load failed — {exc}")
            continue

        model = LogitsWrapper(raw_model)
        model.eval()
        model_device = infer_model_device(raw_model)
        print(f"[phase1] {compression:<6}: model on {model_device}")

        # ── Clean accuracy (computed once, reused for all attacks) ────────────
        print(f"[phase1] {compression:<6}: evaluating clean accuracy …")
        try:
            clean_logits, clean_labels = run_clean_eval(model, loader, model_device)
        except Exception as exc:
            print(f"[phase1] {compression:<6}: clean eval failed — {exc}")
            del raw_model, model
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        c_acc = clean_accuracy(clean_logits, clean_labels)
        print(f"[phase1] {compression:<6}: clean_acc = {c_acc:.4f}")

        # ── Per-attack robustness sweep ────────────────────────────────────────
        for attack_name in remaining:
            print(f"[phase1] {compression:<6} × {attack_name:<5}: building attack …")

            # Patch attack is evaluated on a smaller 500-image subset because
            # each batch requires 150 PGD optimisation steps, making it far
            # more expensive than FGSM/PGD.  FGSM and PGD use the full loader.
            if attack_name == "patch":
                eval_loader = patch_loader
                # Re-collect clean logits/labels for the 500-image subset so
                # that clean_acc and robustness_gap are computed on the same
                # images as the adversarial evaluation.
                try:
                    patch_clean_logits, patch_clean_labels = run_clean_eval(
                        model, patch_loader, model_device
                    )
                except Exception as exc:
                    print(f"[phase1] {compression:<6} × patch  : patch clean eval failed — {exc}")
                    continue
                eval_clean_logits = patch_clean_logits
                eval_clean_labels = patch_clean_labels
            else:
                eval_loader = loader
                eval_clean_logits = clean_logits
                eval_clean_labels = clean_labels

            if attack_name == "fgsm":
                attack = fgsm_mod.build_attack(model, cfg)
            elif attack_name == "pgd":
                attack = pgd_mod.build_attack(model, cfg)
            elif attack_name == "patch":
                attack = patch_mod.build_attack(model, cfg)
            else:
                raise ValueError(f"Unknown attack: {attack_name!r}")

            print(f"[phase1] {compression:<6} × {attack_name:<5}: running on {len(eval_loader.dataset)} images …")
            try:
                adv_logits = run_adv_eval(attack, model, eval_loader, model_device)
            except Exception as exc:
                print(f"[phase1] {compression:<6} × {attack_name:<5}: attack failed — {exc}")
                continue

            rob_acc = robust_accuracy(adv_logits, eval_clean_labels)
            asr = attack_success_rate(adv_logits, eval_clean_labels)
            rob_gap = robustness_gap(eval_clean_logits, adv_logits, eval_clean_labels)

            print(
                f"[phase1] {compression:<6} × {attack_name:<5}: "
                f"robust_acc={rob_acc:.4f}  asr={asr:.4f}  gap={rob_gap:.4f}"
            )

            append_row(RESULTS_FILE, model_name, compression, attack_name, c_acc, rob_acc, asr, rob_gap)
            print(f"[phase1] {compression:<6} × {attack_name:<5}: saved → {RESULTS_FILE}")

        # ── Free GPU memory before loading the next compression level ──────────
        del raw_model, model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n[phase1] All combinations complete.")
    print_summary(RESULTS_FILE, model_name)


if __name__ == "__main__":
    main()
