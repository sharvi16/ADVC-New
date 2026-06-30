"""
experiments/eval_phase3.py

Phase 3: Combined attack (FGSM → PGD → Patch) vs all defenses × compression levels.

For each (compression, defense) pair the script:
  1. Loads DeiT-S at the given compression.
  2. Loads the corresponding defense checkpoint (none / AT / AT+KD).
  3. Evaluates the combined attack on 500 images (same subset as patch eval).
  4. Writes one row per (compression, defense) to results/phase3_results.csv.

Results are written immediately after each pair completes.
Already-written rows are detected on startup and skipped (resumable).

Usage:
    python experiments/eval_phase3.py --model deit_small
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
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model, resolve_data_path
import attacks.combined as combined_mod
from utils.metrics import (
    clean_accuracy,
    robust_accuracy,
    attack_success_rate,
    robustness_gap,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_FILE = None
PHASE = 3
FIELDNAMES = [
    "timestamp",
    "model",
    "compression",
    "defense",
    "attack",
    "clean_acc",
    "robust_acc",
    "asr",
    "robustness_gap",
    "phase",
]
DEFENSE_NAMES = ["none", "at", "at_kd"]
ATTACK_NAME = "combined"

# ── Logits normalisation wrapper ──────────────────────────────────────────────

class LogitsWrapper(nn.Module):
    """Unwrap HuggingFace model output to a plain (N, C) logits tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if hasattr(out, "logits"):
            return out.logits
        return out


# ── ImageNette → ImageNet-1k label remapping ──────────────────────────────────

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
    """Remap ImageFolder targets to ImageNet-1k indices for subset datasets."""
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


# ── Data loader ───────────────────────────────────────────────────────────────

def build_val_loader(cfg: dict, device: str) -> DataLoader:
    """Build the 500-image val loader used for the combined (patch-inclusive) attack.

    Uses the same seed=42 randperm prefix as the patch loaders in Phase 1/2 so
    results are directly comparable across phases.
    """
    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["eval"]

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
    n_full = min(ds_cfg["val_subset_size"], len(full_dataset))
    full_indices = torch.randperm(len(full_dataset), generator=rng)[:n_full].tolist()
    # Use the same 500-image prefix as the patch loaders in Phase 1/2
    indices = full_indices[:500]
    print(f"[phase3] Val subset : 500 images (patch-compatible subset), seed={cfg['seed']}")
    subset = Subset(full_dataset, indices)

    return DataLoader(
        subset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )


# ── Checkpoint loaders ────────────────────────────────────────────────────────

def _load_checkpoint(
    model: nn.Module,
    ckpt_path_full: Path,
    ckpt_path_state: Path,
    label: str,
    device: str = "cuda",
) -> nn.Module:
    """Load a checkpoint from disk — full model or state dict, whichever exists.

    For bitsandbytes quantized layers (Linear8bitLt), load_state_dict requires
    the model to already be on CUDA so the quantized buffers are initialised.
    We always move the model to device before loading the state dict.
    """
    if ckpt_path_full.is_file():
        print(f"[phase3] {label}: loading full model checkpoint from {ckpt_path_full} …")
        loaded = torch.load(str(ckpt_path_full), map_location=device)
        loaded.eval()
        return loaded
    if ckpt_path_state.is_file():
        print(f"[phase3] {label}: loading state dict from {ckpt_path_state} …")
        # bitsandbytes Linear8bitLt requires model on CUDA before load_state_dict
        # so the .CB quantized buffer is populated on the first forward pass.
        model = model.to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            model(dummy)
        state_dict = torch.load(str(ckpt_path_state), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    raise FileNotFoundError(
        f"No checkpoint found for {label}.  Tried:\n"
        f"  {ckpt_path_full}\n  {ckpt_path_state}"
    )


def load_defended_model(
    model: nn.Module,
    compression: str,
    defense: str,
    cfg: dict,
) -> nn.Module:
    """Return the model with the appropriate defense checkpoint applied.

    Args:
        model:       Already-loaded (and compressed) nn.Module.
        compression: "fp32", "int8", or "int4".
        defense:     "none", "at", or "at_kd".
        cfg:         Parsed base.yaml config dict.

    Returns:
        nn.Module in eval mode with defense weights loaded (or unchanged for "none").
    """
    if defense == "none":
        model.eval()
        return model

    epochs: int = cfg["defense"]["epochs"]

    if defense == "at":
        ckpt_dir = _ROOT / cfg["paths"]["checkpoints_at_dir"]
        full_path  = ckpt_dir / f"at_{compression}_epoch{epochs:02d}_full_model.pt"
        state_path = ckpt_dir / f"at_{compression}_epoch{epochs:02d}.pt"
        device = infer_model_device(model) or "cuda"
        return _load_checkpoint(model, full_path, state_path, f"{compression}+AT", device)

    if defense == "at_kd":
        ckpt_dir = _ROOT / cfg["paths"]["checkpoints_atkd_dir"]
        full_path  = ckpt_dir / f"atkd_{compression}_epoch{epochs:02d}_full_model.pt"
        state_path = ckpt_dir / f"atkd_{compression}_epoch{epochs:02d}.pt"
        device = infer_model_device(model) or "cuda"
        return _load_checkpoint(model, full_path, state_path, f"{compression}+AT+KD", device)

    raise ValueError(f"Unknown defense: {defense!r}")


# ── Resumability helpers ──────────────────────────────────────────────────────

def load_completed_runs(results_path: str) -> set:
    """Return the set of (model, compression, defense, attack) tuples already in the CSV."""
    completed: set = set()
    if not os.path.isfile(results_path):
        return completed
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["model"], row["compression"], row["defense"], row["attack"]))
    return completed


def append_row(
    results_path: str,
    model_name: str,
    compression: str,
    defense: str,
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
            "defense": defense,
            "attack": ATTACK_NAME,
            "clean_acc": round(c_acc, 6),
            "robust_acc": round(rob_acc, 6),
            "asr": round(asr, 6),
            "robustness_gap": round(rob_gap, 6),
            "phase": PHASE,
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
    """Collect clean logits and labels for the full loader."""
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
    """Run the attack on every batch and collect adversarial logits."""
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
    """Print a formatted table of all phase3 rows for the given model."""
    if not os.path.isfile(results_path):
        return
    header = (
        f"\n{'compression':<12} {'defense':<8} {'clean_acc':>10} "
        f"{'robust_acc':>11} {'asr':>8} {'gap':>8}"
    )
    print(header)
    print("-" * len(header.strip()))
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != model_name or row["attack"] != ATTACK_NAME:
                continue
            print(
                f"{row['compression']:<12} {row['defense']:<8} "
                f"{float(row['clean_acc']):>10.4f} "
                f"{float(row['robust_acc']):>11.4f} "
                f"{float(row['asr']):>8.4f} "
                f"{float(row['robustness_gap']):>8.4f}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 — combined attack vs all defenses × compression levels."
    )
    parser.add_argument(
        "--model",
        default="deit_small",
        choices=["deit_small"],
        help="Model to evaluate (default: deit_small)",
    )
    args = parser.parse_args()
    model_name: str = args.model

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global RESULTS_FILE
    dataset = cfg["dataset"]["name"]
    RESULTS_FILE = f"results/{dataset}_phase3_results.csv"

    print(f"[phase3] device  : {device}")
    print(f"[phase3] model   : {model_name}")
    print(f"[phase3] results : {RESULTS_FILE}")
    print(f"[phase3] attack  : combined (FGSM → PGD → Patch)")
    print()

    val_loader = build_val_loader(cfg, device)
    completed = load_completed_runs(RESULTS_FILE)

    if completed:
        print(f"[phase3] Resuming — {len(completed)} combination(s) already done, skipping those.\n")

    compression_levels: list[str] = cfg["compression"]["levels"]

    for compression in compression_levels:
        remaining_defenses = [
            d for d in DEFENSE_NAMES
            if (model_name, compression, d, ATTACK_NAME) not in completed
        ]

        if not remaining_defenses:
            print(f"[phase3] {compression:<6}: all defenses already done — skipping model load.")
            continue

        for defense in remaining_defenses:
            label = f"{compression}+{defense}"
            print(f"[phase3] {label}: loading {model_name} …")
            try:
                raw_model = load_model(
                    model_name,
                    compression,
                    cfg,
                    device=device,
                    dataset=cfg["dataset"]["name"],
                )
            except Exception as exc:
                print(f"[phase3] {label}: load failed — {exc}")
                continue

            # Apply defense checkpoint
            try:
                raw_model = load_defended_model(raw_model, compression, defense, cfg)
            except Exception as exc:
                print(f"[phase3] {label}: checkpoint load failed — {exc}")
                del raw_model
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            model = LogitsWrapper(raw_model)
            model.eval()
            model_device = infer_model_device(raw_model)
            print(f"[phase3] {label}: model on {model_device}")

            # Clean accuracy
            print(f"[phase3] {label}: evaluating clean accuracy …")
            try:
                clean_logits, clean_labels = run_clean_eval(model, val_loader, model_device)
            except Exception as exc:
                print(f"[phase3] {label}: clean eval failed — {exc}")
                del raw_model, model
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            c_acc = clean_accuracy(clean_logits, clean_labels)
            print(f"[phase3] {label}: clean_acc = {c_acc:.4f}")

            # Combined attack
            print(f"[phase3] {label}: building combined attack …")
            attack = combined_mod.build_attack(model, cfg)

            print(f"[phase3] {label}: running combined attack on {len(val_loader.dataset)} images …")
            try:
                adv_logits = run_adv_eval(attack, model, val_loader, model_device)
            except Exception as exc:
                print(f"[phase3] {label}: attack failed — {exc}")
                del raw_model, model
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            rob_acc = robust_accuracy(adv_logits, clean_labels)
            asr = attack_success_rate(adv_logits, clean_labels)
            rob_gap = robustness_gap(clean_logits, adv_logits, clean_labels)

            print(
                f"[phase3] {label}: "
                f"robust_acc={rob_acc:.4f}  asr={asr:.4f}  gap={rob_gap:.4f}"
            )

            append_row(RESULTS_FILE, model_name, compression, defense, c_acc, rob_acc, asr, rob_gap)
            print(f"[phase3] {label}: saved → {RESULTS_FILE}")

            del raw_model, model
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n[phase3] All combinations complete.")
    print_summary(RESULTS_FILE, model_name)


if __name__ == "__main__":
    main()
