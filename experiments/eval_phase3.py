"""
experiments/eval_phase3.py

Phase 3: Combined attack (FGSM → PGD → Patch) vs all defenses × all compression levels.

For each (compression, defense) pair the script:
  1. Loads the model at the given compression level.
  2. Loads the saved defense checkpoint (AT or AT+KD), or uses the plain model
     for the no-defense baseline.
  3. Evaluates the combined attack.
  4. Writes one row to results/phase3_results.csv immediately.

Results are written immediately after each pair completes — the script is safe
to interrupt and re-run (resumable).

Expected headline result: combined attack collapses robustness across all
defenses at INT4, with the degree of collapse being the key finding.
AT+KD degrades more gracefully than AT alone — the gap between them under
combined attack, and how it narrows at INT4, is the core contribution.

Usage:
    python experiments/eval_phase3.py                    # full sweep
    python experiments/eval_phase3.py --compression int4 # one level
    python experiments/eval_phase3.py --defense at       # one defense
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

from models.loader import load_config, load_model
import attacks.combined as combined_mod
from utils.metrics import clean_accuracy, robust_accuracy, attack_success_rate, robustness_gap

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_FILE = "results/phase3_results.csv"
PHASE        = 3
ATTACK_NAME  = "combined"
FIELDNAMES   = [
    "timestamp", "model", "compression", "defense", "attack",
    "clean_acc", "robust_acc", "asr", "robustness_gap", "phase",
]
DEFENSES     = ["none", "at", "at_kd"]


# ── LogitsWrapper ─────────────────────────────────────────────────────────────

class LogitsWrapper(nn.Module):
    """Unwrap HuggingFace model output to a plain (N, C) logits tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


# ── ImageNette → ImageNet-1k label remapping ──────────────────────────────────

_IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764": 0,    "n02102040": 217,  "n02979186": 482,
    "n03000684": 491,  "n03028079": 497,  "n03394916": 566,
    "n03417042": 569,  "n03425413": 571,  "n03445777": 574,
    "n03888257": 701,
}

def _remap_subset_labels(dataset: ImageFolder) -> ImageFolder:
    if len(dataset.classes) >= 1000:
        return dataset
    new_samples = []
    for path, lbl in dataset.samples:
        synset = dataset.classes[lbl]
        new_samples.append((path, _IMAGENETTE_TO_IMAGENET.get(synset, lbl)))
    dataset.samples = new_samples
    dataset.targets = [lbl for _, lbl in new_samples]
    return dataset


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_val_loader(cfg: dict, device: str) -> DataLoader:
    ds_cfg   = cfg["dataset"]
    eval_cfg = cfg["eval"]
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])
    full_dataset = _remap_subset_labels(
        ImageFolder(root=str(_ROOT / ds_cfg["val_dir"]), transform=transform)
    )
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n = min(ds_cfg["val_subset_size"], len(full_dataset))
    indices = torch.randperm(len(full_dataset), generator=rng)[:n].tolist()
    print(f"[phase3] Val subset: {n} images, seed={cfg['seed']}, first 5={indices[:5]}")
    return DataLoader(
        Subset(full_dataset, indices),
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )


def build_patch_val_loader(cfg: dict, device: str) -> DataLoader:
    """500-image subset for the patch stage of the combined attack."""
    ds_cfg   = cfg["dataset"]
    eval_cfg = cfg["eval"]
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])
    full_dataset = _remap_subset_labels(
        ImageFolder(root=str(_ROOT / ds_cfg["val_dir"]), transform=transform)
    )
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n_full = min(ds_cfg["val_subset_size"], len(full_dataset))
    full_indices = torch.randperm(len(full_dataset), generator=rng)[:n_full].tolist()
    patch_indices = full_indices[:500]
    print(f"[phase3] Patch val subset: 500 images (prefix of full {n_full})")
    return DataLoader(
        Subset(full_dataset, patch_indices),
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )


# ── Checkpoint loaders ────────────────────────────────────────────────────────

def _load_checkpoint(
    model: nn.Module,
    ckpt_dir: str,
    prefix: str,
    compression: str,
    epochs: int,
) -> nn.Module:
    """Load an AT or AT+KD checkpoint.

    Args:
        model:       Already-loaded compressed nn.Module (used for state_dict loading).
        ckpt_dir:    Checkpoint directory path string.
        prefix:      "at" or "atkd" — determines filename prefix.
        compression: "fp32", "int8", or "int4".
        epochs:      Number of training epochs (determines which epoch file to load).

    Returns:
        model in eval mode with checkpoint weights applied.

    Raises:
        FileNotFoundError: If neither checkpoint file is found.
    """
    full_path = _ROOT / ckpt_dir / f"{prefix}_{compression}_epoch{epochs:02d}_full_model.pt"
    sd_path   = _ROOT / ckpt_dir / f"{prefix}_{compression}_epoch{epochs:02d}.pt"

    if full_path.is_file():
        print(f"[phase3] Loading full model checkpoint: {full_path}")
        loaded = torch.load(str(full_path), map_location="cpu", weights_only=False)
        loaded.eval()
        return loaded

    if sd_path.is_file():
        print(f"[phase3] Loading state dict checkpoint: {sd_path}")
        state = torch.load(str(sd_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    raise FileNotFoundError(
        f"No {prefix.upper()} checkpoint found for compression='{compression}'.  Tried:\n"
        f"  {full_path}\n"
        f"  {sd_path}\n"
        f"Run experiments/eval_phase2_{prefix}.py first to generate the checkpoint."
    )


def load_defended_model(
    model_name: str,
    compression: str,
    defense: str,
    cfg: dict,
    device: str,
) -> nn.Module:
    """Load model at given compression and apply the specified defense.

    Args:
        model_name:  "deit_small"
        compression: "fp32", "int8", "int4"
        defense:     "none", "at", or "at_kd"
        cfg:         Parsed base.yaml config dict.
        device:      "cuda" or "cpu"

    Returns:
        nn.Module in eval mode with defense applied (or plain model for "none").

    Raises:
        FileNotFoundError: If the required checkpoint does not exist.
        ValueError:        If defense is not one of the three valid values.
    """
    raw_model = load_model(model_name, compression, cfg, device=device)
    epochs    = cfg["defense"]["epochs"]

    if defense == "none":
        raw_model.eval()
        return raw_model

    if defense == "at":
        return _load_checkpoint(
            raw_model,
            cfg["paths"]["checkpoints_at_dir"],
            "at", compression, epochs,
        )

    if defense == "at_kd":
        return _load_checkpoint(
            raw_model,
            cfg["paths"]["checkpoints_atkd_dir"],
            "atkd", compression, epochs,
        )

    raise ValueError(f"Unknown defense: {defense!r}. Choose from: none, at, at_kd")


# ── Resumability ──────────────────────────────────────────────────────────────

def load_completed_runs(results_path: str) -> set:
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
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model":          model_name,
            "compression":    compression,
            "defense":        defense,
            "attack":         ATTACK_NAME,
            "clean_acc":      round(c_acc, 6),
            "robust_acc":     round(rob_acc, 6),
            "asr":            round(asr, 6),
            "robustness_gap": round(rob_gap, 6),
            "phase":          PHASE,
        })


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_model_device(model: nn.Module) -> str:
    for p in model.parameters():
        return str(p.device)
    return "cpu"


@torch.no_grad()
def run_clean_eval(
    model: nn.Module,
    loader: DataLoader,
    model_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_list, labels_list = [], []
    for images, labels in loader:
        images = images.to(model_device)
        logits_list.append(model(images).cpu())
        labels_list.append(labels.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def run_adv_eval(
    attack,
    model: nn.Module,
    loader: DataLoader,
    model_device: str,
) -> torch.Tensor:
    adv_logits_list = []
    for images, labels in loader:
        images = images.to(model_device)
        labels = labels.to(model_device)
        adv_images = attack(images, labels)
        with torch.no_grad():
            adv_logits_list.append(model(adv_images).cpu())
    return torch.cat(adv_logits_list)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results_path: str, model_name: str) -> None:
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
        description="Phase 3 — Combined attack vs all defenses × all compression levels."
    )
    parser.add_argument("--model",       default="deit_small", choices=["deit_small"])
    parser.add_argument("--compression", choices=["fp32", "int8", "int4"], default=None)
    parser.add_argument("--defense",     choices=["none", "at", "at_kd"],  default=None)
    args = parser.parse_args()

    model_name: str = args.model
    cfg    = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[phase3] device  : {device}")
    print(f"[phase3] model   : {model_name}")
    print(f"[phase3] results : {RESULTS_FILE}")
    print()

    # Combined attack uses the patch stage on 500 images (same as phases 1–2)
    # and FGSM/PGD on the full val subset — but since it's sequential the whole
    # pipeline runs on the patch subset to keep compute within budget.
    val_loader = build_patch_val_loader(cfg, device)

    completed = load_completed_runs(RESULTS_FILE)
    if completed:
        print(f"[phase3] Resuming — {len(completed)} pair(s) already done, skipping.\n")

    all_compressions: list[str] = cfg["compression"]["levels"]
    compressions = [args.compression] if args.compression else all_compressions
    defenses     = [args.defense]     if args.defense     else DEFENSES

    for compression in compressions:
        for defense in defenses:
            key = (model_name, compression, defense, ATTACK_NAME)
            if key in completed:
                print(f"[phase3] {compression:<6} × {defense:<6}: already done — skipping.")
                continue

            print(f"\n[phase3] {compression:<6} × {defense:<6}: loading model …")
            try:
                raw_model = load_defended_model(
                    model_name, compression, defense, cfg, device
                )
            except FileNotFoundError as exc:
                print(f"[phase3] {compression:<6} × {defense:<6}: SKIPPED — {exc}")
                continue
            except Exception as exc:
                print(f"[phase3] {compression:<6} × {defense:<6}: load failed — {exc}")
                continue

            model       = LogitsWrapper(raw_model)
            model.eval()
            model_device = infer_model_device(raw_model)
            print(f"[phase3] {compression:<6} × {defense:<6}: model on {model_device}")

            # Clean accuracy
            print(f"[phase3] {compression:<6} × {defense:<6}: evaluating clean accuracy …")
            try:
                clean_logits, clean_labels = run_clean_eval(model, val_loader, model_device)
            except Exception as exc:
                print(f"[phase3] {compression:<6} × {defense:<6}: clean eval failed — {exc}")
                del raw_model, model
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            c_acc = clean_accuracy(clean_logits, clean_labels)
            print(f"[phase3] {compression:<6} × {defense:<6}: clean_acc = {c_acc:.4f}")

            # Combined attack
            print(f"[phase3] {compression:<6} × {defense:<6}: building combined attack …")
            attack = combined_mod.build_attack(model, cfg)

            print(
                f"[phase3] {compression:<6} × {defense:<6}: "
                f"running on {len(val_loader.dataset)} images …"
            )
            try:
                adv_logits = run_adv_eval(attack, model, val_loader, model_device)
            except Exception as exc:
                print(f"[phase3] {compression:<6} × {defense:<6}: attack failed — {exc}")
                del raw_model, model
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            rob_acc  = robust_accuracy(adv_logits, clean_labels)
            asr      = attack_success_rate(adv_logits, clean_labels)
            rob_gap  = robustness_gap(clean_logits, adv_logits, clean_labels)

            print(
                f"[phase3] {compression:<6} × {defense:<6}: "
                f"robust_acc={rob_acc:.4f}  asr={asr:.4f}  gap={rob_gap:.4f}"
            )

            append_row(
                RESULTS_FILE, model_name, compression, defense,
                c_acc, rob_acc, asr, rob_gap,
            )
            print(f"[phase3] {compression:<6} × {defense:<6}: saved → {RESULTS_FILE}")

            del raw_model, model
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n[phase3] All combinations complete.")
    print_summary(RESULTS_FILE, model_name)


if __name__ == "__main__":
    main()
