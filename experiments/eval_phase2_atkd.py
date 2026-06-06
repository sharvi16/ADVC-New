"""
experiments/eval_phase2_atkd.py

Phase 2b: Adversarial Training + Knowledge Distillation (AT+KD) defense sweep
across compression levels × attacks.

For each compression level the script:
  1. Loads DeiT-S at the given compression (student).
  2. Fine-tunes the student with AT+KD using a frozen FP32 teacher.
  3. Evaluates all 3 attacks (FGSM, PGD, Patch) on the defended student.
  4. Writes one row per (compression, attack) to results/phase2_atkd_results.csv.

The FP32 teacher is loaded once and reused across all compression levels.
It is never updated — teacher.eval() is enforced throughout.

Results are written immediately after each (compression, attack) pair completes.
Already-written rows are detected on startup and skipped, so the script is safe
to interrupt and re-run (resumable).

Usage:
    python experiments/eval_phase2_atkd.py                          # all compression levels
    python experiments/eval_phase2_atkd.py --compression int8       # one level only
    python experiments/eval_phase2_atkd.py --compression fp32 --skip-training
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
import attacks.fgsm as fgsm_mod
import attacks.pgd as pgd_mod
import attacks.patch as patch_mod
from defenses.at_kd import at_kd_train
from utils.metrics import (
    clean_accuracy,
    robust_accuracy,
    attack_success_rate,
    robustness_gap,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_FILE = "results/phase2_atkd_results.csv"
DEFENSE_NAME = "at_kd"
PHASE = 2
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


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_val_loader(cfg: dict, device: str) -> DataLoader:
    """Build a deterministic validation subset loader.

    Uses seed=42 via randperm, matching configs/base.yaml so results are
    reproducible across runs.
    """
    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["eval"]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])

    val_path = resolve_data_path(_ROOT, ds_cfg["val_dir"])
    print(f"[phase2-ATKD] val_dir   : {val_path}")
    full_dataset = ImageFolder(root=str(val_path), transform=transform)
    full_dataset = _remap_subset_labels(full_dataset)

    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n = min(ds_cfg["val_subset_size"], len(full_dataset))
    indices = torch.randperm(len(full_dataset), generator=rng)[:n].tolist()
    print(f"[phase2-ATKD] Val subset : {n} images, seed={cfg['seed']}, first 5 indices={indices[:5]}")
    subset = Subset(full_dataset, indices)

    return DataLoader(
        subset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )


def build_patch_val_loader(cfg: dict, device: str) -> DataLoader:
    """Build a smaller validation loader used exclusively for the patch attack.

    The patch attack runs 150 PGD-style optimisation steps per batch, making
    it ~7.5× more expensive than FGSM/PGD.  Using 500 images instead of the
    full val subset keeps patch evaluation tractable while
    still producing a statistically meaningful ASR estimate.
    FGSM and PGD always use the full val_subset_size loader.
    """
    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["eval"]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])

    val_path = resolve_data_path(_ROOT, ds_cfg["val_dir"])
    full_dataset = ImageFolder(root=str(val_path), transform=transform)
    full_dataset = _remap_subset_labels(full_dataset)

    # Draw from the same shuffled order as build_val_loader so the 500 images
    # are a strict prefix of the full val subset — results stay comparable.
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n_full = min(ds_cfg["val_subset_size"], len(full_dataset))
    full_indices = torch.randperm(len(full_dataset), generator=rng)[:n_full].tolist()
    patch_indices = full_indices[:500]
    print(f"[phase2-ATKD] Patch val subset : 500 images (subset of full {n_full}), seed={cfg['seed']}")
    subset = Subset(full_dataset, patch_indices)

    return DataLoader(
        subset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )


def build_train_loader(cfg: dict, device: str) -> DataLoader:
    """Build a deterministic training subset loader for AT+KD fine-tuning.

    Uses seed=42 via randperm matching configs/base.yaml.
    Batch size is cfg["defense"]["batch_size"] (32 per CLAUDE.md).
    """
    ds_cfg = cfg["dataset"]
    defense_cfg = cfg["defense"]

    transform = T.Compose([
        T.RandomResizedCrop(ds_cfg["image_size"]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])

    train_path = resolve_data_path(_ROOT, ds_cfg["train_dir"])
    print(f"[phase2-ATKD] train_dir : {train_path}")
    full_dataset = ImageFolder(root=str(train_path), transform=transform)
    full_dataset = _remap_subset_labels(full_dataset)

    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n = min(ds_cfg["train_subset_size"], len(full_dataset))
    indices = torch.randperm(len(full_dataset), generator=rng)[:n].tolist()
    subset = Subset(full_dataset, indices)

    return DataLoader(
        subset,
        batch_size=defense_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["eval"]["num_workers"],
        pin_memory=(device == "cuda"),
    )


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
            "defense": DEFENSE_NAME,
            "attack": attack_name,
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
    """Collect clean logits and labels for the full validation loader.

    Returns:
        all_logits: (N, C) tensor on CPU.
        all_labels: (N,)   tensor on CPU.
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
    """Print a formatted table of all phase2-AT+KD rows for the given model."""
    if not os.path.isfile(results_path):
        return
    header = (
        f"\n{'compression':<12} {'attack':<8} {'clean_acc':>10} "
        f"{'robust_acc':>11} {'asr':>8} {'gap':>8}"
    )
    print(header)
    print("-" * len(header.strip()))
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != model_name or row["defense"] != DEFENSE_NAME:
                continue
            print(
                f"{row['compression']:<12} {row['attack']:<8} "
                f"{float(row['clean_acc']):>10.4f} "
                f"{float(row['robust_acc']):>11.4f} "
                f"{float(row['asr']):>8.4f} "
                f"{float(row['robustness_gap']):>8.4f}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def load_atkd_checkpoint(
    model: nn.Module,
    compression: str,
    cfg: dict,
) -> nn.Module:
    """Load the final AT+KD checkpoint for the given compression level.

    INT4 checkpoints are saved as full model objects
    (atkd_{compression}_epoch{N:02d}_full_model.pt) because bitsandbytes NF4
    embeds absmax / quant_state metadata into parameter tensors — reloading a
    state_dict into a freshly-quantised model causes a state conflict.

    FP32 and INT8 checkpoints are saved as plain state_dicts
    (atkd_{compression}_epoch{N:02d}.pt).

    The function detects which format is on disk and loads accordingly.
    For full-model checkpoints the `model` argument is ignored; a freshly
    deserialised object is returned instead.

    Args:
        model:       Already-loaded (and compressed) nn.Module.
                     Used only for state_dict loading (fp32 / int8).
        compression: Compression level string — "fp32", "int8", or "int4".
        cfg:         Parsed base.yaml config dict.

    Returns:
        model: nn.Module with AT+KD weights, in eval mode.

    Raises:
        FileNotFoundError: If neither checkpoint file is found on disk.
    """
    ckpt_dir = cfg["paths"]["checkpoints_atkd_dir"]
    epochs: int = cfg["defense"]["epochs"]

    full_model_path = _ROOT / ckpt_dir / f"atkd_{compression}_epoch{epochs:02d}_full_model.pt"
    state_dict_path = _ROOT / ckpt_dir / f"atkd_{compression}_epoch{epochs:02d}.pt"

    if full_model_path.is_file():
        # INT4: full model serialised — deserialise directly
        print(
            f"[phase2-ATKD] {compression:<6}: loading full model checkpoint "
            f"from {full_model_path} …"
        )
        loaded_model = torch.load(str(full_model_path), map_location="cpu")
        loaded_model.eval()
        print(f"[phase2-ATKD] {compression:<6}: full model checkpoint loaded.")
        return loaded_model

    if state_dict_path.is_file():
        # FP32 / INT8: plain state_dict — load into the provided model
        print(
            f"[phase2-ATKD] {compression:<6}: loading state dict checkpoint "
            f"from {state_dict_path} …"
        )
        state_dict = torch.load(str(state_dict_path), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[phase2-ATKD] {compression:<6}: state dict checkpoint loaded.")
        return model

    raise FileNotFoundError(
        f"No AT+KD checkpoint found for compression='{compression}'.  Tried:\n"
        f"  {full_model_path}\n"
        f"  {state_dict_path}\n"
        "Run without --skip-training first to generate the checkpoint, "
        "or copy it from your backup location."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2b — AT+KD defense sweep: compression × attack."
    )
    parser.add_argument(
        "--model",
        default="deit_small",
        choices=["deit_small"],
        help="Model to evaluate (default: deit_small)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip AT+KD fine-tuning and load the saved final-epoch checkpoint "
            "from results/checkpoints/atkd/ instead.  Useful when AT+KD has already "
            "run and you only want to re-evaluate the defended student."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=["int8", "int4"],
        default=None,
        help="Run only this compression level. Default runs all three.",
    )
    args = parser.parse_args()
    model_name: str = args.model
    skip_training: bool = args.skip_training

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[phase2-ATKD] device        : {device}")
    print(f"[phase2-ATKD] model         : {model_name}")
    print(f"[phase2-ATKD] results       : {RESULTS_FILE}")
    print(f"[phase2-ATKD] skip-training : {skip_training}")
    print()

    val_loader = build_val_loader(cfg, device)
    patch_val_loader = build_patch_val_loader(cfg, device)
    train_loader = build_train_loader(cfg, device) if not skip_training else None
    completed = load_completed_runs(RESULTS_FILE)

    if completed:
        print(
            f"[phase2-ATKD] Resuming — {len(completed)} combination(s) already done, "
            "skipping those.\n"
        )

    # ── Load FP32 teacher once — reused across all compression levels ──────────
    # Teacher is always FP32 DeiT-S. It must never be updated.
    print(f"[phase2-ATKD] Loading FP32 teacher …")
    teacher = load_model(model_name, "fp32", cfg, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_device = infer_model_device(teacher)
    print(f"[phase2-ATKD] Teacher on {teacher_device} (frozen)\n")

    all_levels: list[str] = cfg["compression"]["levels"]
    compression_levels: list[str] = [args.compression] if args.compression else all_levels

    for compression in compression_levels:
        remaining = [
            a for a in ATTACK_NAMES
            if (model_name, compression, DEFENSE_NAME, a) not in completed
        ]

        if not remaining:
            print(
                f"[phase2-ATKD] {compression:<6}: all attacks already done — "
                "skipping model load."
            )
            continue

        # ── Load compressed student ────────────────────────────────────────────
        print(f"[phase2-ATKD] {compression:<6}: loading {model_name} (student) …")
        try:
            raw_student = load_model(model_name, compression, cfg, device=device)
        except Exception as exc:
            print(f"[phase2-ATKD] {compression:<6}: load failed — {exc}")
            continue

        # ── Apply AT+KD defense (or load saved checkpoint) ────────────────────
        if skip_training:
            print(f"[phase2-ATKD] {compression:<6}: --skip-training set, loading checkpoint …")
            try:
                raw_student = load_atkd_checkpoint(raw_student, compression, cfg)
            except Exception as exc:
                print(f"[phase2-ATKD] {compression:<6}: checkpoint load failed — {exc}")
                del raw_student
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
        else:
            print(f"[phase2-ATKD] {compression:<6}: applying AT+KD …")
            try:
                raw_student = at_kd_train(
                    raw_student, teacher, train_loader, cfg, compression=compression
                )
            except Exception as exc:
                print(f"[phase2-ATKD] {compression:<6}: AT+KD failed — {exc}")
                del raw_student
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

        student = LogitsWrapper(raw_student)
        student.eval()
        student_device = infer_model_device(raw_student)
        mode_label = "checkpoint" if skip_training else "post-AT+KD"
        print(f"[phase2-ATKD] {compression:<6}: student on {student_device} ({mode_label})")

        # ── Clean accuracy (computed once, reused for all attacks) ────────────
        print(f"[phase2-ATKD] {compression:<6}: evaluating clean accuracy …")
        try:
            clean_logits, clean_labels = run_clean_eval(
                student, val_loader, student_device
            )
        except Exception as exc:
            print(f"[phase2-ATKD] {compression:<6}: clean eval failed — {exc}")
            del raw_student, student
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        c_acc = clean_accuracy(clean_logits, clean_labels)
        print(f"[phase2-ATKD] {compression:<6}: clean_acc = {c_acc:.4f}")

        # ── Per-attack robustness sweep ────────────────────────────────────────
        for attack_name in remaining:
            print(
                f"[phase2-ATKD] {compression:<6} × {attack_name:<5}: "
                "building attack …"
            )

            # Patch attack is evaluated on a smaller 500-image subset because
            # each batch requires 150 PGD optimisation steps, making it far
            # more expensive than FGSM/PGD.  FGSM and PGD use the full loader.
            if attack_name == "patch":
                eval_loader = patch_val_loader
                # Re-collect clean logits/labels for the 500-image subset so
                # that clean_acc and robustness_gap are computed on the same
                # images as the adversarial evaluation.
                try:
                    patch_clean_logits, patch_clean_labels = run_clean_eval(
                        student, patch_val_loader, student_device
                    )
                except Exception as exc:
                    print(
                        f"[phase2-ATKD] {compression:<6} × patch  : "
                        f"patch clean eval failed — {exc}"
                    )
                    continue
                eval_clean_logits = patch_clean_logits
                eval_clean_labels = patch_clean_labels
            else:
                eval_loader = val_loader
                eval_clean_logits = clean_logits
                eval_clean_labels = clean_labels

            if attack_name == "fgsm":
                attack = fgsm_mod.build_attack(student, cfg)
            elif attack_name == "pgd":
                attack = pgd_mod.build_attack(student, cfg)
            elif attack_name == "patch":
                attack = patch_mod.build_attack(student, cfg)
            else:
                raise ValueError(f"Unknown attack: {attack_name!r}")

            print(
                f"[phase2-ATKD] {compression:<6} × {attack_name:<5}: "
                f"running on {len(eval_loader.dataset)} images …"
            )
            try:
                adv_logits = run_adv_eval(attack, student, eval_loader, student_device)
            except Exception as exc:
                print(
                    f"[phase2-ATKD] {compression:<6} × {attack_name:<5}: "
                    f"attack failed — {exc}"
                )
                continue

            rob_acc = robust_accuracy(adv_logits, eval_clean_labels)
            asr = attack_success_rate(adv_logits, eval_clean_labels)
            rob_gap = robustness_gap(eval_clean_logits, adv_logits, eval_clean_labels)

            print(
                f"[phase2-ATKD] {compression:<6} × {attack_name:<5}: "
                f"robust_acc={rob_acc:.4f}  asr={asr:.4f}  gap={rob_gap:.4f}"
            )

            append_row(
                RESULTS_FILE, model_name, compression, attack_name,
                c_acc, rob_acc, asr, rob_gap,
            )
            print(
                f"[phase2-ATKD] {compression:<6} × {attack_name:<5}: "
                f"saved → {RESULTS_FILE}"
            )

        # ── Free student GPU memory; teacher stays loaded for the next level ───
        del raw_student, student
        if device == "cuda":
            torch.cuda.empty_cache()

    # Teacher released after all compression levels are done.
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n[phase2-ATKD] All combinations complete.")
    print_summary(RESULTS_FILE, model_name)


if __name__ == "__main__":
    main()
