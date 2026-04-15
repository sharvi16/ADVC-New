"""
Metrics for adversarial robustness evaluation.

All per-batch functions accept raw logits (or softmax outputs) and integer labels
as torch.Tensor and return a Python float in [0, 1].
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def clean_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of clean examples correctly classified.

    Args:
        logits: (N, C) model outputs (pre- or post-softmax).
        labels: (N,) integer ground-truth class indices.

    Returns:
        Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def robust_accuracy(adv_logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of adversarial examples correctly classified.

    Args:
        adv_logits: (N, C) model outputs on adversarial inputs.
        labels: (N,) integer ground-truth class indices.

    Returns:
        Robust accuracy in [0, 1].
    """
    preds = adv_logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def attack_success_rate(adv_logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of adversarial examples that fool the model (untargeted).

    ASR = 1 - robust_accuracy when evaluated on originally-correct examples only.
    Here we compute it over the full batch for simplicity; callers can pre-filter
    to only examples the clean model got right if a stricter definition is needed.

    Args:
        adv_logits: (N, C) model outputs on adversarial inputs.
        labels: (N,) integer ground-truth class indices.

    Returns:
        Attack success rate in [0, 1].
    """
    preds = adv_logits.argmax(dim=1)
    return (preds != labels).float().mean().item()


def robustness_gap(
    logits: torch.Tensor,
    adv_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Absolute drop in accuracy from clean to adversarial inputs.

    robustness_gap = clean_accuracy - robust_accuracy

    A larger gap indicates the model is more sensitive to the attack.

    Args:
        logits:     (N, C) model outputs on clean inputs.
        adv_logits: (N, C) model outputs on adversarial inputs.
        labels:     (N,) integer ground-truth class indices.

    Returns:
        Robustness gap in [0, 1].
    """
    return clean_accuracy(logits, labels) - robust_accuracy(adv_logits, labels)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

# Canonical column order for the results CSV.
# Matches the schema defined in CLAUDE.md — do not reorder.
_CSV_FIELDNAMES = [
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


def save_results_to_csv(
    results_dir: str,
    model: str,
    compression: str,
    defense: str,
    attack: str,
    clean_acc: float,
    robust_acc: float,
    asr: float,
    robustness_gap_val: float,
    phase: int,
    filename: str = "results.csv",
) -> str:
    """Append a result row to results/<filename>, creating it with headers if needed.

    Args:
        results_dir:        Path to the results directory (e.g. "results").
        model:              Model identifier, e.g. "deit_small".
        compression:        Compression level, one of "fp32", "int8", "int4".
        defense:            Defense applied, one of "none", "at", "at_kd".
        attack:             Attack used, one of "fgsm", "pgd", "patch", "combined".
        clean_acc:          Clean accuracy in [0, 1].
        robust_acc:         Robust accuracy in [0, 1].
        asr:                Attack success rate in [0, 1].
        robustness_gap_val: Robustness gap in [0, 1].
        phase:              Experiment phase: 1, 2, or 3.
        filename:           CSV filename inside results_dir.

    Returns:
        Absolute path to the CSV file.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(results_dir, filename)
    file_exists = os.path.isfile(csv_path)

    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "compression": compression,
        "defense": defense,
        "attack": attack,
        "clean_acc": round(clean_acc, 6),
        "robust_acc": round(robust_acc, 6),
        "asr": round(asr, 6),
        "robustness_gap": round(robustness_gap_val, 6),
        "phase": phase,
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return os.path.abspath(csv_path)
