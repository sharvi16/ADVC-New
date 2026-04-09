"""
attacks/fgsm.py

FGSM attack wrapper using torchattacks.
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from attacks.fgsm import load_config, build_attack
    cfg = load_config()
    attack = build_attack(model, config=cfg)
    adv_images = attack(images, labels)
"""

import yaml
import torch
import torchattacks
from models.loader import load_config  # noqa: F401 — re-exported for convenience


def build_attack(
    model: torch.nn.Module,
    config: dict,
) -> torchattacks.FGSM:
    """
    Build an FGSM attack bound to the given model.

    Args:
        model:  A torch.nn.Module in eval mode.
        config: Parsed base.yaml config dict.

    Returns:
        attack: torchattacks.FGSM instance ready for inference.
    """
    eps = config["attack"]["eps"]
    attack = torchattacks.FGSM(model, eps=eps)
    return attack


def run_attack(
    attack: torchattacks.FGSM,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Generate adversarial examples for a batch.

    Args:
        attack: A configured torchattacks.FGSM instance.
        images: Clean input batch, shape (N, 3, H, W), values in [0, 1].
        labels: Ground-truth class indices, shape (N,).

    Returns:
        adv_images: Adversarial batch, same shape and device as images.
    """
    adv_images = attack(images, labels)
    return adv_images


def print_attack_info(config: dict) -> None:
    """Print a quick summary of the configured attack."""
    eps = config["attack"]["eps"]
    print(f"[fgsm] Attack : FGSM")
    print(f"       eps    : {eps:.5f}  ({round(eps * 255)}/255)")
    print(f"       norm   : L-inf")


# Sanity check — run directly to verify attack builds and runs
if __name__ == "__main__":
    from models.loader import load_model

    cfg = load_config()
    print("=== Sanity check: FGSM on DeiT-S FP32 ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("deit_small", "fp32", cfg, device=device)

    print_attack_info(cfg)

    attack = build_attack(model, cfg)

    dummy_images = torch.rand(2, 3, 224, 224).to(device)
    dummy_labels = torch.zeros(2, dtype=torch.long).to(device)

    adv_images = run_attack(attack, dummy_images, dummy_labels)

    print(f"\n[fgsm] Input  shape : {dummy_images.shape}")
    print(f"[fgsm] Output shape : {adv_images.shape}")
    print(f"[fgsm] Max perturbation: {(adv_images - dummy_images).abs().max():.5f}")
    print(f"[fgsm] Expected max    : {cfg['attack']['eps']:.5f}")
