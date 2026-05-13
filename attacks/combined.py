"""
attacks/combined.py

Combined attack: FGSM → PGD → Patch applied sequentially to the same input.
Each stage uses the same epsilon / config as the individual attacks in phases 1–2.

The output of each stage feeds into the next — the final adversarial image has
been perturbed by all three attacks in sequence, representing the worst-case
threat model for edge-deployed ViTs.

Usage:
    from attacks.combined import build_attack
    attack = build_attack(model, config)
    adv_images = attack(images, labels)
"""

import torch
import torch.nn as nn
from models.loader import load_config  # noqa: F401 — re-exported for convenience

import attacks.fgsm  as fgsm_mod
import attacks.pgd   as pgd_mod
import attacks.patch as patch_mod


class CombinedAttack:
    """
    Sequential combined attack: FGSM → PGD → Patch.

    Each sub-attack is applied to the output of the previous one.
    All parameters come from configs/base.yaml — never hardcode here.

    Stage order and rationale:
      1. FGSM  — fast global perturbation, seeds the adversarial direction
      2. PGD   — iterative refinement of the global perturbation
      3. Patch — localised high-magnitude perturbation on top of PGD output

    Args:
        fgsm:  Configured torchattacks.FGSM instance.
        pgd:   Configured torchattacks.PGD instance.
        patch: Configured PatchAttack instance.
    """

    def __init__(self, fgsm, pgd, patch) -> None:
        self.fgsm  = fgsm
        self.pgd   = pgd
        self.patch = patch

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FGSM → PGD → Patch sequentially.

        Args:
            images: ImageNet-normalised input batch, shape (N, 3, H, W).
            labels: Ground-truth class indices, shape (N,).

        Returns:
            adv_images: Adversarial batch, same shape and device as images.
        """
        adv = self.fgsm(images, labels)
        adv = self.pgd(adv, labels)
        adv = self.patch(adv, labels)
        return adv


def build_attack(
    model: nn.Module,
    config: dict,
) -> CombinedAttack:
    """
    Build a CombinedAttack bound to the given model.

    All three sub-attacks share the same model reference so gradients are
    computed against the same (defended or undefended) model throughout.

    Args:
        model:  A torch.nn.Module in eval mode, wrapped in LogitsWrapper so
                HuggingFace INT8/INT4 models return plain tensors.
        config: Parsed base.yaml config dict.

    Returns:
        CombinedAttack instance ready for inference.
    """
    fgsm  = fgsm_mod.build_attack(model, config)
    pgd   = pgd_mod.build_attack(model, config)
    patch = patch_mod.build_attack(model, config)
    return CombinedAttack(fgsm=fgsm, pgd=pgd, patch=patch)


def run_attack(
    attack: CombinedAttack,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Generate adversarial examples for a batch.

    Args:
        attack: A configured CombinedAttack instance.
        images: ImageNet-normalised input batch, shape (N, 3, H, W).
        labels: Ground-truth class indices, shape (N,).

    Returns:
        adv_images: Adversarial batch, same shape and device as images.
    """
    return attack(images, labels)


# Sanity check — run directly to verify the chain executes
if __name__ == "__main__":
    import torch
    from models.loader import load_model

    cfg = load_config()
    print("=== Sanity check: Combined attack on DeiT-S FP32 ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model("deit_small", "fp32", cfg, device=device)

    attack = build_attack(model, cfg)

    dummy_images = torch.rand(2, 3, 224, 224).to(device)
    dummy_labels = torch.zeros(2, dtype=torch.long).to(device)

    adv = run_attack(attack, dummy_images, dummy_labels)

    print(f"[combined] Input  shape : {dummy_images.shape}")
    print(f"[combined] Output shape : {adv.shape}")
    print(f"[combined] Max delta    : {(adv - dummy_images).abs().max():.5f}")
    print("[combined] Sanity check passed.")
