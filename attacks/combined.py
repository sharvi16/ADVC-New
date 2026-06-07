"""
attacks/combined.py

Combined attack: chains FGSM → PGD → Patch sequentially on each batch.
The output of each attack is the input to the next.
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from attacks.combined import load_config, build_attack
    cfg = load_config()
    attack = build_attack(model, config=cfg)
    adv_images = attack(images, labels)
"""

import torch
from models.loader import load_config  # noqa: F401 — re-exported for convenience
import attacks.fgsm as fgsm_mod
import attacks.pgd as pgd_mod
import attacks.patch as patch_mod


class CombinedAttack:
    """
    Sequential combination attack: FGSM → PGD → Patch.

    Each attack is applied in order; the adversarial output of one becomes
    the input of the next.  All three attacks share the same labels and
    operate in ImageNet-normalised space.

    Args:
        fgsm:  Configured FGSM attack instance.
        pgd:   Configured PGD attack instance.
        patch: Configured PatchAttack instance.
    """

    def __init__(self, fgsm, pgd, patch) -> None:
        self.fgsm = fgsm
        self.pgd = pgd
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
            adv_images: Adversarial batch (still normalised), same shape and
                        device as images.
        """
        adv = self.fgsm(images, labels)
        adv = self.pgd(adv, labels)
        adv = self.patch(adv, labels)
        return adv


def build_attack(
    model: torch.nn.Module,
    config: dict,
) -> CombinedAttack:
    """
    Build a combined FGSM+PGD+Patch attack bound to the given model.

    Args:
        model:  A torch.nn.Module in eval mode.
        config: Parsed base.yaml config dict.

    Returns:
        attack: CombinedAttack instance ready for inference.
    """
    fgsm = fgsm_mod.build_attack(model, config)
    pgd = pgd_mod.build_attack(model, config)
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


def print_attack_info(config: dict) -> None:
    """Print a quick summary of the configured combined attack."""
    print(f"[combined] Attack  : FGSM → PGD → Patch (sequential)")
    print(f"           fgsm    : eps={config['fgsm']['eps']:.5f} ({round(config['fgsm']['eps']*255)}/255)")
    print(f"           pgd     : eps={config['pgd']['eps']:.5f}, alpha={config['pgd']['alpha']:.5f}, steps={config['pgd']['steps']}")
    print(f"           patch   : size={config['patch']['patch_size']}x{config['patch']['patch_size']}, steps={config['patch']['steps']}")


# Sanity check — run directly to verify attack builds and runs
if __name__ == "__main__":
    from models.loader import load_model

    cfg = load_config()
    print("=== Sanity check: Combined attack on DeiT-S FP32 ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("deit_small", "fp32", cfg, device=device)

    print_attack_info(cfg)

    attack = build_attack(model, cfg)

    dummy_images = torch.rand(2, 3, 224, 224).to(device)
    dummy_labels = torch.zeros(2, dtype=torch.long).to(device)

    adv_images = run_attack(attack, dummy_images, dummy_labels)

    print(f"\n[combined] Input  shape : {dummy_images.shape}")
    print(f"[combined] Output shape : {adv_images.shape}")
    print(f"[combined] Max pixel delta : {(adv_images - dummy_images).abs().max():.5f}")
