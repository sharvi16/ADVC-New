"""
attacks/patch.py

Adversarial patch attack.
A fixed-size square patch is placed at a random location on each image and
optimised via PGD-style gradient ascent to maximise the cross-entropy loss.
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from attacks.patch import load_config, build_attack
    cfg = load_config()
    attack = build_attack(model, config=cfg)
    adv_images = attack(images, labels)
"""

import torch
import torch.nn.functional as F
from models.loader import load_config  # noqa: F401 — re-exported for convenience


class PatchAttack:
    """
    Adversarial patch attack.

    Optimises a patch_size × patch_size square patch via PGD-style gradient
    ascent (sign updates, lr step size) to maximise the cross-entropy loss.
    The patch is placed at a uniformly random location for each forward call.

    Normalisation handling: inputs are expected to be ImageNet-normalised
    (values outside [0, 1]).  The attack internally un-normalises to [0, 1]
    before compositing, passes re-normalised images to the model, and returns
    re-normalised adversarial images so the output matches the input range.

    Args:
        model:      A torch.nn.Module in eval mode.
        patch_size: Side length of the square patch in pixels.
        steps:      Number of optimisation steps (PGD iterations).
        lr:         Step size for each gradient-sign update.
        mean:       ImageNet normalisation mean, list of 3 floats.
        std:        ImageNet normalisation std,  list of 3 floats.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int,
        steps: int,
        lr: float,
        mean: list,
        std: list,
    ) -> None:
        self.model = model
        self.patch_size = patch_size
        self.steps = steps
        self.lr = lr
        self.mean = mean
        self.std = std

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial examples by optimising a patch placed on each image.

        Args:
            images: ImageNet-normalised input batch, shape (N, 3, H, W).
            labels: Ground-truth class indices, shape (N,).

        Returns:
            adv_images: Perturbed batch (still normalised), same shape and
                        device as images.
        """
        device = images.device
        B, C, H, W = images.shape
        ps = self.patch_size

        assert ps <= H and ps <= W, (
            f"patch_size {ps} exceeds image dimensions ({H}x{W})"
        )

        # Normalisation tensors — shape (1, C, 1, 1) for broadcasting
        mean_t = torch.tensor(self.mean, dtype=images.dtype, device=device).view(1, C, 1, 1)
        std_t  = torch.tensor(self.std,  dtype=images.dtype, device=device).view(1, C, 1, 1)

        # Warn and un-normalise to [0, 1] for patch optimisation
        if images.max().item() > 2.0:
            print(
                f"[patch] Warning: input max={images.max().item():.3f} — "
                "images are not in [0, 1] (likely ImageNet-normalised). "
                "Un-normalising before patch optimisation."
            )

        images_unnorm = images * std_t + mean_t
        images_unnorm = images_unnorm.clamp(0.0, 1.0)
        assert images_unnorm.max().item() <= 1.0 + 1e-3, (
            f"Un-normalised images still out of [0, 1]: "
            f"max={images_unnorm.max().item():.4f}"
        )

        # Random top-left corner — same placement for all images in the batch
        row = torch.randint(0, H - ps + 1, (1,)).item()
        col = torch.randint(0, W - ps + 1, (1,)).item()

        # Spatial mask — 1 where patch goes, 0 elsewhere; broadcast over batch
        mask = torch.zeros(1, C, H, W, device=device)
        mask[:, :, row:row + ps, col:col + ps] = 1.0

        # Initialise patch uniformly in [0, 1]
        patch = torch.rand(C, ps, ps, device=device).requires_grad_(True)

        self.model.eval()
        for _ in range(self.steps):
            if patch.grad is not None:
                patch.grad.zero_()

            # Embed patch into a full-image canvas via differentiable padding.
            # F.pad operates on the last two dims: (left, right, top, bottom).
            patch_full = F.pad(
                patch,
                (col, W - col - ps, row, H - row - ps),
            ).unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, H, W)

            # Composite in [0, 1] space, re-normalise for model forward pass
            adv_unnorm = images_unnorm.detach() * (1.0 - mask) + patch_full * mask
            adv_unnorm = adv_unnorm.clamp(0.0, 1.0)
            adv_norm = (adv_unnorm - mean_t) / std_t

            logits = self.model(adv_norm)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            with torch.no_grad():
                patch.data.add_(self.lr * patch.grad.sign())
                patch.data.clamp_(0.0, 1.0)

        # Final composite — return re-normalised images to match input range
        with torch.no_grad():
            patch_full = F.pad(
                patch,
                (col, W - col - ps, row, H - row - ps),
            ).unsqueeze(0).expand(B, -1, -1, -1)
            adv_unnorm = images_unnorm * (1.0 - mask) + patch_full * mask
            adv_unnorm = adv_unnorm.clamp(0.0, 1.0)
            adv_images = (adv_unnorm - mean_t) / std_t  # re-normalise

        return adv_images


def build_attack(
    model: torch.nn.Module,
    config: dict,
) -> PatchAttack:
    """
    Build an adversarial patch attack bound to the given model.

    Args:
        model:  A torch.nn.Module in eval mode.
        config: Parsed base.yaml config dict.

    Returns:
        attack: PatchAttack instance ready for inference.
    """
    patch_cfg = config["patch"]
    mean = config["dataset"]["mean"]
    std  = config["dataset"]["std"]
    return PatchAttack(
        model=model,
        patch_size=patch_cfg["patch_size"],
        steps=patch_cfg["steps"],
        lr=patch_cfg["lr"],
        mean=mean,
        std=std,
    )


def run_attack(
    attack: PatchAttack,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Generate adversarial examples for a batch.

    Args:
        attack: A configured PatchAttack instance.
        images: Clean input batch, shape (N, 3, H, W), values in [0, 1].
        labels: Ground-truth class indices, shape (N,).

    Returns:
        adv_images: Adversarial batch, same shape and device as images.
    """
    adv_images = attack(images, labels)
    return adv_images


def print_attack_info(config: dict) -> None:
    """Print a quick summary of the configured attack."""
    patch_cfg = config["patch"]
    ps = patch_cfg["patch_size"]
    img_size = config["models"]["deit_small"]["input_size"]
    coverage = 100.0 * (ps * ps) / (img_size * img_size)
    print(f"[patch] Attack     : Adversarial Patch")
    print(f"        patch_size : {ps}x{ps} pixels")
    print(f"        coverage   : {coverage:.1f}% of {img_size}x{img_size} image")
    print(f"        steps      : {patch_cfg['steps']}")
    print(f"        lr         : {patch_cfg['lr']}")
    print(f"        placement  : random (per batch)")


# Sanity check — run directly to verify attack builds and runs
if __name__ == "__main__":
    from models.loader import load_model

    cfg = load_config()
    print("=== Sanity check: Adversarial Patch on DeiT-S FP32 ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("deit_small", "fp32", cfg, device=device)

    print_attack_info(cfg)

    attack = build_attack(model, cfg)

    dummy_images = torch.rand(2, 3, 224, 224).to(device)
    dummy_labels = torch.zeros(2, dtype=torch.long).to(device)

    adv_images = run_attack(attack, dummy_images, dummy_labels)

    ps = cfg["patch"]["patch_size"]
    print(f"\n[patch] Input  shape : {dummy_images.shape}")
    print(f"[patch] Output shape : {adv_images.shape}")
    print(f"[patch] Max pixel delta : {(adv_images - dummy_images).abs().max():.5f}")
    print(f"[patch] Patch can change pixels by up to 1.0 (unbounded, [0,1] clamp only)")
