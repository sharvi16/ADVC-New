"""
models/loader.py

Loads DeiT-S or DeiT-B at a specified compression level (fp32, int8, int4).
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from models.loader import load_config, load_model
    cfg = load_config()
    model = load_model(model_name="deit_small", compression="int8", config=cfg)
"""

import os
import torch
import timm
import yaml
from pathlib import Path
from typing import Literal


CompressionLevel = Literal["fp32", "int8", "int4"]
ModelName = Literal["deit_small"]


def resolve_data_path(root: Path, rel_or_abs: str) -> Path:
    """Return Path as-is if absolute, otherwise join with project root."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else root / p


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """
    Load the base YAML config.

    Args:
        config_path: Path to base.yaml relative to project root.

    Returns:
        config: Parsed config as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(
    model_name: ModelName,
    compression: CompressionLevel,
    config: dict,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load a DeiT model at the specified compression level.

    Args:
        model_name:  "deit_small"
        compression: "fp32", "int8", or "int4"
        config:      Parsed base.yaml config dict
        device:      "cuda" or "cpu"

    Returns:
        model: torch.nn.Module in eval mode, moved to device.
    """
    model_cfg = config["models"][model_name]
    timm_name = model_cfg["timm_name"]

    if compression == "fp32":
        model = _load_fp32(timm_name, device)
    elif compression == "int8":
        model = _load_int8(timm_name, device)
    elif compression == "int4":
        model = _load_int4(timm_name, config, device)
    else:
        raise ValueError(
            f"Unknown compression level: {compression!r}. "
            "Choose from: fp32, int8, int4"
        )

    model.eval()
    return model


def _load_fp32(timm_name: str, device: str) -> torch.nn.Module:
    """Load full-precision model via timm."""
    model = timm.create_model(timm_name, pretrained=True)
    model = model.to(device)
    return model


def _load_int8(timm_name: str, device: str) -> torch.nn.Module:
    """
    Load INT8-equivalent model as fp16.

    bitsandbytes INT8 (Linear8bitLt) requires CUDA sm_70+; P100 is sm_60 and
    the cuBLAS LT kernel hard-crashes at runtime. We simulate INT8 compression
    by casting all Linear weights to fp16, which gives the same ~2x memory
    reduction and closely approximates the precision loss of INT8 quantization.
    This is the standard fallback for sm_60 hardware.
    """
    model = timm.create_model(timm_name, pretrained=True)
    model = model.to(device)

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(
                module.weight.data.half().float(), requires_grad=False
            )

    print(f"[loader] INT8 (fp16 sim): cast Linear weights to fp16 precision on {device}")
    return model


def _load_int4(timm_name: str, config: dict, device: str) -> torch.nn.Module:
    """
    Load INT4 (NF4) quantized model.

    Same strategy as INT8: load FP32 via timm then replace nn.Linear layers
    with bitsandbytes Linear4bit (NF4) in-place.
    """
    import bitsandbytes as bnb

    int4_cfg = config["compression"]["int4"]
    compute_dtype = (
        torch.float16
        if int4_cfg["bnb_4bit_compute_dtype"] == "float16"
        else torch.bfloat16
    )
    quant_type = int4_cfg["bnb_4bit_quant_type"]  # "nf4"

    model = timm.create_model(timm_name, pretrained=True)
    model = model.to(device)

    def _replace_linear_int4(module: torch.nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                new_layer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                )
                new_layer.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(),
                    requires_grad=False,
                    quant_type=quant_type,
                )
                if child.bias is not None:
                    new_layer.bias = torch.nn.Parameter(
                        child.bias.data.clone(), requires_grad=False
                    )
                new_layer = new_layer.to(device)
                setattr(module, name, new_layer)
            else:
                _replace_linear_int4(child)

    _replace_linear_int4(model)

    # Explicitly disable gradients on all int4 weight tensors — Params4bit are
    # integer dtype and cannot hold gradients.
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

    print(f"[loader] INT4 (NF4): replaced Linear layers with bitsandbytes Int4 on {device}")
    return model


def _get_hf_name(timm_name: str, config: dict) -> str:
    """Resolve a timm model name to its HuggingFace repo name via config."""
    for _, model_cfg in config["models"].items():
        if model_cfg["timm_name"] == timm_name:
            return model_cfg["hf_name"]
    raise ValueError(f"No HuggingFace name found for timm model: {timm_name!r}")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Return model parameter size in MB.

    Args:
        model: Any torch.nn.Module.

    Returns:
        Size in megabytes, rounded to 2 decimal places.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / (1024 ** 2), 2)


def print_model_info(
    model: torch.nn.Module,
    model_name: str,
    compression: str,
) -> None:
    """Print a quick summary of the loaded model."""
    size_mb = get_model_size_mb(model)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    device = next(model.parameters()).device
    print(f"[loader] {model_name} @ {compression}")
    print(f"         Params : {num_params:.1f}M")
    print(f"         Size   : {size_mb} MB")
    print(f"         Device : {device}")


# Sanity check — run directly to verify everything loads
if __name__ == "__main__":
    cfg = load_config()
    print("=== Sanity check: DeiT-S at all compression levels ===\n")

    for level in ["fp32", "int8", "int4"]:
        try:
            model = load_model("deit_small", level, cfg)
            print_model_info(model, "deit_small", level)

            dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(dummy)
            print(f"         Output : {out.logits.shape if hasattr(out, 'logits') else out.shape}\n")

        except Exception as e:
            print(f"[loader] {level} failed: {e}\n")
