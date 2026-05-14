"""
models/loader.py

Loads DeiT-S or DeiT-B at a specified compression level (fp32, int8, int4).
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from models.loader import load_config, load_model
    cfg = load_config()
    model = load_model(model_name="deit_small", compression="int8", config=cfg)
"""

import torch
import timm
import yaml
from typing import Literal


CompressionLevel = Literal["fp32", "int8", "int4"]
ModelName = Literal["deit_small"]


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
        model = _load_int8(timm_name, config, device)
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


def _load_int8(timm_name: str, config: dict, device: str) -> torch.nn.Module:
    """
    Load INT8 quantized model.
    Uses bitsandbytes if available, falls back to torch static quantization.
    """
    backend = config["compression"]["int8"]["backend"]

    if backend == "bitsandbytes":
        try:
            from transformers import AutoModelForImageClassification, BitsAndBytesConfig
            import bitsandbytes  # noqa: F401

            hf_name = _get_hf_name(timm_name, config)

            # load_in_8bit as a direct kwarg was removed in newer transformers;
            # it must be passed via BitsAndBytesConfig instead.
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            import os
            model = AutoModelForImageClassification.from_pretrained(
                hf_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
            )
            return model

        except (ImportError, Exception) as exc:
            print(
                f"[loader] bitsandbytes INT8 failed ({exc}), "
                "falling back to torch static quantization."
            )
            backend = "torch"

    if backend == "torch":
        model = timm.create_model(timm_name, pretrained=True)
        model = model.to("cpu")
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        if device == "cuda":
            print("[loader] Note: torch static quantization runs on CPU only.")
        return model

    raise ValueError(f"Unknown INT8 backend: {backend!r}")


def _load_int4(timm_name: str, config: dict, device: str) -> torch.nn.Module:
    """Load INT4 (NF4) quantized model via bitsandbytes."""
    from transformers import AutoModelForImageClassification, BitsAndBytesConfig

    int4_cfg = config["compression"]["int4"]
    compute_dtype = (
        torch.float16
        if int4_cfg["bnb_4bit_compute_dtype"] == "float16"
        else torch.bfloat16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=int4_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    hf_name = _get_hf_name(timm_name, config)

    import os
    model = AutoModelForImageClassification.from_pretrained(
        hf_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
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

            dummy = torch.randn(1, 3, 224, 224)
            if level == "fp32":
                dummy = dummy.cuda()
            with torch.no_grad():
                out = model(dummy)
            print(f"         Output : {out.logits.shape if hasattr(out, 'logits') else out.shape}\n")

        except Exception as e:
            print(f"[loader] {level} failed: {e}\n")
