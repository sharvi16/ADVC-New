# CLAUDE.md — Adversarial Robustness Under Compression for Edge ViTs

## Project overview

Research project studying whether adversarial defenses remain effective after model
compression (quantization, pruning) targeting edge deployment. We jointly evaluate
attacks and defenses across multiple compression levels on Vision Transformers
measuring both robustness and inference efficiency.

The core hypothesis: compression preserves clean accuracy but silently degrades
adversarial robustness, making defenses that work on full-precision models unreliable
at edge compression levels.

## Hardware & environment

- **Platform:** Local machine, VS Code with Claude Code extension
- **Python:** [FILL IN — check with `python --version`]
- **Virtual env:** .venv — activate before every session
- **GPU:** [FILL IN — e.g. RTX 3060 8GB / CPU-only]
- **CUDA:** [FILL IN — e.g. 12.1 / N/A]

### Activate environment (always first in terminal)

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Session startup check

```python
import torch
print(torch.cuda.is_available())       # True if GPU available
print(torch.cuda.get_device_name(0))  # GPU name
print(torch.cuda.mem_get_info())      # free / total VRAM in bytes
```

### VS Code settings

`.vscode/settings.json` should contain:
```json
{
  "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
  "python.terminal.activateEnvironment": true
}
```
Use `bin/python` instead of `Scripts/python.exe` on Mac/Linux.

## Models in scope

- **DeiT-S** (`facebook/deit-small-patch16-224`) — 22M params, primary model
- **DeiT-B** (`facebook/deit-base-patch16-224`) — 86M params, secondary model

Load via `timm` (preferred — better feature extraction support).

**Memory guidance:**
- DeiT-S FP32: ~85MB
- DeiT-B FP32: ~330MB
- DeiT-B INT8: ~85MB
- DeiT-B INT4: ~45MB
- Never load both DeiT-S and DeiT-B into memory at the same time

## Compression levels

Three fixed operating points used consistently across all experiments:

| Level | Method                  | Tool                    | Target             |
|-------|-------------------------|-------------------------|--------------------|
| FP32  | None (baseline)         | —                       | Full precision     |
| INT8  | Post-training quant     | `bitsandbytes` / torch  | 2x memory reduction|
| INT4  | Post-training quant     | `bitsandbytes` (NF4)    | 4x memory reduction|

## Attacks in scope

- **FGSM** (`torchattacks.FGSM(model, eps=8/255)`) — single-step, fast, primary attack

Parameters fixed across all experiments:
- Epsilon: 8/255 (L-inf norm)
- Dataset subset: 1000 images from ImageNet-1k validation (seed=42, always same split)

## Defenses in scope

- **Adversarial Training (AT)** — fine-tune the compressed model on FGSM-augmented
  batches for a small number of epochs. Use the same epsilon as the attack (8/255).

AT parameters:
- Optimizer: SGD, lr=0.01, momentum=0.9
- Epochs: 3
- Training data: 5000 images from ImageNet-1k train split (seed=42)
- Save checkpoint after every epoch

Key question AT is answering: does adversarial fine-tuning after compression recover
robustness, and does the recovery cost differ across FP32 vs INT8 vs INT4?

## Project structure

```
project/
├── CLAUDE.md                    # this file
├── README.md
├── requirements.txt
├── .vscode/
│   └── settings.json            # Python interpreter path
├── configs/
│   └── base.yaml                # all hyperparameters live here
├── models/
│   └── loader.py                # load DeiT-S/B at FP32, INT8, INT4
├── attacks/
│   └── fgsm.py                  # FGSM wrapper
├── defenses/
│   └── adversarial_training.py  # AT fine-tuning with checkpointing
├── experiments/
│   ├── eval_clean.py            # baseline clean accuracy
│   ├── eval_robust.py           # robustness under FGSM
│   └── eval_efficiency.py       # latency + memory profiling
├── results/
│   └── .gitkeep
├── notebooks/
│   ├── 00_setup_check.ipynb     # verify environment
│   ├── 01_baseline_eval.ipynb   # exploratory eval
│   └── 02_results_viz.ipynb     # plot results
└── utils/
    ├── metrics.py               # accuracy, ASR, robustness gap
    └── profiler