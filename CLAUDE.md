# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick reference for Claude Code

### Running experiments (via JupyterHub terminal on the H100 lab server)

```bash
# Phase 1 — baseline, no defense
python experiments/eval_phase1.py --model deit_small

# Phase 2a — AT defense (trains then evaluates; use --skip-training to load checkpoint)
python experiments/eval_phase2_at.py --model deit_small [--skip-training] [--compression int8|int4]

# Phase 2b — AT+KD defense
python experiments/eval_phase2_atkd.py --model deit_small [--skip-training] [--compression int8|int4]
```

### Attack interface (every attack must match this signature)

```python
attack(model, images, labels) -> perturbed_images   # Tensor, same shape as images
```

`build_attack(model, config)` returns the callable; `run_attack(attack, images, labels)` is the uniform wrapper.

### Adding a new attack or experiment

1. Read `attacks/fgsm.py` as the canonical template — same structure for every attack.
2. Pull all params from `configs/base.yaml` — never hardcode epsilon, steps, patch size, etc.
3. Include a `if __name__ == "__main__":` sanity-check block (dummy forward pass).
4. For experiment scripts, copy the resumability pattern from `experiments/eval_phase1.py:load_completed_runs()`.

### Key invariants the code enforces

- **Compress before defend** — model is always quantized before AT or AT+KD fine-tuning.
- **Teacher always frozen** — `teacher.eval()` + `torch.no_grad()` on every teacher forward in `defenses/at_kd.py`.
- **INT4 checkpoints save the full model** (not state_dict) because `bitsandbytes` NF4 weights can't be loaded with `load_state_dict` alone — see `save_checkpoint()` in both defense files.
- **LogitsWrapper** — HuggingFace model outputs are dataclasses; both defense files and phase1 wrap models with `_LogitsWrapper` so downstream code receives plain tensors.
- **Per-compression learning rates** in `configs/base.yaml` (`at.lr_per_compression`) — INT4 gets 10× lower LR than FP32 to avoid destroying fragile quantized weights.

### Architecture in one paragraph

`models/loader.py` loads DeiT-S at fp32/int8/int4 via timm + bitsandbytes and returns an eval-mode model. `attacks/` wraps torchattacks (FGSM, PGD) or provides a custom class (PatchAttack); each file is standalone with its own config loader. `defenses/` fine-tunes the already-compressed model: AT with FGSM adversarial inputs only, AT+KD adding a frozen FP32 teacher KL loss. `experiments/` scripts wire the pieces together in loops over compression levels and attacks, appending to CSV immediately after each row via `utils/metrics.py:save_results_to_csv()`. All hyperparameters live exclusively in `configs/base.yaml`.

### Files not yet built

```
notebooks/01_results_viz.ipynb   # paper plots from phase1/2a/2b CSVs
```

---

# Research spec — Adversarial Robustness Under Compression for Edge ViTs

## Project overview

Research project studying whether adversarial defenses remain effective after model
compression (quantization) on Vision Transformers (ViTs) deployed at the edge.

We evaluate two defenses — Adversarial Training (AT) and Adversarial Training with
Knowledge Distillation (AT+KD) — applied to a compressed DeiT-S model, then attacked
individually and in combination across three compression levels.

### Core research questions

> 1. After compressing DeiT-S (INT8, INT4), does AT still recover meaningful robustness
>    — and does recovery degrade as compression becomes more aggressive?
> 2. Does AT+KD recover MORE robustness than AT alone on a compressed model — and does
>    this advantage shrink at INT4 where quantization is most aggressive?

### Scope decision

- **Model: DeiT-S only.** DeiT-B excluded due to VRAM constraints on the H100 slice.
  DeiT-B is noted as future work in the paper.

---

## Core pipeline (always in this order — never deviate)

```
Pretrained DeiT-S (FP32)
        ↓
    Compress                         ← PTQ to INT8 or INT4 via bitsandbytes
        ↓
Compressed DeiT-S                    ← this is the edge-deployed model
        ↓
Apply Defense (AT or AT+KD)
        ↓  AT:    fine-tune compressed model on FGSM adversarial inputs
        ↓  AT+KD: fine-tune using FGSM adversarial inputs
        ↓          + FP32 teacher soft labels via KL divergence
        ↓
Apply Attack (FGSM / PGD / Patch)
        ↓
Measure and save results
```

### Why these two defenses and not others

- **AT** — directly retrains compressed weights. The interaction with quantization
  precision is real and measurable. Included.
- **AT+KD** — adds a FP32 teacher to guide compressed student recovery. Teacher
  soft labels provide richer supervision signal than hard labels alone, potentially
  recovering vision representations that PTQ destroyed. Included.

---

## Hardware & environment

- **Platform:** University AI lab — JupyterHub portal
- **GPU:** NVIDIA H100 (8.6 GB VRAM slice)
- **CPU:** 1 core Intel Xeon Platinum 8480+
- **RAM:** 28 GB
- **Access:** JupyterHub web portal — open a terminal or notebook from the launcher
- **Persistence:** Files on the server persist between sessions, but the dataset must be copied to the working directory each session (see below)
- **Code editor:** VS Code with Claude Code locally — push to GitHub, pull on server

### Session startup (run in a JupyterHub terminal or notebook cell)

> **HF_TOKEN required before any model load.**
> `facebook/deit-small-patch16-224` requires a HuggingFace token for INT8/INT4
> loading via `AutoModelForImageClassification`. Set it as an environment variable
> before running any script. Without this, `from_pretrained` will fail with a 401.

```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'  # or set in ~/.bashrc on the server

# Verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))   # should say H100
print(f"{torch.cuda.mem_get_info()[0]/1e9:.1f} GB free")
```

```bash
# Pull latest code
git clone https://github.com/Jmanav/ADVC.git
cd ADVC
pip install -r requirements.txt
```

### Dataset — copy to project directory each session

The dataset must be copied to `data/imagenet/` inside the project root each session.
Scripts use paths relative to the project root (`data/imagenet/val`, `data/imagenet/train`).

```python
import shutil, os
# Update src_dir to wherever the dataset lives on the server
src_dir = '/path/to/lab/storage/imagenet'
dst_dir = 'data/imagenet'
if not os.path.exists(dst_dir):
    print("Copying dataset to project directory...")
    shutil.copytree(src_dir, dst_dir)
    print("Done.")
else:
    print("Dataset already present — skipping copy.")
```

### After every run — back up results

Results CSVs are in `results/` and persist on the server between sessions. Back up
to a safe location (personal storage, GitHub, etc.) after each completed phase.

```bash
cp results/phase1_results.csv /path/to/your/backup/
cp results/phase2_at_results.csv /path/to/your/backup/
cp results/phase2_atkd_results.csv /path/to/your/backup/
```

---

## Model

| Model | timm name | HuggingFace ID | Params |
|-------|-----------|---------------|--------|
| DeiT-S | `deit_small_patch16_224` | `facebook/deit-small-patch16-224` | 22M |

- Load via `timm` for all cases
- For AT+KD: load FP32 teacher and compressed student simultaneously
  (~85MB + ~25MB = ~110MB total — fits within the 8.6 GB H100 VRAM slice)
- Always `model.eval()` before inference, `model.train()` before fine-tuning
- Teacher in AT+KD must always stay frozen: `teacher.eval()` + `torch.no_grad()`

---

## Compression levels

| Level | Method | Tool | Notes |
|-------|--------|------|-------|
| INT8 | Post-training quantization | `bitsandbytes` | `load_in_8bit=True` |
| INT4 | Post-training quantization NF4 | `bitsandbytes` | `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"` |

All compression is post-training (PTQ). Compression always before defense.

---

## Attacks in scope

### Individual attacks (Phase 1 and Phase 2)

| Attack | File | Params | Notes |
|--------|------|--------|-------|
| FGSM | `attacks/fgsm.py` | eps=8/255 | ✅ done |
| PGD | `attacks/pgd.py` | eps=8/255, alpha=2/255, steps=20 | Standard in robustness literature |
| Adversarial Patch | `attacks/patch.py` | patch_size=32, steps=150 | Physically realizable |

**Fixed across all experiments:**
- Epsilon: 8/255 L-inf for FGSM and PGD
- Patch: 32×32 pixels on 224×224 image
- Validation subset: 5000 images, ImageNet-1k val, seed=42
- Training subset: 10000 images, ImageNet-1k train, seed=42
- Batch size: 32 for eval, 16 for AT/AT+KD training

---

## Defenses in scope

### Defense 1 — Adversarial Training (AT)

Fine-tune the already-compressed model on FGSM adversarial inputs.

AT parameters:
- Optimizer: AdamW, per-compression lr (int8=5e-6, int4=1e-6), weight_decay=0.01
- Epochs: 7, warmup epoch 1 at lr/10
- AT epsilon: 8/255 — must always match attack epsilon
- Checkpoint: save after every epoch to `results/checkpoints/at/`

### Defense 2 — Adversarial Training + Knowledge Distillation (AT+KD)

Fine-tune compressed student using adversarial inputs and FP32 teacher soft labels.

Loss function:
```
adv_images   = FGSM(student, images, labels, eps=8/255)
teacher_soft = softmax(teacher(adv_images) / temperature)   # no gradients
student_soft = softmax(student(adv_images) / temperature)

loss = alpha * CrossEntropy(student(adv_images), true_labels)
     + (1 - alpha) * KLDivergence(student_soft, teacher_soft)
```

AT+KD parameters:
- All AT params above, plus:
- Temperature: 4.0
- Alpha: 0.5
- Teacher: FP32 DeiT-S, frozen throughout — never update teacher weights
- Checkpoint: save after every epoch to `results/checkpoints/atkd/`

---

## Experiment matrix — DeiT-S only

### Phase 1 — No defense (baseline)

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| INT8 | ○ | ○ | ○ |
| INT4 | ○ | ○ | ○ |

6 rows → `results/phase1_results.csv`

### Phase 2a — AT defense

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| INT8 + AT | ○ | ○ | ○ |
| INT4 + AT | ○ | ○ | ○ |

6 rows → `results/phase2_at_results.csv`

### Phase 2b — AT+KD defense

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| INT8 + AT+KD | ○ | ○ | ○ |
| INT4 + AT+KD | ○ | ○ | ○ |

6 rows → `results/phase2_atkd_results.csv`

**Total: 18 rows across all phases.**

---

## Metrics — every CSV row must contain all fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | `deit_small` |
| `compression` | str | `int8` or `int4` |
| `defense` | str | `none`, `at`, or `at_kd` |
| `attack` | str | `fgsm`, `pgd`, or `patch` |
| `clean_acc` | float | Accuracy on clean inputs (0–1) |
| `robust_acc` | float | Accuracy under attack (0–1) |
| `asr` | float | Attack success rate = 1 − robust_acc |
| `robustness_gap` | float | clean_acc − robust_acc |
| `phase` | int | 1 or 2 |

---

## Project structure

```
ADVC/
├── CLAUDE.md                              # this file — read before every task
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml                          # all hyperparameters — never hardcode
├── models/
│   └── loader.py                          ✅ done
├── attacks/
│   ├── fgsm.py                            ✅ done
│   ├── pgd.py                             ✅ done
│   └── patch.py                           ✅ done
├── defenses/
│   ├── adversarial_training.py            ✅ done
│   └── at_kd.py                           ✅ done
├── experiments/
│   ├── eval_phase1.py                     ✅ done — no defense, 3 attacks
│   ├── eval_phase2_at.py                  ✅ done — AT defense, 3 attacks
│   └── eval_phase2_atkd.py                ✅ done — AT+KD defense, 3 attacks
├── results/
│   ├── phase1_results.csv
│   ├── phase2_at_results.csv
│   ├── phase2_atkd_results.csv
│   ├── checkpoints/
│   │   ├── at/                            ← epoch checkpoints for AT
│   │   └── atkd/                          ← epoch checkpoints for AT+KD
│   └── figures/                           ← all paper plots saved here
├── notebooks/
│   ├── 00_setup_check.ipynb
│   └── 01_results_viz.ipynb
└── utils/
    └── metrics.py                         ✅ done
```

---

## Remaining work

```
1. notebooks/01_results_viz.ipynb   paper plots from phase1/2a/2b CSVs
```

---

## Coding conventions

- **Read CLAUDE.md first** before writing any file
- **No hardcoded values** — all params from `configs/base.yaml`
- **Attack interface** — every attack callable as
  `attack(model, images, labels) → perturbed_images`
- **Compress before defend** — never apply AT or AT+KD before compression
- **Teacher always frozen** in AT+KD — `teacher.eval()` and `torch.no_grad()`
  on every teacher forward pass, no exceptions
- **Seeds everywhere** — `torch.manual_seed(42)`, `random.seed(42)`,
  `np.random.seed(42)` at top of every script
- **tqdm on all loops** — progress visible in output
- **Type hints** on all function signatures
- **Docstrings** on every function
- **Append to CSV immediately** after each row — never accumulate in memory
- **Batch size** — use 32 for eval, 16 for AT/AT+KD training

---

## Resumability pattern (mandatory in every experiment script)

```python
import os, csv

RESULTS_FILE = "results/phase1_results.csv"

completed = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            completed.add((row['model'], row['compression'],
                           row['defense'], row['attack']))

for compression in COMPRESSIONS:
    for attack_name in ATTACKS:
        key = ('deit_small', compression, defense_name, attack_name)
        if key in completed:
            print(f"Skipping {key} — already done")
            continue
        result = run_eval(compression, attack_name)
        append_to_csv(RESULTS_FILE, result)
```

---

## What NOT to do

- Do not apply defense before compression — always compress first
- Do not use input preprocessing or randomized smoothing
- Do not load full ImageNet — 5000 val / 10000 train subsets only
- Do not run more than 7 AT/AT+KD epochs
- Do not apply gradients to the teacher in AT+KD — frozen always
- Do not use `plt.show()` in scripts — save to `results/figures/`
- Do not commit weights, checkpoints, or CSVs to git
- Do not hardcode any path or hyperparameter

---

## Run tracker

Update manually after each experiment run:

| Phase | Status |
|-------|--------|
| Phase 1 (baseline, INT8+INT4) | pending |
| Phase 2a (AT, INT8+INT4) | pending |
| Phase 2b (AT+KD, INT8+INT4) | pending |

---

## Expected findings to watch for

**Phase 1:**
- ASR increases INT8 → INT4 — compression makes models more vulnerable
- PGD ASR > FGSM ASR at every level — PGD is strictly stronger
- If this pattern does not hold, check the attack implementation

**Phase 2:**
- Both AT and AT+KD reduce ASR vs Phase 1
- AT+KD should outperform AT — richer teacher supervision
- The AT vs AT+KD gap should narrow at INT4 — core finding if confirmed
- If gap does NOT narrow: also significant, means KD helps equally regardless of compression

**Sanity checks — flag immediately if any of these occur:**
- clean_acc drops more than 5% INT8 → INT4 → quantization setup wrong
- AT+KD performs worse than AT → teacher not frozen or temperature wrong
- ASR = 0.0 or 1.0 for any cell → implementation bug, do not include in paper
