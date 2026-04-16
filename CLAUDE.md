# CLAUDE.md — Adversarial Robustness Under Compression for Edge ViTs

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

- **Model: DeiT-S only.** DeiT-B excluded to stay within 100 CU Colab Pro budget.
  DeiT-B is noted as future work in the paper.
- **Compute budget: ~100 CU total.** A100 costs ~3 CU/hr. Every experiment decision
  must account for this. Estimated usage: ~58 CU experiments + ~40 CU buffer for
  reruns and debugging.

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
Apply Attack (FGSM / PGD / Patch / Combined)
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

- **Platform:** Google Colab Pro
- **GPU:** A100 (40GB VRAM) — always select A100 in Runtime > Change runtime type
- **CU rate:** ~3 CU/hr on A100
- **Remaining CU:** track this manually — check via Colab Pro dashboard
- **Session length:** up to 24h, background execution available
- **Code editor:** VS Code with Claude Code — write here, push to GitHub, run on Colab
- **Persistence:** Mount Google Drive. Back up every CSV to Drive after each run.

### Colab session startup (run every session)

```python
# Cell 1 — verify A100
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))   # should say A100
print(f"{torch.cuda.mem_get_info()[0]/1e9:.1f} GB free")

# Cell 2 — install dependencies
!pip install -q timm torchattacks bitsandbytes optimum pyyaml

# Cell 3 — mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 4 — pull latest code
!git clone https://github.com/Jmanav/ADVC.git
%cd ADVC

# Cell 5 — restore any previous results from Drive
import shutil, os
drive_dir = '/content/drive/MyDrive/research'
os.makedirs('results', exist_ok=True)
for f in ['phase1_results.csv', 'phase2_at_results.csv',
          'phase2_atkd_results.csv', 'phase3_results.csv']:
    src = f'{drive_dir}/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'results/{f}')
        print(f'Restored {f}')

# Cell 6 — copy dataset to local NVMe (REQUIRED — do not skip)
# Reading images directly from Drive is ~100x slower than local disk.
# Scripts use paths relative to the project root, so copying here makes
# data/imagenet/train and data/imagenet/val resolve to local NVMe automatically.
import shutil, os
drive_data = '/content/drive/MyDrive/research/data/imagenet'
local_data = '/content/ADVC/data/imagenet'
if not os.path.exists(local_data):
    print("Copying dataset from Drive to local NVMe — this takes ~2 min...")
    shutil.copytree(drive_data, local_data)
    print("Done. Data is now on local disk.")
else:
    print("Dataset already on local disk — skipping copy.")
```

### After every run — back up immediately

```python
import shutil, os
drive_dir = '/content/drive/MyDrive/research'
os.makedirs(drive_dir, exist_ok=True)
for f in ['phase1_results.csv', 'phase2_at_results.csv',
          'phase2_atkd_results.csv', 'phase3_results.csv']:
    src = f'results/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'{drive_dir}/{f}')
        print(f'Backed up {f}')
```

---

## Model

| Model | timm name | HuggingFace ID | Params |
|-------|-----------|---------------|--------|
| DeiT-S | `deit_small_patch16_224` | `facebook/deit-small-patch16-224` | 22M |

- Load via `timm` for all cases
- For AT+KD: load FP32 teacher and compressed student simultaneously
  (~85MB + ~25MB = ~110MB total — trivially fits on A100 40GB)
- Always `model.eval()` before inference, `model.train()` before fine-tuning
- Teacher in AT+KD must always stay frozen: `teacher.eval()` + `torch.no_grad()`

---

## Compression levels

| Level | Method | Tool | Notes |
|-------|--------|------|-------|
| FP32 | None — baseline | — | Always run first, establishes upper bound |
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

### Combined attack (Phase 3)

FGSM + PGD + Patch applied sequentially to the same input.
Implemented in `attacks/combined.py`.

**Fixed across all experiments:**
- Epsilon: 8/255 L-inf for FGSM and PGD
- Patch: 32×32 pixels on 224×224 image
- Validation subset: 5000 images, ImageNet-1k val, seed=42
- Training subset: 10000 images, ImageNet-1k train, seed=42
- Batch size: 64 (A100 can handle this comfortably for DeiT-S)

---

## Defenses in scope

### Defense 1 — Adversarial Training (AT)

Fine-tune the already-compressed model on FGSM adversarial inputs.

AT parameters:
- Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=1e-4
- Epochs: 7
- AT epsilon: 8/255 — must always match attack epsilon
- Checkpoint: save after every epoch to `results/checkpoints/at/` AND Drive

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
- Checkpoint: save after every epoch to `results/checkpoints/atkd/` AND Drive

---

## Experiment matrix — DeiT-S only

### Phase 1 — No defense (baseline) — ~4 CU

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| FP32 | ○ | ○ | ○ |
| INT8 | ○ | ○ | ○ |
| INT4 | ○ | ○ | ○ |

9 rows → `results/phase1_results.csv`

### Phase 2a — AT defense — ~18 CU

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| FP32 + AT | ○ | ○ | ○ |
| INT8 + AT | ○ | ○ | ○ |
| INT4 + AT | ○ | ○ | ○ |

9 rows → `results/phase2_at_results.csv`

### Phase 2b — AT+KD defense — ~22 CU

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| FP32 + AT+KD | ○ | ○ | ○ |
| INT8 + AT+KD | ○ | ○ | ○ |
| INT4 + AT+KD | ○ | ○ | ○ |

9 rows → `results/phase2_atkd_results.csv`

### Phase 3 — Combined attack vs all defenses — ~9 CU

| Compression | No defense | AT | AT+KD |
|-------------|-----------|-----|-------|
| FP32 | ○ | ○ | ○ |
| INT8 | ○ | ○ | ○ |
| INT4 | ○ | ○ | ○ |

9 rows → `results/phase3_results.csv`

**Total: 36 rows, ~53 CU estimated, ~47 CU buffer for reruns and debugging.**

---

## Metrics — every CSV row must contain all fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | `deit_small` |
| `compression` | str | `fp32`, `int8`, or `int4` |
| `defense` | str | `none`, `at`, or `at_kd` |
| `attack` | str | `fgsm`, `pgd`, `patch`, or `combined` |
| `clean_acc` | float | Accuracy on clean inputs (0–1) |
| `robust_acc` | float | Accuracy under attack (0–1) |
| `asr` | float | Attack success rate = 1 − robust_acc |
| `robustness_gap` | float | clean_acc − robust_acc |
| `phase` | int | 1, 2, or 3 |

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
│   ├── pgd.py                             ← build next
│   ├── patch.py                           ← after pgd
│   └── combined.py                        ← last (Phase 3)
├── defenses/
│   ├── adversarial_training.py            ← AT on compressed model
│   └── at_kd.py                           ← AT+KD, frozen FP32 teacher
├── experiments/
│   ├── eval_phase1.py                     ← no defense, 3 attacks
│   ├── eval_phase2_at.py                  ← AT defense, 3 attacks
│   ├── eval_phase2_atkd.py                ← AT+KD defense, 3 attacks
│   └── eval_phase3.py                     ← combined attack, all defenses
├── results/
│   ├── phase1_results.csv
│   ├── phase2_at_results.csv
│   ├── phase2_atkd_results.csv
│   ├── phase3_results.csv
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

## Build order for remaining files

```
1. attacks/pgd.py                   same structure as fgsm.py
2. attacks/patch.py                 same calling convention
3. defenses/adversarial_training.py AT on compressed model, checkpoint every epoch
4. defenses/at_kd.py                frozen FP32 teacher + compressed student
5. experiments/eval_phase1.py       no defense — phase1_results.csv
6. experiments/eval_phase2_at.py    AT defense — phase2_at_results.csv
7. experiments/eval_phase2_atkd.py  AT+KD defense — phase2_atkd_results.csv
8. attacks/combined.py              chains fgsm → pgd → patch
9. experiments/eval_phase3.py       combined attack — phase3_results.csv
10. notebooks/01_results_viz.ipynb  paper plots
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
- **tqdm on all loops** — progress visible in Colab output
- **Type hints** on all function signatures
- **Docstrings** on every function
- **Append to CSV immediately** after each row — never accumulate in memory
- **A100 batch size** — use 64 for eval, 32 for AT/AT+KD training

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
- Do not run more than 7 AT/AT+KD epochs — CU budget
- Do not apply gradients to the teacher in AT+KD — frozen always
- Do not use `plt.show()` in scripts — save to `results/figures/`
- Do not commit weights, checkpoints, or CSVs to git
- Do not hardcode any path or hyperparameter
- Do not switch to V100 or T4 — always use A100 on Colab Pro

---

## CU budget tracker

Update this manually after each experiment run:

| Phase | Estimated CU | Actual CU | Status |
|-------|-------------|-----------|--------|
| Phase 1 | ~4 CU | — | pending |
| Phase 2a (AT) | ~18 CU | — | pending |
| Phase 2b (AT+KD) | ~22 CU | — | pending |
| Phase 3 | ~9 CU | — | pending |
| Buffer | ~47 CU | — | reserved |
| **Total** | **~100 CU** | | |

---

## Expected findings to watch for

**Phase 1:**
- ASR increases FP32 → INT8 → INT4 — compression makes models more vulnerable
- PGD ASR > FGSM ASR at every level — PGD is strictly stronger
- If this pattern does not hold, check the attack implementation

**Phase 2:**
- Both AT and AT+KD reduce ASR vs Phase 1
- AT+KD should outperform AT — richer teacher supervision
- The AT vs AT+KD gap should narrow at INT4 — core finding if confirmed
- If gap does NOT narrow: also significant, means KD helps equally regardless of compression

**Phase 3:**
- Combined ASR > any individual attack
- AT+KD should still outperform AT under combined attack
- If AT+KD collapses at INT4 under combined attack — headline result

**Sanity checks — flag immediately if any of these occur:**
- clean_acc drops more than 5% FP32 → INT4 → quantization setup wrong
- AT+KD performs worse than AT → teacher not frozen or temperature wrong
- ASR = 0.0 or 1.0 for any cell → implementation bug, do not include in paper
- A100 not available → do not run on T4, wait for A100 to preserve CU budget
