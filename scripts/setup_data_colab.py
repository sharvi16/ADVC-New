"""
scripts/setup_data_colab.py

Downloads exactly 1000 ImageNet-1k validation images and saves them in
torchvision ImageFolder format under data/imagenet/val/.

Run once in Colab before any experiment:
    !python scripts/setup_data_colab.py --method a   # HuggingFace (recommended)
    !python scripts/setup_data_colab.py --method b   # Kaggle

Both produce:
    data/imagenet/val/
    ├── n01440764/
    │   ├── img_00000.JPEG
    │   └── ...
    ├── n01443537/
    └── ...
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_IMAGES = 1000
VAL_DIR = "data/imagenet/val"

# ─────────────────────────────────────────────────────────────────────────────
# METHOD A — HuggingFace datasets streaming (recommended)
#
# Streams only the images you need — no multi-GB download.
# Requires:
#   1. A HuggingFace account
#   2. Accept terms at: https://huggingface.co/datasets/imagenet-1k
#   3. Your HF token (from https://huggingface.co/settings/tokens)
# ─────────────────────────────────────────────────────────────────────────────

def setup_method_a():
    # ── Install deps if missing ───────────────────────────────────────────────
    try:
        from huggingface_hub import login
    except ImportError:
        print("[setup] Installing huggingface_hub …")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import login

    try:
        from datasets import load_dataset
    except ImportError:
        print("[setup] Installing datasets …")
        os.system("pip install -q datasets pillow")
        from datasets import load_dataset

    # ── Login ─────────────────────────────────────────────────────────────────
    # Priority: env var HF_TOKEN → Colab secret HF_TOKEN → interactive prompt
    token = os.environ.get("HF_TOKEN", None)

    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            print("[setup] Using HF_TOKEN from Colab secrets.")
        except Exception:
            pass

    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("[setup] No HF_TOKEN found — prompting for login.")
        print("[setup] Get your token at: https://huggingface.co/settings/tokens")
        login()  # interactive prompt

    # ── Stream dataset ────────────────────────────────────────────────────────
    print(f"\n[setup] Streaming {N_IMAGES} images from imagenet-1k validation split …")

    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    dataset = dataset.shuffle(seed=SEED, buffer_size=5000)

    # Build label index → synset name mapping from dataset features
    label_to_synset = {}

    saved = 0
    for example in dataset:
        if saved >= N_IMAGES:
            break

        # Build mapping on first example
        if not label_to_synset:
            try:
                names = dataset.features["label"].names
                label_to_synset = {i: name for i, name in enumerate(names)}
            except Exception:
                label_to_synset = {i: f"class_{i:04d}" for i in range(1000)}

        label_id: int = example["label"]
        image = example["image"]
        synset = label_to_synset.get(label_id, f"class_{label_id:04d}")

        class_dir = Path(VAL_DIR) / synset
        class_dir.mkdir(parents=True, exist_ok=True)

        img_path = class_dir / f"img_{saved:05d}.JPEG"
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(img_path, "JPEG")

        saved += 1
        if saved % 100 == 0:
            print(f"[setup]   {saved}/{N_IMAGES} saved …")

    print(f"[setup] Done — {saved} images written to {VAL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD B — Kaggle API
#
# Downloads the full validation tar (~6 GB) then keeps N_IMAGES images.
# Requires:
#   1. kaggle.json (from kaggle.com → Settings → API → Create New Token)
#   2. Accept competition terms at:
#      https://www.kaggle.com/competitions/imagenet-object-localization-challenge
# ─────────────────────────────────────────────────────────────────────────────

def setup_method_b():
    # ── Place kaggle.json ─────────────────────────────────────────────────────
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        print("[setup] kaggle.json not found — opening Colab file picker …")
        try:
            from google.colab import files
            uploaded = files.upload()
            kaggle_cfg.parent.mkdir(parents=True, exist_ok=True)
            for fname, data in uploaded.items():
                kaggle_cfg.write_bytes(data)
                print(f"[setup] Saved → {kaggle_cfg}")
        except ImportError:
            sys.exit(
                "[setup] Not in Colab and ~/.kaggle/kaggle.json missing.\n"
                "        Place your kaggle.json there and re-run."
            )
    kaggle_cfg.chmod(0o600)

    # ── Install kaggle CLI ────────────────────────────────────────────────────
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[setup] Installing kaggle …")
        os.system("pip install -q kaggle")

    # ── Download val tar ──────────────────────────────────────────────────────
    print("[setup] Downloading ImageNet val split from Kaggle (~6 GB) …")
    os.makedirs("data/imagenet", exist_ok=True)
    os.system(
        "kaggle competitions download "
        "imagenet-object-localization-challenge "
        "-f ILSVRC/Data/CLS-LOC/val.tar "
        "-p data/imagenet"
    )

    # ── Extract flat ──────────────────────────────────────────────────────────
    flat_dir = "data/imagenet/val_flat"
    os.makedirs(flat_dir, exist_ok=True)
    print("[setup] Extracting …")
    os.system(f"tar -xf data/imagenet/val.tar -C {flat_dir} --strip-components=5")

    # ── Download synset label mapping ─────────────────────────────────────────
    print("[setup] Downloading label mapping …")
    os.system(
        "wget -q https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/slim/datasets/imagenet_2012_validation_synset_labels.txt "
        "-O data/imagenet/val_labels.txt"
    )

    label_file = Path("data/imagenet/val_labels.txt")
    if not label_file.exists():
        sys.exit("[setup] Could not download label mapping.")

    synset_per_image = label_file.read_text().strip().splitlines()
    flat_images = sorted(Path(flat_dir).glob("*.JPEG"))
    print(f"[setup] Found {len(flat_images)} extracted images.")

    random.seed(SEED)
    selected = random.sample(range(len(flat_images)), min(N_IMAGES, len(flat_images)))

    print(f"[setup] Moving {N_IMAGES} images into ImageFolder structure …")
    out_dir = Path(VAL_DIR)
    for idx in selected:
        img_path = flat_images[idx]
        synset = synset_per_image[idx]
        dest_dir = out_dir / synset
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, dest_dir / img_path.name)

    print(f"[setup] Done — {N_IMAGES} images in {VAL_DIR}")
    print("[setup] Cleaning up temp files …")
    shutil.rmtree(flat_dir, ignore_errors=True)


# ── Verification ──────────────────────────────────────────────────────────────

def verify():
    try:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(VAL_DIR)
        print(f"\n[verify] ImageFolder loaded OK")
        print(f"[verify]   images  : {len(ds)}")
        print(f"[verify]   classes : {len(ds.classes)}")
        if len(ds) < 100:
            print("[verify] WARNING: fewer than 100 images — check the setup.")
    except Exception as e:
        print(f"[verify] FAILED: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["a", "b"],
        default="a",
        help="a = HuggingFace streaming (default)  |  b = Kaggle download",
    )
    args = parser.parse_args()

    if Path(VAL_DIR).exists() and any(Path(VAL_DIR).iterdir()):
        print(f"[setup] {VAL_DIR} already exists and is non-empty — skipping download.")
    else:
        if args.method == "a":
            setup_method_a()
        else:
            setup_method_b()

    verify()
