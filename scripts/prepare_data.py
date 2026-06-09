"""Convert DatasetNinja GlaS release into PNG images + uint16 instance masks.

Also generates stratified 2-fold CV and fixed 70/15 train/val splits.

Usage:
    uv run python scripts/prepare_data.py --dataset ../Dataset --out data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from dcan.data.ninja_parse import get_image_tags, parse_annotation


def process_split(src_dir: Path, out_dir: Path):
    img_dir = src_dir / "img"
    ann_dir = src_dir / "ann"
    out_img = out_dir / "images"
    out_inst = out_dir / "instances"
    out_img.mkdir(parents=True, exist_ok=True)
    out_inst.mkdir(parents=True, exist_ok=True)

    bmp_files = sorted(img_dir.glob("*.bmp"))
    print(f"  Processing {len(bmp_files)} images from {src_dir.name}")

    meta = {}
    for bmp_path in bmp_files:
        name = bmp_path.stem
        ann_path = ann_dir / f"{bmp_path.name}.json"

        # Read and save image as PNG
        image = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out_img / f"{name}.png"), image)

        # Parse annotation
        (h, w), instance_mask = parse_annotation(ann_path)
        cv2.imwrite(str(out_inst / f"{name}.png"), instance_mask.astype(np.uint16))

        # Collect tags
        tags = get_image_tags(ann_path)
        n_instances = len(np.unique(instance_mask)) - 1  # exclude 0
        is_benign = "benign" in tags
        meta[name] = {
            "n_instances": n_instances,
            "benign": is_benign,
            "tags": {k: str(v) for k, v in tags.items()},
        }
        print(f"    {name}: {h}x{w}, {n_instances} glands, {'benign' if is_benign else 'malignant'}")

    return meta


def generate_splits(meta: dict, out_dir: Path, seed: int = 42):
    """Generate 2-fold CV and 70/15 fixed split from training set metadata."""
    names = sorted(meta.keys())
    benign = [n for n in names if meta[n]["benign"]]
    malignant = [n for n in names if not meta[n]["benign"]]

    rng = random.Random(seed)
    rng.shuffle(benign)
    rng.shuffle(malignant)

    # 2-fold CV: alternate assignment
    fold = {}
    for i, n in enumerate(benign):
        fold[n] = i % 2
    for i, n in enumerate(malignant):
        fold[n] = i % 2

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for f in range(2):
        fold_val = [n for n in names if fold[n] == f]
        fold_train = [n for n in names if fold[n] != f]
        (splits_dir / f"fold{f}_train.txt").write_text("\n".join(sorted(fold_train)))
        (splits_dir / f"fold{f}_val.txt").write_text("\n".join(sorted(fold_val)))

    # Fixed 70/15 split
    n_val_b = max(1, len(benign) * 15 // 85)
    n_val_m = max(1, len(malignant) * 15 // 85)
    val_names = benign[:n_val_b] + malignant[:n_val_m]
    train_names = [n for n in names if n not in val_names]
    (splits_dir / "fixed_train.txt").write_text("\n".join(sorted(train_names)))
    (splits_dir / "fixed_val.txt").write_text("\n".join(sorted(val_names)))

    # All-85 split for the paper-benchmark configs (train on all training images).
    (splits_dir / "all_train.txt").write_text("\n".join(names))

    print(f"\n  Splits saved to {splits_dir}")
    print(f"    2-fold CV: fold0={sum(1 for v in fold.values() if v==0)}, fold1={sum(1 for v in fold.values() if v==1)}")
    print(f"    Fixed: train={len(train_names)}, val={len(val_names)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GlaS dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Dataset/ dir")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)

    for split in ["train", "test_a", "test_b"]:
        src = dataset / split
        if src.exists():
            print(f"\n[{split}]")
            meta = process_split(src, out / split)
            if split == "train":
                generate_splits(meta, out, seed=args.seed)

    # Save meta
    meta_path = out / "meta.json"
    print(f"\nDone. Output at {out}")


if __name__ == "__main__":
    main()
