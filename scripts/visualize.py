"""Qualitative comparison panels: input | GT | FCN | DCAN.

Colorizes instance label maps and composes side-by-side panels that show how the
contour-aware DCAN separates touching glands where the object-only FCN baseline
merges them -- the key visual for the demonstration.

Usage:
    uv run python scripts/visualize.py                 # auto-picks busiest val images
    uv run python scripts/visualize.py --names train_1 train_22 --out runs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb


def _load_instances(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.int32)


def _colorize(label_map: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
    """Random distinct color per instance, black background."""
    return label2rgb(label_map, bg_label=0, bg_color=(0, 0, 0))


def _count_instances(path: Path) -> int:
    inst = _load_instances(path)
    return int(len(np.unique(inst)) - (1 if 0 in inst else 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/train/images")
    parser.add_argument("--gt", default="data/train/instances")
    parser.add_argument("--dcan", default="runs/dcan/pred/train")
    parser.add_argument("--fcn", default="runs/fcn_baseline/pred/train")
    parser.add_argument("--names", nargs="*", default=None, help="image stems; default = busiest val images")
    parser.add_argument("--val-split", default="data/splits/fixed_val.txt")
    parser.add_argument("--max", type=int, default=6, help="how many to render when auto-picking")
    parser.add_argument("--out", default="runs/figures")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.names:
        names = args.names
    else:
        val_names = [l.strip() for l in Path(args.val_split).read_text().splitlines() if l.strip()]
        # Busiest images (most GT glands) best showcase touching-gland separation.
        names = sorted(val_names, key=lambda n: _count_instances(gt_dir / f"{n}.png"), reverse=True)[: args.max]

    titles = ["Input (H&E)", "Ground truth", "FCN baseline (object-only)", "DCAN (contour-aware)"]
    panels_for_montage = []

    for name in names:
        image = cv2.cvtColor(cv2.imread(str(Path(args.images) / f"{name}.png")), cv2.COLOR_BGR2RGB)
        gt = _load_instances(gt_dir / f"{name}.png")
        fcn = _load_instances(Path(args.fcn) / f"{name}.png")
        dcan = _load_instances(Path(args.dcan) / f"{name}.png")

        cells = [image, _colorize(gt), _colorize(fcn), _colorize(dcan)]
        counts = [None, _count_instances(gt_dir / f"{name}.png"),
                  int(fcn.max()), int(dcan.max())]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax, cell, title, cnt in zip(axes, cells, titles, counts):
            ax.imshow(cell)
            ax.set_title(title + (f"\n({cnt} instances)" if cnt is not None else ""), fontsize=11)
            ax.axis("off")
        fig.suptitle(f"{name}", fontsize=13, y=1.02)
        fig.tight_layout()
        save_path = out_dir / f"panel_{name}.png"
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        panels_for_montage.append((name, cells, titles, counts))
        print(f"  saved {save_path}")

    # Combined montage (one row per image).
    n = len(panels_for_montage)
    if n:
        fig, axes = plt.subplots(n, 4, figsize=(18, 4.2 * n))
        if n == 1:
            axes = axes[None, :]
        for row, (name, cells, _t, counts) in enumerate(panels_for_montage):
            for col, (cell, title, cnt) in enumerate(zip(cells, titles, counts)):
                ax = axes[row, col]
                ax.imshow(cell)
                if row == 0:
                    ax.set_title(title, fontsize=12)
                ax.set_ylabel(name if col == 0 else "")
                ax.set_xticks([]); ax.set_yticks([])
                if cnt is not None:
                    ax.set_xlabel(f"{cnt} inst.", fontsize=9)
        fig.tight_layout()
        montage_path = out_dir / "montage_comparison.png"
        fig.savefig(montage_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {montage_path}")

    print(f"\nFigures written to {out_dir}")


if __name__ == "__main__":
    main()
