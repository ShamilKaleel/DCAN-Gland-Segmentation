"""Evaluate predictions against ground-truth instance masks using GlaS metrics.

Usage:
    uv run python scripts/evaluate.py --pred runs/dcan/pred/train --gt data/train/instances
    uv run python scripts/evaluate.py --pred runs/dcan/pred/train --gt data/train/instances --names data/splits/fixed_val.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dcan.metrics import compute_all_metrics, rank_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Dir with predicted instance PNGs")
    parser.add_argument("--gt", type=str, required=True, help="Dir with GT instance PNGs")
    parser.add_argument("--names", type=str, default=None, help="Optional: text file with image names to evaluate (one per line)")
    parser.add_argument("--out", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)

    if args.names:
        names = [l.strip() for l in Path(args.names).read_text().splitlines() if l.strip()]
    else:
        names = sorted(p.stem for p in pred_dir.glob("*.png"))

    all_metrics = []
    for name in names:
        gt_path = gt_dir / f"{name}.png"
        pred_path = pred_dir / f"{name}.png"

        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED).astype(np.uint16)

        m = compute_all_metrics(gt_mask, pred_mask)
        m["name"] = name
        all_metrics.append(m)
        print(f"  {name}: F1={m['F1']:.3f}  Dice={m['Dice_obj']:.3f}  Hausdorff={m['Hausdorff_obj']:.1f}")

    # Aggregate
    metric_keys = ["F1", "Dice_obj", "Hausdorff_obj"]
    agg = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print("\n=== Aggregate Results ===")
    for k in metric_keys:
        print(f"  {k}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")

    results = {"per_image": all_metrics, "aggregate": agg}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
    else:
        print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
