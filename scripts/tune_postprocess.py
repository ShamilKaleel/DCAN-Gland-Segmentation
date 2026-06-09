"""Tune fusion / post-processing thresholds on a checkpoint — no retraining.

Computes the (p_obj, p_cnt) probability maps once per image (optionally with D4
test-time augmentation), caches them, then sweeps the fusion thresholds and
min-area over a grid and reports the best config and its gain over the current
defaults (t_o=t_c=0.5, min_area=120). Because the probability maps are cached and
the per-combo work is just CPU post-processing + metrics, the sweep is cheap.

Quantifies two no-retraining wins at once when ``--tta`` is given: TTA vs no-TTA,
each at default and best thresholds.

Usage:
    uv run python scripts/tune_postprocess.py \
        --ckpt runs/dcan_cv/fold0/best.pt \
        --names data/splits/fold0_val.txt --split train --data-dir data --tta
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dcan.infer import overlap_tile_predict, predict_with_tta
from dcan.metrics import compute_all_metrics, detection_f1, object_dice
from dcan.models.dcan import DCAN
from dcan.models.fcn_baseline import FCNBaseline
from dcan.postprocess import fuse_predictions, postprocess_fcn, postprocess_mask

DEFAULT = {"t_o": 0.5, "t_c": 0.5, "min_area": 120}


def load_model(ckpt_path: str, model_name: str, output_stride: int, device: str):
    if model_name == "dcan":
        model = DCAN(pretrained=False, output_stride=output_stride).to(device)
    else:
        model = FCNBaseline(pretrained=False, output_stride=output_stride).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _instances(prob_obj, prob_cnt, t_o, t_c, min_area, smooth_r, is_dcan):
    if is_dcan:
        fused = fuse_predictions(prob_obj, prob_cnt, t_o=t_o, t_c=t_c)
        return postprocess_mask(fused, smooth_radius=smooth_r, min_area=min_area)
    return postprocess_fcn(prob_obj, t_o=t_o, smooth_radius=smooth_r, min_area=min_area)


def _mean_full(probs, gts, cfg, smooth_r, is_dcan):
    """Full metrics (incl. slow Hausdorff) averaged over the dataset."""
    f1s, dices, hds = [], [], []
    for (p_obj, p_cnt), gt in zip(probs, gts):
        inst = _instances(p_obj, p_cnt, cfg["t_o"], cfg.get("t_c", 0.5), cfg["min_area"], smooth_r, is_dcan)
        m = compute_all_metrics(gt, inst)
        f1s.append(m["F1"]); dices.append(m["Dice_obj"]); hds.append(m["Hausdorff_obj"])
    return {"F1": float(np.mean(f1s)), "Dice_obj": float(np.mean(dices)), "Hausdorff_obj": float(np.mean(hds))}


def _mean_fast(probs, gts, cfg, smooth_r, is_dcan):
    """F1 + Dice only (skip Hausdorff) — used to rank the grid quickly."""
    f1s, dices = [], []
    for (p_obj, p_cnt), gt in zip(probs, gts):
        inst = _instances(p_obj, p_cnt, cfg["t_o"], cfg.get("t_c", 0.5), cfg["min_area"], smooth_r, is_dcan)
        f1s.append(detection_f1(gt, inst)["f1"])
        dices.append(object_dice(gt, inst))
    return float(np.mean(f1s)), float(np.mean(dices))


def sweep(probs, gts, is_dcan, smooth_r, t_o_grid, t_c_grid, min_area_grid):
    """Return (best_cfg, ranked_rows). Ranked by mean object-Dice."""
    t_c_values = t_c_grid if is_dcan else [0.5]
    rows = []
    for t_o, t_c, ma in itertools.product(t_o_grid, t_c_values, min_area_grid):
        cfg = {"t_o": t_o, "t_c": t_c, "min_area": ma}
        f1, dice = _mean_fast(probs, gts, cfg, smooth_r, is_dcan)
        rows.append((dice, f1, cfg))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[0][2], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="dcan", choices=["dcan", "fcn_baseline"])
    ap.add_argument("--output-stride", type=int, default=8)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="train", help="split subdir under data-dir holding images/ and instances/")
    ap.add_argument("--names", required=True, help="text file of image stems (e.g. data/splits/fold0_val.txt)")
    ap.add_argument("--tta", action="store_true", help="also evaluate with D4 TTA prob maps")
    ap.add_argument("--crop-size", type=int, default=480)
    ap.add_argument("--stride", type=int, default=240)
    ap.add_argument("--smooth-radius", type=int, default=3)
    ap.add_argument("--t-o-grid", default="0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--t-c-grid", default="0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--min-area-grid", default="60,120,200,300")
    ap.add_argument("--out", default=None, help="optional JSON results path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_dcan = args.model == "dcan"
    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.split / "images"
    gt_dir = data_dir / args.split / "instances"
    names = [l.strip() for l in Path(args.names).read_text().splitlines() if l.strip()]
    print(f"Model: {args.model}  device: {device}  images: {len(names)} from {args.split}\n")

    model = load_model(args.ckpt, args.model, args.output_stride, device)

    t_o_grid = [float(x) for x in args.t_o_grid.split(",")]
    t_c_grid = [float(x) for x in args.t_c_grid.split(",")]
    min_area_grid = [int(x) for x in args.min_area_grid.split(",")]

    gts = []
    probs_base: list[tuple] = []
    probs_tta: list[tuple] = []
    for i, name in enumerate(names, 1):
        image = cv2.cvtColor(cv2.imread(str(img_dir / f"{name}.png")), cv2.COLOR_BGR2RGB)
        gts.append(cv2.imread(str(gt_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16))
        probs_base.append(overlap_tile_predict(model, image, args.crop_size, args.stride, device, is_dcan))
        if args.tta:
            probs_tta.append(predict_with_tta(model, image, args.crop_size, args.stride, device, is_dcan))
        print(f"  [{i}/{len(names)}] {name} prob maps computed", end="\r")
    print(" " * 60, end="\r")

    results = {}
    variants = [("no-TTA", probs_base)] + ([("TTA", probs_tta)] if args.tta else [])
    for label, probs in variants:
        default_m = _mean_full(probs, gts, DEFAULT, args.smooth_radius, is_dcan)
        best_cfg, rows = sweep(probs, gts, is_dcan, args.smooth_radius, t_o_grid, t_c_grid, min_area_grid)
        best_m = _mean_full(probs, gts, best_cfg, args.smooth_radius, is_dcan)
        results[label] = {"default": {"cfg": DEFAULT, "metrics": default_m},
                          "best": {"cfg": best_cfg, "metrics": best_m},
                          "top5": [{"Dice_obj": d, "F1": f, "cfg": c} for d, f, c in rows[:5]]}

        print(f"===== {label} =====")
        print(f"  default {DEFAULT}:")
        print(f"     F1={default_m['F1']:.4f}  Dice={default_m['Dice_obj']:.4f}  HD={default_m['Hausdorff_obj']:.2f}")
        print(f"  best {best_cfg}:")
        print(f"     F1={best_m['F1']:.4f}  Dice={best_m['Dice_obj']:.4f}  HD={best_m['Hausdorff_obj']:.2f}")
        print(f"  delta: dDice={best_m['Dice_obj']-default_m['Dice_obj']:+.4f}  "
              f"dF1={best_m['F1']-default_m['F1']:+.4f}  dHD={best_m['Hausdorff_obj']-default_m['Hausdorff_obj']:+.2f}")
        print("  top-5 by Dice:")
        for d, f, c in rows[:5]:
            print(f"     Dice={d:.4f} F1={f:.4f}  {c}")
        print()

    if args.tta:
        b0 = results["no-TTA"]["default"]["metrics"]
        b1 = results["TTA"]["best"]["metrics"]
        print("===== combined (no-TTA default -> TTA + best thresholds) =====")
        print(f"  Dice {b0['Dice_obj']:.4f} -> {b1['Dice_obj']:.4f} ({b1['Dice_obj']-b0['Dice_obj']:+.4f})")
        print(f"  F1   {b0['F1']:.4f} -> {b1['F1']:.4f} ({b1['F1']-b0['F1']:+.4f})")
        print(f"  HD   {b0['Hausdorff_obj']:.2f} -> {b1['Hausdorff_obj']:.2f} ({b1['Hausdorff_obj']-b0['Hausdorff_obj']:+.2f})")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
