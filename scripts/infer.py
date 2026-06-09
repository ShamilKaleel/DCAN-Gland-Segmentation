"""Run inference on a dataset split and save instance predictions.

Usage:
    uv run python scripts/infer.py --config configs/dcan.yaml --ckpt runs/dcan/best.pt --split test_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dcan.infer import overlap_tile_predict
from dcan.models.dcan import DCAN
from dcan.models.fcn_baseline import FCNBaseline
from dcan.postprocess import fuse_predictions, postprocess_fcn, postprocess_mask
from dcan.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--dump-probs",
        action="store_true",
        help="also save p_obj/p_cnt probability maps as .npy for offline threshold sweeping",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="optional text file of image stems to predict (default: all images in the split)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    is_dcan = cfg.get("model", "dcan") == "dcan"

    data_dir = Path(cfg["data_dir"])
    img_dir = data_dir / args.split / "images"

    out_dir = Path(args.out) if args.out else Path(cfg["run_dir"]) / "pred" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = out_dir.parent / f"{args.split}_probs"
    if args.dump_probs:
        prob_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if is_dcan:
        model = DCAN(pretrained=False).to(device)
    else:
        model = FCNBaseline(pretrained=False).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    crop_size = cfg.get("crop_size", 480)
    stride = cfg.get("infer_stride", 240)
    t_o = cfg.get("t_o", 0.5)
    t_c = cfg.get("t_c", 0.5)
    smooth_r = cfg.get("smooth_radius", 3)
    min_area = cfg.get("min_area", 120)

    if args.names:
        names = [l.strip() for l in Path(args.names).read_text().splitlines() if l.strip()]
    else:
        names = sorted(p.stem for p in img_dir.glob("*.png"))
    print(f"Running inference on {len(names)} images from {args.split}")

    for name in tqdm(names):
        image = cv2.imread(str(img_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        p_obj, p_cnt = overlap_tile_predict(
            model, image, crop_size=crop_size, stride=stride,
            device=device, is_dcan=is_dcan,
        )

        if is_dcan:
            fused = fuse_predictions(p_obj, p_cnt, t_o=t_o, t_c=t_c)
            instances = postprocess_mask(fused, smooth_radius=smooth_r, min_area=min_area)
        else:
            instances = postprocess_fcn(p_obj, t_o=t_o, smooth_radius=smooth_r, min_area=min_area)

        cv2.imwrite(str(out_dir / f"{name}.png"), instances.astype(np.uint16))

        if args.dump_probs:
            np.save(prob_dir / f"{name}_obj.npy", p_obj.astype(np.float32))
            if is_dcan and p_cnt is not None:
                np.save(prob_dir / f"{name}_cnt.npy", p_cnt.astype(np.float32))

    print(f"Predictions saved to {out_dir}")


if __name__ == "__main__":
    main()
