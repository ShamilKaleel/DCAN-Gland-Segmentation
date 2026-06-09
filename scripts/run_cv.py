"""Stratified 2-fold cross-validation driver (proposal objective 3).

For each fold: train on ``fold{f}_train``, predict ``fold{f}_val``, evaluate.
Then pool per-image results from both folds — each of the 85 images is tested
exactly once — and report the mean ± std. This pooled value is the headline,
cross-validated number.

Usage:
    uv run python scripts/run_cv.py --config configs/dcan.yaml
    uv run python scripts/run_cv.py --config configs/fcn_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
METRIC_KEYS = ["F1", "Dice_obj", "Hausdorff_obj"]


def run(cmd: list[str]) -> None:
    print("\n+ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": "src"}
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=2)
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.config).read_text())
    name = base.get("model", "model")
    cv_dir = REPO / "runs" / f"{name}_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    fold_results = []
    for f in range(args.folds):
        run_dir = cv_dir / f"fold{f}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = dict(base)
        cfg["train_split"] = f"data/splits/fold{f}_train.txt"
        cfg["val_split"] = f"data/splits/fold{f}_val.txt"
        cfg["run_dir"] = f"runs/{name}_cv/fold{f}"
        tmp_cfg = run_dir / "config_fold.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg))
        cfg_rel = f"runs/{name}_cv/fold{f}/config_fold.yaml"
        val_split = f"data/splits/fold{f}_val.txt"

        run([py, "scripts/train.py", "--config", cfg_rel])
        run([py, "scripts/infer.py", "--config", cfg_rel,
             "--ckpt", f"runs/{name}_cv/fold{f}/best.pt", "--split", "train", "--names", val_split])
        out_json = f"runs/{name}_cv/fold{f}/results.json"
        run([py, "scripts/evaluate.py", "--pred", f"runs/{name}_cv/fold{f}/pred/train",
             "--gt", "data/train/instances", "--names", val_split, "--out", out_json])
        fold_results.append(json.loads((REPO / out_json).read_text()))

    # Pool per-image results across folds (each image appears once).
    pooled = []
    for fr in fold_results:
        pooled.extend(fr["per_image"])
    agg = {
        k: {"mean": float(np.mean([p[k] for p in pooled])), "std": float(np.std([p[k] for p in pooled]))}
        for k in METRIC_KEYS
    }
    out = {"per_image": pooled, "aggregate": agg, "n_images": len(pooled), "folds": args.folds}
    out_path = cv_dir / "results_cv.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== {name}: {args.folds}-fold CV (pooled over {len(pooled)} images) ===")
    for k in METRIC_KEYS:
        print(f"  {k}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
