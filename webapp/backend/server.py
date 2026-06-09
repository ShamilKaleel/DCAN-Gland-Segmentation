"""FastAPI backend for the DCAN gland-segmentation demo.

Loads the trained DCAN checkpoint once and exposes a small JSON API the React
frontend calls. Reuses the project's existing inference / post-processing /
metric utilities so the numbers shown match `scripts/evaluate.py`.

Run:
    uv run uvicorn webapp.backend.server:app --host 0.0.0.0 --port 8000
Checkpoint path can be overridden with the DCAN_CKPT env var.
"""

from __future__ import annotations

import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dcan.infer import overlap_tile_predict  # noqa: E402
from dcan.metrics import compute_all_metrics  # noqa: E402
from dcan.models.dcan import DCAN  # noqa: E402
from dcan.postprocess import fuse_predictions, postprocess_mask  # noqa: E402

from webapp.backend.render import (  # noqa: E402
    boundary_overlay,
    colorize_instances,
    heatmap,
    thumbnail_b64,
    to_b64_png,
)

# --------------------------------------------------------------------------- #
# Model + config (loaded once at import)
# --------------------------------------------------------------------------- #
CKPT_PATH = Path(os.environ.get("DCAN_CKPT", REPO / "runs" / "dcan_all85" / "best.pt"))
if not CKPT_PATH.exists():
    CKPT_PATH = REPO / "runs" / "dcan_cv" / "fold0" / "best.pt"

CFG_PATH = CKPT_PATH.parent / "config.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text()) if CFG_PATH.exists() else {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_STRIDE = CFG.get("output_stride", 8)
INFER = dict(
    crop_size=CFG.get("crop_size", 480),
    stride=CFG.get("infer_stride", 240),
    t_o=CFG.get("t_o", 0.5),
    t_c=CFG.get("t_c", 0.5),
    smooth_radius=CFG.get("smooth_radius", 3),
    min_area=CFG.get("min_area", 120),
)

# Headline results from REPORT.md, shown as header badges.
BENCHMARKS = {
    "cv_dice": 0.798,
    "cv_f1": 0.692,
    "partA_dice": 0.840,
    "partA_f1": 0.764,
}

_model = None


def get_model():
    global _model
    if _model is None:
        if not CKPT_PATH.exists():
            raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")
        model = DCAN(pretrained=False, output_stride=OUTPUT_STRIDE).to(DEVICE)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _model = model
    return _model


def run_pipeline(image_rgb: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """Run DCAN on an RGB uint8 image → (instance map, panels, stats)."""
    model = get_model()
    t0 = time.time()
    p_obj, p_cnt = overlap_tile_predict(
        model, image_rgb, crop_size=INFER["crop_size"], stride=INFER["stride"],
        device=DEVICE, is_dcan=True,
    )
    fused = fuse_predictions(p_obj, p_cnt, t_o=INFER["t_o"], t_c=INFER["t_c"])
    inst = postprocess_mask(fused, smooth_radius=INFER["smooth_radius"], min_area=INFER["min_area"])
    runtime = time.time() - t0

    panels = {
        "original": to_b64_png(image_rgb),
        "separation": to_b64_png(colorize_instances(inst)),
        "overlay": to_b64_png(boundary_overlay(image_rgb, inst)),
        "prob_obj": to_b64_png(heatmap(p_obj, "viridis")),
        "prob_cnt": to_b64_png(heatmap(p_cnt, "magma")),
    }
    stats = {"n_glands": int(inst.max()), "runtime_s": round(runtime, 2)}
    return inst, panels, stats


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="DCAN Gland Segmentation Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/info")
def info():
    return {
        "model": "DCAN (Deep Contour-Aware Network)",
        "backbone": "VGG-16 atrous /8 (DeepLab-LargeFOV), ImageNet-pretrained",
        "device": DEVICE,
        "checkpoint": str(CKPT_PATH.relative_to(REPO)),
        "benchmarks": BENCHMARKS,
    }


@app.get("/api/gallery")
def gallery():
    items = []
    for split in ("test_a", "test_b"):
        img_dir = REPO / "data" / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.png")):
            arr = np.array(Image.open(p).convert("RGB"))
            items.append({
                "split": split,
                "name": p.stem,
                "part": "Part A" if split == "test_a" else "Part B",
                "thumb": thumbnail_b64(arr),
            })
    return {"images": items}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = np.array(Image.open(BytesIO(contents)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")
    _, panels, stats = run_pipeline(image)
    return {"panels": panels, "stats": stats, "metrics": None, "source": file.filename}


class GalleryRequest(BaseModel):
    split: str
    name: str


@app.post("/api/predict_gallery")
def predict_gallery(req: GalleryRequest):
    img_path = REPO / "data" / req.split / "images" / f"{req.name}.png"
    gt_path = REPO / "data" / req.split / "instances" / f"{req.name}.png"
    if not img_path.exists() or not gt_path.exists():
        raise HTTPException(status_code=404, detail="Gallery image not found.")

    image = np.array(Image.open(img_path).convert("RGB"))
    gt = np.array(Image.open(gt_path)).astype(np.uint16)
    inst, panels, stats = run_pipeline(image)

    panels["ground_truth"] = to_b64_png(colorize_instances(gt))
    metrics = compute_all_metrics(gt, inst)
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["gt_glands"] = int(gt.max())
    return {"panels": panels, "stats": stats, "metrics": metrics,
            "source": f"{req.split}/{req.name}"}


# Serve the built React app (production single-server mode), if present.
_DIST = REPO / "webapp" / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
