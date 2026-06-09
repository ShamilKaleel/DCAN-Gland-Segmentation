"""Decode DatasetNinja / Supervisely bitmap annotations into instance masks."""

from __future__ import annotations

import base64
import json
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def decode_bitmap(data_b64: str) -> np.ndarray:
    raw = base64.b64decode(data_b64)
    png_bytes = zlib.decompress(raw)
    img = Image.open(BytesIO(png_bytes))
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def parse_annotation(ann_path: str | Path) -> tuple[tuple[int, int], np.ndarray]:
    """Parse a DatasetNinja annotation JSON and return (H, W) and uint16 instance mask."""
    with open(ann_path, "r") as f:
        ann = json.load(f)

    h = ann["size"]["height"]
    w = ann["size"]["width"]
    instance_mask = np.zeros((h, w), dtype=np.uint16)

    for idx, obj in enumerate(ann["objects"], start=1):
        bmp = obj["bitmap"]
        mask = decode_bitmap(bmp["data"])
        ox, oy = bmp["origin"]  # origin is [x, y]
        mh, mw = mask.shape
        # clip to image bounds
        y0, x0 = max(oy, 0), max(ox, 0)
        y1, x1 = min(oy + mh, h), min(ox + mw, w)
        sy0, sx0 = y0 - oy, x0 - ox
        roi = mask[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
        instance_mask[y0:y1, x0:x1][roi > 0] = idx

    return (h, w), instance_mask


def get_image_tags(ann_path: str | Path) -> dict[str, str | None]:
    """Extract image-level tags (benign/malignant, patient_id, etc.)."""
    with open(ann_path, "r") as f:
        ann = json.load(f)
    tags = {}
    for t in ann.get("tags", []):
        tags[t["name"]] = t.get("value")
    return tags
