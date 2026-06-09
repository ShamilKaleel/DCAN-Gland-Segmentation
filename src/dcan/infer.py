"""Overlap-tile inference for variable-size images."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dcan.data.augment import IMAGENET_MEAN, IMAGENET_STD


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize uint8 RGB image with ImageNet stats → float32 CHW."""
    img = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)  # CHW


def overlap_tile_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    crop_size: int = 480,
    stride: int = 240,
    device: str = "cuda",
    is_dcan: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run overlap-tile inference and return (p_obj, p_cnt) probability maps.

    For FCN baseline, p_cnt is None.
    """
    model.eval()
    h, w = image.shape[:2]

    # Pad so that tiles cover the full image
    pad_h = max(crop_size - h, 0) + (stride - (max(h, crop_size) - crop_size) % stride) % stride
    pad_w = max(crop_size - w, 0) + (stride - (max(w, crop_size) - crop_size) % stride) % stride
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]

    sum_obj = np.zeros((2, ph, pw), dtype=np.float64)
    sum_cnt = np.zeros((2, ph, pw), dtype=np.float64) if is_dcan else None
    counts = np.zeros((ph, pw), dtype=np.float64)

    with torch.no_grad():
        for y in range(0, ph - crop_size + 1, stride):
            for x in range(0, pw - crop_size + 1, stride):
                tile = padded[y : y + crop_size, x : x + crop_size]
                inp = torch.from_numpy(normalize_image(tile)).unsqueeze(0).to(device)
                out = model(inp)

                p_o = F.softmax(out["obj_main"], dim=1)[0].cpu().numpy()
                sum_obj[:, y : y + crop_size, x : x + crop_size] += p_o

                if is_dcan:
                    p_c = F.softmax(out["cnt_main"], dim=1)[0].cpu().numpy()
                    sum_cnt[:, y : y + crop_size, x : x + crop_size] += p_c

                counts[y : y + crop_size, x : x + crop_size] += 1.0

    counts = np.maximum(counts, 1.0)
    avg_obj = sum_obj / counts[None, :, :]
    p_obj = avg_obj[1, :h, :w]  # foreground probability

    p_cnt = None
    if is_dcan:
        avg_cnt = sum_cnt / counts[None, :, :]
        p_cnt = avg_cnt[1, :h, :w]

    return p_obj, p_cnt
