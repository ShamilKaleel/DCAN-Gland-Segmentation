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

    # GlaS images vary in size and exceed the 480² training crop, so we slide a
    # crop_size window with 50% overlap (stride 240). Reflect-pad up to the next
    # whole number of strides so every pixel is covered by >=1 tile (and most by
    # several), which removes the seams a non-overlapping tiling would leave.
    pad_h = max(crop_size - h, 0) + (stride - (max(h, crop_size) - crop_size) % stride) % stride
    pad_w = max(crop_size - w, 0) + (stride - (max(w, crop_size) - crop_size) % stride) % stride
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]

    # Accumulate the per-class softmax probabilities and a hit-count per pixel, so
    # overlapping tiles are AVERAGED (not overwritten).
    sum_obj = np.zeros((2, ph, pw), dtype=np.float64)
    sum_cnt = np.zeros((2, ph, pw), dtype=np.float64) if is_dcan else None
    counts = np.zeros((ph, pw), dtype=np.float64)

    with torch.no_grad():
        for y in range(0, ph - crop_size + 1, stride):
            for x in range(0, pw - crop_size + 1, stride):
                tile = padded[y : y + crop_size, x : x + crop_size]
                inp = torch.from_numpy(normalize_image(tile)).unsqueeze(0).to(device)
                out = model(inp)

                # softmax -> probabilities (averaging logits across tiles would be wrong).
                p_o = F.softmax(out["obj_main"], dim=1)[0].cpu().numpy()
                sum_obj[:, y : y + crop_size, x : x + crop_size] += p_o

                if is_dcan:
                    p_c = F.softmax(out["cnt_main"], dim=1)[0].cpu().numpy()
                    sum_cnt[:, y : y + crop_size, x : x + crop_size] += p_c

                counts[y : y + crop_size, x : x + crop_size] += 1.0

    counts = np.maximum(counts, 1.0)              # guard against divide-by-zero
    avg_obj = sum_obj / counts[None, :, :]        # mean probability per pixel
    p_obj = avg_obj[1, :h, :w]  # channel 1 = foreground; crop padding back off

    p_cnt = None
    if is_dcan:
        avg_cnt = sum_cnt / counts[None, :, :]
        p_cnt = avg_cnt[1, :h, :w]

    return p_obj, p_cnt


# --------------------------------------------------------------------------- #
# Test-time augmentation (D4: 4 rotations x optional horizontal flip = 8 views)
# --------------------------------------------------------------------------- #
# Each prediction is run on a flipped/rotated copy of the image and the returned
# probability map is mapped back to the original orientation before averaging.
# Both np.rot90 (axes 0,1) and np.fliplr (axis 1) act on the spatial dims, so the
# same helpers work for (H, W, 3) images and (H, W) probability maps.
_D4 = [(k, flip) for flip in (False, True) for k in range(4)]


def _apply_tf(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    return np.fliplr(out) if flip else out


def _invert_tf(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    # Forward is fliplr(rot90(x, k)); invert by undoing the flip, then the rotation.
    if flip:
        arr = np.fliplr(arr)
    return np.rot90(arr, -k)


def predict_with_tta(
    model: torch.nn.Module,
    image: np.ndarray,
    crop_size: int = 480,
    stride: int = 240,
    device: str = "cuda",
    is_dcan: bool = True,
    transforms: str = "d4",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Test-time-augmented `overlap_tile_predict`.

    Averages probability maps over the D4 symmetry group (8 views), or the 4
    rotations only (``transforms="rot4"``). Free accuracy at inference, no
    retraining. Returns (p_obj, p_cnt) in the original orientation.
    """
    views = _D4 if transforms == "d4" else [(k, False) for k in range(4)]

    sum_obj: np.ndarray | None = None
    sum_cnt: np.ndarray | None = None
    for k, flip in views:
        t_img = np.ascontiguousarray(_apply_tf(image, k, flip))
        p_obj, p_cnt = overlap_tile_predict(model, t_img, crop_size, stride, device, is_dcan)
        p_obj = _invert_tf(p_obj, k, flip)
        sum_obj = p_obj if sum_obj is None else sum_obj + p_obj
        if is_dcan and p_cnt is not None:
            p_cnt = _invert_tf(p_cnt, k, flip)
            sum_cnt = p_cnt if sum_cnt is None else sum_cnt + p_cnt

    n = len(views)
    p_obj = sum_obj / n
    p_cnt = (sum_cnt / n) if (is_dcan and sum_cnt is not None) else None
    return p_obj, p_cnt
