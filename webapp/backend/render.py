"""Rendering helpers for the demo: colorized panels and base64 encoding.

All functions return uint8 RGB arrays (H, W, 3); ``to_b64_png`` turns one into a
data-URI string the browser can display directly.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from matplotlib import colormaps
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

# A fixed, high-contrast palette so instance colors are vivid and stable.
_PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (102, 194, 165), (252, 141, 98), (141, 160, 203), (231, 138, 195),
    (166, 216, 84), (255, 217, 47), (229, 196, 148), (179, 179, 179),
]


def colorize_instances(label_map: np.ndarray) -> np.ndarray:
    """Color each instance id with a distinct color on a black background."""
    colors = [tuple(c / 255.0 for c in rgb) for rgb in _PALETTE]
    rgb = label2rgb(label_map, bg_label=0, bg_color=(0, 0, 0), colors=colors)
    return (rgb * 255).astype(np.uint8)


def boundary_overlay(image: np.ndarray, label_map: np.ndarray, color=(255, 255, 0)) -> np.ndarray:
    """Draw instance boundaries on top of the original H&E image."""
    out = image.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    boundaries = find_boundaries(label_map, mode="outer")
    out[boundaries] = color
    return out.astype(np.uint8)


def heatmap(prob: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """Map a [0,1] probability map to an RGB colormap image."""
    p = np.clip(prob, 0.0, 1.0)
    rgba = colormaps[cmap](p)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def to_b64_png(rgb: np.ndarray) -> str:
    """Encode a uint8 RGB array as a base64 PNG data URI."""
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail_b64(image: np.ndarray, max_side: int = 160) -> str:
    """Downscaled base64 thumbnail for the gallery grid."""
    img = Image.fromarray(image.astype(np.uint8))
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
