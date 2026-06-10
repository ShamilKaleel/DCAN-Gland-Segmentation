"""Generate object masks and contour masks from instance label maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk


def instance_to_object_mask(instance_mask: np.ndarray) -> np.ndarray:
    return (instance_mask > 0).astype(np.uint8)


def instance_to_contour_mask(
    instance_mask: np.ndarray, dilation_radius: int = 3
) -> np.ndarray:
    """Per-instance boundary extraction + dilation.

    Touching glands get a contour between them because boundaries are
    computed per-instance before merging.
    """
    ids = np.unique(instance_mask)
    ids = ids[ids > 0]  # 0 is background
    contour = np.zeros_like(instance_mask, dtype=np.uint8)
    selem = disk(dilation_radius)

    # Compute each gland's boundary SEPARATELY, then union. This is the crucial
    # detail: where two glands touch, each contributes its own edge, so the shared
    # seam is labelled as contour from both sides. A boundary taken on the merged
    # foreground would miss that seam and the glands could never be split apart.
    for gid in ids:
        inst = (instance_mask == gid)
        # 1-px ring = (dilate by 1) minus (erode by 1) around this instance only.
        boundary = binary_dilation(inst, iterations=1) & ~binary_erosion(inst, iterations=1)
        # Thicken the ring to `dilation_radius` so the contour class isn't a single
        # pixel line (too rare to learn) and the seam is a visible band.
        dilated = binary_dilation(boundary, structure=selem)
        contour[dilated] = 1

    return contour
