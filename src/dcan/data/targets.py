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
    ids = ids[ids > 0]
    contour = np.zeros_like(instance_mask, dtype=np.uint8)
    selem = disk(dilation_radius)

    for gid in ids:
        inst = (instance_mask == gid)
        boundary = binary_dilation(inst, iterations=1) & ~binary_erosion(inst, iterations=1)
        dilated = binary_dilation(boundary, structure=selem)
        contour[dilated] = 1

    return contour
