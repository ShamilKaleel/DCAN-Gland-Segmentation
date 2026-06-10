"""Post-processing: fusion rule, morphology, connected-component labeling."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import disk, opening, remove_small_objects


def fuse_predictions(
    p_obj: np.ndarray,
    p_cnt: np.ndarray,
    t_o: float = 0.5,
    t_c: float = 0.5,
) -> np.ndarray:
    """Fusion rule from DCAN: foreground where p_obj >= t_o AND p_cnt < t_c.

    A pixel is kept as gland body only if the object branch is confident it is
    foreground AND the contour branch is confident it is NOT an edge. Subtracting
    the predicted contour ridge is what carves apart touching glands so connected-
    component labelling then yields separate instances. (paper Eq. 3, t_o=t_c=0.5)
    """
    return ((p_obj >= t_o) & (p_cnt < t_c)).astype(np.uint8)


def postprocess_mask(
    fused: np.ndarray,
    smooth_radius: int = 3,
    min_area: int = 120,
) -> np.ndarray:
    """Apply morphological cleaning and return uint16 instance label map.

    Order matters: opening -> fill-holes -> label -> drop specks -> relabel.
    """
    selem = disk(smooth_radius)
    cleaned = opening(fused, selem)                          # remove thin bridges / noise specks
    cleaned = binary_fill_holes(cleaned).astype(np.uint8)    # fill gland interiors left hollow by fusion
    labeled = label(cleaned).astype(np.uint16)              # connected components -> one id per blob
    labeled = remove_small_objects(labeled, min_size=min_area)  # drop tiny false-positive objects
    # remove_small_objects leaves gaps in the id sequence; relabel so ids are 1..N.
    labeled = label(labeled > 0).astype(np.uint16)
    return labeled


def postprocess_fcn(
    p_obj: np.ndarray,
    t_o: float = 0.5,
    smooth_radius: int = 3,
    min_area: int = 120,
) -> np.ndarray:
    """Post-process for object-only FCN baseline (no contour fusion)."""
    binary = (p_obj >= t_o).astype(np.uint8)
    return postprocess_mask(binary, smooth_radius, min_area)
