"""Official GlaS evaluation metrics: F1 detection, Dice_obj, Hausdorff_obj, rank-sum."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label, regionprops


def _match_instances(gt: np.ndarray, pred: np.ndarray):
    """Match GT instances to predicted instances using the official GlaS rule.

    Returns (tp_pairs, fp_ids, fn_ids).

    A predicted object is a true positive if it overlaps >= 50% of the area of
    the ground-truth object it is matched to. Matching is **one-to-one**: each
    predicted object can be the TP match of at most one GT object (and vice
    versa). This prevents a single merged blob from being counted as a TP for
    every gland it covers, which would otherwise inflate the score of methods
    that fail to separate touching glands. Assignment is greedy by intersection
    size (largest overlaps matched first), which is equivalent to the maximal-
    overlap criterion used by the GlaS challenge.
    """
    gt_ids = sorted(int(i) for i in np.unique(gt) if i != 0)
    pred_ids = sorted(int(i) for i in np.unique(pred) if i != 0)

    gt_area = {gid: int((gt == gid).sum()) for gid in gt_ids}

    # Candidate (intersection, gid, pid) pairs meeting the 50%-of-GT criterion.
    candidates = []
    for gid in gt_ids:
        overlaps = pred[gt == gid]
        pids, counts = np.unique(overlaps, return_counts=True)
        for pid, inter in zip(pids, counts):
            if pid == 0:
                continue
            if inter >= 0.5 * gt_area[gid]:
                candidates.append((int(inter), gid, int(pid)))

    # Greedy one-to-one assignment, largest intersection first.
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp_pairs = []
    for _inter, gid, pid in candidates:
        if gid in matched_gt or pid in matched_pred:
            continue
        tp_pairs.append((gid, pid))
        matched_gt.add(gid)
        matched_pred.add(pid)

    fp_ids = set(pred_ids) - matched_pred
    fn_ids = set(gt_ids) - matched_gt
    return tp_pairs, fp_ids, fn_ids


def detection_f1(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    # Counts are over OBJECTS, not pixels: tp = matched glands, fp = spurious
    # predicted glands, fn = missed GT glands (see the one-to-one rule above).
    tp_pairs, fp_ids, fn_ids = _match_instances(gt, pred)
    tp = len(tp_pairs)
    fp = len(fp_ids)
    fn = len(fn_ids)
    precision = tp / max(tp + fp, 1)   # of predicted glands, how many are real
    recall = tp / max(tp + fn, 1)      # of real glands, how many we found
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return 2.0 * inter / max(a.sum() + b.sum(), 1e-8)


def object_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Object-level Dice as defined in the GlaS challenge.

    D_obj = 0.5 * [sum_i w_i * Dice(G_i, S*(G_i)) + sum_j w_j * Dice(S_j, G*(S_j))]
    """
    gt_ids = set(np.unique(gt)) - {0}
    pred_ids = set(np.unique(pred)) - {0}

    total_gt_area = sum((gt == gid).sum() for gid in gt_ids) if gt_ids else 1
    total_pred_area = sum((pred == pid).sum() for pid in pred_ids) if pred_ids else 1

    # GT → best matching pred
    term1 = 0.0
    for gid in gt_ids:
        g_mask = gt == gid
        w = g_mask.sum() / total_gt_area
        overlaps = pred[g_mask]
        candidates = set(np.unique(overlaps)) - {0}
        best_dice = 0.0
        for pid in candidates:
            p_mask = pred == pid
            d = _dice(g_mask, p_mask)
            best_dice = max(best_dice, d)
        term1 += w * best_dice

    # Pred → best matching GT
    term2 = 0.0
    for pid in pred_ids:
        p_mask = pred == pid
        w = p_mask.sum() / total_pred_area
        overlaps = gt[p_mask]
        candidates = set(np.unique(overlaps)) - {0}
        best_dice = 0.0
        for gid in candidates:
            g_mask = gt == gid
            d = _dice(g_mask, p_mask)
            best_dice = max(best_dice, d)
        term2 += w * best_dice

    return 0.5 * (term1 + term2)


def _hausdorff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    pts_a = np.argwhere(mask_a)
    pts_b = np.argwhere(mask_b)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")
    d1 = directed_hausdorff(pts_a, pts_b)[0]
    d2 = directed_hausdorff(pts_b, pts_a)[0]
    return max(d1, d2)


def object_hausdorff(gt: np.ndarray, pred: np.ndarray) -> float:
    """Object-level Hausdorff distance (area-weighted, same matching as Dice)."""
    gt_ids = set(np.unique(gt)) - {0}
    pred_ids = set(np.unique(pred)) - {0}

    if not gt_ids and not pred_ids:
        return 0.0

    total_gt_area = sum((gt == gid).sum() for gid in gt_ids) if gt_ids else 1
    total_pred_area = sum((pred == pid).sum() for pid in pred_ids) if pred_ids else 1

    # GT → best matching pred
    term1 = 0.0
    for gid in gt_ids:
        g_mask = gt == gid
        w = g_mask.sum() / total_gt_area
        overlaps = pred[g_mask]
        candidates = set(np.unique(overlaps)) - {0}
        best_hd = float("inf")
        for pid in candidates:
            p_mask = pred == pid
            hd = _hausdorff(g_mask, p_mask)
            best_hd = min(best_hd, hd)
        if not candidates:
            h_img = gt.shape[0]
            w_img = gt.shape[1]
            best_hd = np.sqrt(h_img**2 + w_img**2)
        term1 += w * best_hd

    # Pred → best matching GT
    term2 = 0.0
    for pid in pred_ids:
        p_mask = pred == pid
        w = p_mask.sum() / total_pred_area
        overlaps = gt[p_mask]
        candidates = set(np.unique(overlaps)) - {0}
        best_hd = float("inf")
        for gid in candidates:
            g_mask = gt == gid
            hd = _hausdorff(g_mask, p_mask)
            best_hd = min(best_hd, hd)
        if not candidates:
            h_img = gt.shape[0]
            w_img = gt.shape[1]
            best_hd = np.sqrt(h_img**2 + w_img**2)
        term2 += w * best_hd

    return 0.5 * (term1 + term2)


def compute_all_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    f1_res = detection_f1(gt, pred)
    d_obj = object_dice(gt, pred)
    h_obj = object_hausdorff(gt, pred)
    return {
        "F1": f1_res["f1"],
        "precision": f1_res["precision"],
        "recall": f1_res["recall"],
        "Dice_obj": d_obj,
        "Hausdorff_obj": h_obj,
    }


def rank_sum(results: dict[str, dict[str, float]], higher_better: dict[str, bool] | None = None) -> dict[str, int]:
    """Compute rank-sum across methods for each metric.

    Args:
        results: {method_name: {metric_name: value}}
        higher_better: {metric_name: True/False}, defaults to F1/Dice=higher, Hausdorff=lower.

    Returns:
        {method_name: total_rank} (lower is better).
    """
    if higher_better is None:
        higher_better = {"F1": True, "Dice_obj": True, "Hausdorff_obj": False}

    methods = list(results.keys())
    total = {m: 0 for m in methods}

    for metric, ascending in higher_better.items():
        vals = [(m, results[m].get(metric, float("nan"))) for m in methods]
        vals.sort(key=lambda x: x[1], reverse=ascending)
        for rank, (m, _) in enumerate(vals, start=1):
            total[m] += rank

    return total
