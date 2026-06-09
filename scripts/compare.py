"""Compare methods on the official GlaS metrics and compute rank-sum scoring.

This closes objective 2 of the proposal (official evaluation reproduction): it
aggregates per-method results and applies the GlaS rank-sum ranking logic
(`dcan.metrics.rank_sum`).

Usage:
    uv run python scripts/compare.py
    uv run python scripts/compare.py \
        --runs DCAN=runs/dcan/results_val.json FCN=runs/fcn_baseline/results_val.json \
        --paper --out runs/comparison

Outputs a markdown table (<out>.md), a CSV (<out>.csv) and prints to stdout.
The paper reference rows (--paper) are shown for context only and are NOT
included in the rank-sum (they come from the hidden GlaS test set, a different
split, so they are not directly comparable to our held-out validation scores).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dcan.metrics import rank_sum

METRICS = ["F1", "Dice_obj", "Hausdorff_obj"]

# CUMedVision2 (= DCAN) as reported by Chen et al., CVPR 2016, on the hidden
# GlaS test set. Reference only -- different split from our validation labels.
PAPER_REFERENCE = {
    "DCAN paper (Part A)": {"F1": 0.9116, "Dice_obj": 0.8974, "Hausdorff_obj": 45.42},
    "DCAN paper (Part B)": {"F1": 0.7158, "Dice_obj": 0.7810, "Hausdorff_obj": 160.35},
}


def load_aggregate(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    data = json.loads(Path(path).read_text())
    agg = data["aggregate"]
    means = {k: float(agg[k]["mean"]) for k in METRICS}
    stds = {k: float(agg[k].get("std", float("nan"))) for k in METRICS}
    return means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "DCAN=runs/dcan/results_val.json",
            "FCN=runs/fcn_baseline/results_val.json",
        ],
        help="label=path pairs pointing to results JSON files",
    )
    parser.add_argument("--paper", action="store_true", help="append paper reference rows (not ranked)")
    parser.add_argument("--out", default="runs/comparison", help="output path prefix (.md and .csv)")
    args = parser.parse_args()

    means: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        means[label], stds[label] = load_aggregate(path)

    # Rank-sum over OUR methods only (lower total rank = better overall).
    ranks = rank_sum(means)

    # ---- Build table rows ----
    header = ["Method", "F1 ↑", "Dice_obj ↑", "Hausdorff_obj ↓", "Rank-sum ↓"]
    rows = []
    for label in means:
        m, s = means[label], stds[label]
        rows.append([
            label,
            f"{m['F1']:.4f} ± {s['F1']:.3f}",
            f"{m['Dice_obj']:.4f} ± {s['Dice_obj']:.3f}",
            f"{m['Hausdorff_obj']:.2f} ± {s['Hausdorff_obj']:.1f}",
            str(ranks[label]),
        ])
    # Order our methods by rank-sum (best first).
    rows.sort(key=lambda r: int(r[-1]))

    if args.paper:
        for label, m in PAPER_REFERENCE.items():
            rows.append([
                label,
                f"{m['F1']:.4f}",
                f"{m['Dice_obj']:.4f}",
                f"{m['Hausdorff_obj']:.2f}",
                "— (ref)",
            ])

    # ---- Markdown ----
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md_text = "\n".join(md) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md_text)
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(md_text)
    print(f"Rank-sum (lower is better, our methods only): {ranks}")
    print(f"\nSaved: {out.with_suffix('.md')} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
