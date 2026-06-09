"""Per-metric comparison graphs: DCAN vs FCN baseline on the held-out test set.

For each official GlaS metric (detection F1, object-level Dice, object-level
Hausdorff) this draws ONE figure comparing the two models in the project:
  * left  — grouped bar of the mean (+/- std) with value labels and the DCAN
            paper score as a dashed reference line;
  * right — the distribution of all per-image scores (box + jittered points),
            so the spread/robustness is visible, not just the average.
A combined 1x3 overview of the three bar charts is also written for a single
summary slide.

Reuses the result-JSON layout and paper-reference values from
``scripts/compare.py``. matplotlib only (already a project dependency).

Usage:
    uv run python scripts/plot_metrics.py
    uv run python scripts/plot_metrics.py \
        --dcan runs/dcan_all85/results_test_a.json \
        --fcn  runs/fcn_all85/results_test_a.json \
        --out  runs/figures_eval --no-paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Consistent identity for the two models across every figure.
DCAN_COLOR = "#1b9e77"  # teal
FCN_COLOR = "#d95f02"  # orange
PAPER_COLOR = "#666666"  # grey dashed reference

# (json key, display title, axis label, higher_is_better, DCAN-paper reference).
# Paper values are CUMedVision2 (= DCAN) as reported by Chen et al., CVPR 2016
# on the public test set -- same numbers used in scripts/compare.py.
METRICS = [
    ("F1", "Detection F1", "F1 score", True, 0.9116),
    ("Dice_obj", "Object-level Dice", "Dice", True, 0.8974),
    ("Hausdorff_obj", "Object-level Hausdorff", "Hausdorff distance (px)", False, 45.42),
]


def load_results(path: str | Path) -> tuple[dict, list[dict]]:
    """Return (aggregate, per_image) from a results JSON."""
    data = json.loads(Path(path).read_text())
    return data["aggregate"], data["per_image"]


def _direction_note(higher_better: bool) -> str:
    return "↑ higher is better" if higher_better else "↓ lower is better"


def _per_image_values(per_image: list[dict], key: str) -> np.ndarray:
    vals = np.array([float(r[key]) for r in per_image if np.isfinite(float(r[key]))])
    return vals


def _bar_panel(ax, key, title, ylabel, higher_better, paper, dcan_agg, fcn_agg, show_paper):
    """Grouped bar: DCAN vs FCN mean +/- std, value labels, paper reference line."""
    labels = ["DCAN", "FCN baseline"]
    means = [dcan_agg[key]["mean"], fcn_agg[key]["mean"]]
    stds = [dcan_agg[key].get("std", 0.0), fcn_agg[key].get("std", 0.0)]
    colors = [DCAN_COLOR, FCN_COLOR]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.6,
                  edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    # Value labels above each bar (offset clear of the error bar).
    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f"{m:.3f}", xy=(bar.get_x() + bar.get_width() / 2, m + s),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    if show_paper:
        ax.axhline(paper, ls="--", color=PAPER_COLOR, lw=1.5)
        # Centred between the two bars so it never overlaps a value label.
        ax.annotate(f"DCAN paper = {paper:g}", xy=(0.5, paper),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=PAPER_COLOR)

    # Headroom so labels / paper line never clip.
    top = max(means[0] + stds[0], means[1] + stds[1], paper if show_paper else 0)
    ax.set_ylim(0, top * 1.18)
    ax.set_title(f"{title}  ({_direction_note(higher_better)})", fontsize=11)


def _dist_panel(ax, key, ylabel, paper, dcan_vals, fcn_vals, show_paper):
    """Box plot of per-image scores for DCAN vs FCN with jittered points."""
    data = [dcan_vals, fcn_vals]
    colors = [DCAN_COLOR, FCN_COLOR]
    positions = [1, 2]

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                               "markeredgecolor": "black", "markersize": 6})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_color("black")

    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, data, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=14, color=c,
                   alpha=0.6, edgecolor="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(["DCAN", "FCN baseline"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    if show_paper:
        ax.axhline(paper, ls="--", color=PAPER_COLOR, lw=1.5)
        ax.annotate(f"DCAN paper = {paper:g}", xy=(1.5, paper),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=PAPER_COLOR)
    ax.set_title(f"Per-image distribution (n={len(dcan_vals)})", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcan", default="runs/dcan_all85/results_test_a.json")
    parser.add_argument("--fcn", default="runs/fcn_all85/results_test_a.json")
    parser.add_argument("--out", default="runs/figures_eval")
    parser.add_argument("--paper", dest="paper", action="store_true", default=True,
                        help="show DCAN paper reference line (default on)")
    parser.add_argument("--no-paper", dest="paper", action="store_false")
    args = parser.parse_args()

    dcan_agg, dcan_pi = load_results(args.dcan)
    fcn_agg, fcn_pi = load_results(args.fcn)
    n_imgs = len(dcan_pi)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DCAN: {args.dcan}\nFCN : {args.fcn}\nimages: {n_imgs}\n")

    # ---- One figure per metric: bar (left) + distribution (right) ----
    for key, title, ylabel, higher_better, paper in METRICS:
        dcan_vals = _per_image_values(dcan_pi, key)
        fcn_vals = _per_image_values(fcn_pi, key)

        fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(11, 4.6))
        _bar_panel(ax_bar, key, title, ylabel, higher_better, paper,
                   dcan_agg, fcn_agg, args.paper)
        _dist_panel(ax_box, key, ylabel, paper, dcan_vals, fcn_vals, args.paper)
        fig.suptitle(f"{title} — DCAN vs FCN baseline ({n_imgs} test images)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_png = out_dir / f"metric_{key.lower()}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        d, f = dcan_agg[key]["mean"], fcn_agg[key]["mean"]
        better = "DCAN" if (d > f) == higher_better else "FCN"
        print(f"  {title:<24} DCAN={d:.4f}  FCN={f:.4f}  -> {better} wins   saved {out_png}")

    # ---- Combined overview: 1x3 bar charts for a single slide ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, title, ylabel, higher_better, paper) in zip(axes, METRICS):
        _bar_panel(ax, key, title, ylabel, higher_better, paper,
                   dcan_agg, fcn_agg, args.paper)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DCAN_COLOR, alpha=0.9),
        plt.Rectangle((0, 0), 1, 1, color=FCN_COLOR, alpha=0.9),
        plt.Line2D([0], [0], ls="--", color=PAPER_COLOR, lw=1.5),
    ]
    fig.legend(handles, ["DCAN (contour-aware)", "FCN baseline (object-only)", "DCAN paper (ref)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"DCAN vs FCN baseline on the held-out test set ({n_imgs} images)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    overview = out_dir / "metrics_overview.png"
    fig.savefig(overview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {overview}")
    print(f"\nFigures written to {out_dir}")


if __name__ == "__main__":
    main()
