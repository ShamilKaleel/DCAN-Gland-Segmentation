"""Export training curves from TensorBoard event files.

Recovers and plots the loss / learning-rate / auxiliary-weight curves logged
during training and regenerates a CSV of the run. This is needed because a
re-launched (and aborted) training run truncated ``runs/dcan/train_log.csv`` --
but the original 100-epoch TensorBoard event file survives, so the full DCAN
training history is recovered here.

Usage:
    uv run python scripts/export_curves.py
    uv run python scripts/export_curves.py --runs runs/dcan runs/fcn_baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCALAR_TAGS = ["loss/train", "loss/val", "lr", "w_a"]


def load_best_accumulator(logs_dir: Path) -> EventAccumulator | None:
    """Return the event accumulator with the most scalar steps in a logs dir.

    A logs dir may contain several event files (from re-launched runs); we keep
    the one carrying the real training history.
    """
    best = None
    for f in sorted(logs_dir.glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])
        if "loss/train" not in scalars:
            continue
        n = len(ea.Scalars("loss/train"))
        if best is None or n > best[0]:
            best = (n, ea)
    return best[1] if best else None


def series(ea: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    evs = ea.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


def plot_run(run_dir: Path) -> dict | None:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        print(f"  [skip] no logs dir in {run_dir}")
        return None
    ea = load_best_accumulator(logs_dir)
    if ea is None:
        print(f"  [skip] no scalar data in {logs_dir}")
        return None

    data = {tag: series(ea, tag) for tag in SCALAR_TAGS}
    name = run_dir.name

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # Loss
    e_tr, v_tr = data["loss/train"]
    e_va, v_va = data["loss/val"]
    axes[0].plot(e_tr, v_tr, label="train", color="C0")
    axes[0].plot(e_va, v_va, label="val", color="C1")
    axes[0].set_title(f"{name}: loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # LR
    e_lr, v_lr = data["lr"]
    axes[1].plot(e_lr, v_lr, color="C2")
    axes[1].set_yscale("log"); axes[1].set_title(f"{name}: learning rate")
    axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3)
    # w_a
    e_wa, v_wa = data["w_a"]
    axes[2].plot(e_wa, v_wa, color="C3")
    axes[2].set_yscale("log"); axes[2].set_title(f"{name}: aux weight w_a")
    axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out_png = run_dir / "training_curves.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}  ({len(e_tr)} epochs)")

    # Regenerate CSV from recovered scalars.
    steps = e_tr
    tr = dict(zip(e_tr, v_tr)); va = dict(zip(e_va, v_va))
    lr = dict(zip(e_lr, v_lr)); wa = dict(zip(e_wa, v_wa))
    csv_path = run_dir / "train_log_recovered.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr", "w_a"])
        for s in steps:
            w.writerow([s, f"{tr.get(s, ''):.6f}", f"{va.get(s, ''):.6f}",
                        f"{lr.get(s, ''):.1e}", f"{wa.get(s, ''):.6f}"])
    print(f"  saved {csv_path}")
    return {"name": name, "loss_val": (e_va, v_va)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["runs/dcan", "runs/fcn_baseline"])
    parser.add_argument("--out", default="runs/val_loss_comparison.png")
    args = parser.parse_args()

    summaries = []
    for run in args.runs:
        print(f"Processing {run}")
        s = plot_run(Path(run))
        if s:
            summaries.append(s)

    # Combined val-loss comparison.
    if len(summaries) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s in summaries:
            e, v = s["loss_val"]
            ax.plot(e, v, label=s["name"])
        ax.set_title("Validation loss comparison"); ax.set_xlabel("epoch")
        ax.set_ylabel("val loss"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
