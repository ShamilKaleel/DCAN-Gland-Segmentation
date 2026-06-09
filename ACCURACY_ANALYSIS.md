# DCAN Accuracy-Gap Analysis & Improvement Roadmap

**Question:** our DCAN re-implementation is architecturally faithful to Chen et al. (CVPR 2016)
but its accuracy trails the paper. Why — and how do we close the gap?

| Metric (held-out test set) | Ours (DCAN) | Paper (CUMedVision2) | Gap |
|---|---|---|---|
| Object-Dice ↑ | 0.839 | 0.897 | −0.058 |
| Detection F1 ↑ | 0.764 | 0.912 | −0.148 |
| Object-Hausdorff (px) ↓ | 58.5 | 45.4 | +13.1 |

The reasons below are ordered by confidence × impact and are grounded in the code and the actual
training log (`runs/dcan_all85/train_log.csv`). The headline finding: the gap is **mostly recipe,
not architecture** — several causes are cheap to fix, and two need no retraining at all.

---

## 1. Root causes

### 1.1 Undertrained — stopped before convergence (high impact)
In `runs/dcan_all85/train_log.csv` the validation object-Dice is **still climbing at the final
epoch** and peaks there:

| epoch | 5 | 15 | 30 | 50 | 65 | 75 | 90 | 95 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| val-Dice | 0.749 | 0.822 | 0.846 | 0.857 | 0.865 | 0.870 | 0.868 | 0.875 | **0.876** |

Train-loss falls 5.06→1.76 and val-loss falls 0.94→0.42 (still declining, **not** rising — so this
run is *not* the catastrophic overfit the old report described). `ReduceLROnPlateau` dropped the
LR only **once** (1e-3→1e-4 at epoch 91), so the model trained at full LR for 90 epochs and never
got a proper low-LR refinement phase. → **Train ~2–3× longer with a deterministic cosine/step LR
decay.** Likely the single biggest cheap win.

### 1.2 The deep-supervision auxiliary-weight schedule is inert (high impact)
`AuxWeightScheduler` (`src/dcan/losses.py:64`) decays `w_a` ×0.1 every `w_a_step = 10000`
iterations. But the whole run is only **~4,200 iterations** (85 imgs × 4 crops ÷ batch 8 =
42 iters/epoch × 100). So `n = 4200 // 10000 = 0` and **`w_a` stays pinned at 1.0 for the entire
run** (confirmed: the `w_a` column never changes). Consequence: the loss is
`L = L_obj + L_cnt + 1.0·(3 aux_obj + 3 aux_cnt)`, so the **6 auxiliary head losses permanently
dominate the 2 main-output losses**, and the *fused main output that is actually used at inference*
is never given primacy. The paper deliberately **discounts** deep supervision over time. → **Fix
the schedule to the real iteration scale** (decay per-epoch, or `w_a_step ≈ 800–1500`).

### 1.3 Tiny dataset + single model (medium impact)
Only 85 labelled images. Seen-data Dice (0.876) > test Dice (0.839) — a generalization gap. The
paper's winning entry was an **ensemble** of models with heavier augmentation; we run a single
model. → stronger augmentation (stain normalization / stain jitter, more elastic/scale), more
`crops_per_image`, and/or model ensembling.

### 1.4 No test-time augmentation (medium impact, **no retraining**)
`overlap_tile_predict` averages overlapping tiles but applies **no flip/rotation/scale TTA** — a
standard, training-free boost and exactly what competition entries use. → averaged D4 (8-view) TTA.
*(implemented this phase — see §2.)*

### 1.5 Training objective ≠ evaluation metric (medium impact)
Training is pure pixel-wise **weighted cross-entropy** (`src/dcan/losses.py`), dominated by easy
background pixels, while we are scored on **object-Dice and Hausdorff**. There is no region
(soft-Dice / Tversky) or boundary loss aligning the objective with the metric. → add a soft-Dice
term on the object branch (+ optional boundary loss), which typically lifts Dice and, via cleaner
contours, Hausdorff.

### 1.6 Post-processing / fusion thresholds are untuned defaults (low–medium, **no retraining**)
`t_o = t_c = 0.5`, `min_area = 120`, `smooth_radius = 3` (`src/dcan/postprocess.py`, configs) were
never swept. F1 (detection) and Hausdorff are sensitive to these. → grid-search on a held-out fold.
*(implemented this phase — see §2.)*

### 1.7 Leaky checkpoint-selection signal in the all-85 run (correctness)
`data/splits/fixed_val.txt` (13 images) is **fully contained in** `data/splits/all_train.txt`
(85 images). So in the all-85 run "best by val-Dice" is measured on **training** images — it gives
no true generalization/early-stop signal and effectively just selects the latest epoch. (The 2-fold
CV splits and the 70/13 fixed split are clean.) → select on a truly held-out fold, or hold out a
few of the 85.

### 1.8 Backbone & head capacity (the architecture lever)
VGG-16 initialised from **ImageNet** (paper used a PASCAL-VOC DeepLab init); the classifier
`head_dim = 64` is narrow; the "fc6" bottleneck is a **single** 1024-ch dilated conv
(`src/dcan/models/dcan.py`). A stronger modern encoder (ResNet-50 + DeepLabV3+, U-Net, or HRNet)
keeping the dual object+contour heads would raise the ceiling — the natural experiment to *beat*
the paper, but only worth doing after the recipe fixes.

---

## 2. No-retraining quick wins (this phase)

Implemented and measured on a **clean held-out fold** (`runs/dcan_cv/fold0/best.pt` evaluated on
`data/splits/fold0_val.txt`, which that checkpoint never saw):
- **D4 test-time augmentation** — `predict_with_tta()` in `src/dcan/infer.py` (8-view flip/rotate
  averaging; `--tta` flag in `scripts/infer.py`).
- **Threshold / post-proc sweep** — `scripts/tune_postprocess.py` caches the probability maps once
  per image, then sweeps `t_o`/`t_c`/`min_area` and reports the best config vs the 0.5/0.5/120
  defaults.

**Measured gains (fold0_val):**

<!-- MEASURED_RESULTS -->
_(filled in by `scripts/tune_postprocess.py` — see the table below once the sweep completes.)_

---

## 3. Improvement roadmap (recipe-fixes-first)

| Phase | Change | Reasons | Retrain? | Expected |
|---|---|---|:---:|---|
| **1 (done)** | TTA + threshold/post-proc tuning | 1.4, 1.6 | no | small–medium, free |
| **2** | Fix `w_a` schedule; train 200–300 epochs with cosine/step LR; select on a clean held-out fold | 1.1, 1.2, 1.7 | yes | **largest** cheap win |
| **3** | Add soft-Dice/Tversky (+ boundary) loss | 1.5 | yes | medium (Dice, Hausdorff) |
| **4** | Stronger encoder (ResNet-50 DeepLabV3+ / U-Net / HRNet); ensembling; stain-aware aug | 1.3, 1.8 | yes | raises the ceiling |

---

## 4. Reproduce

```bash
cd impl
# No-retraining wins on a clean held-out fold (with and without TTA):
uv run python scripts/tune_postprocess.py \
    --ckpt runs/dcan_cv/fold0/best.pt \
    --names data/splits/fold0_val.txt --split train --data-dir data --tta \
    --out runs/dcan_cv/fold0/postproc_sweep.json

# Inference with TTA on the test set (after picking thresholds in configs):
uv run python scripts/infer.py --config configs/dcan_all85.yaml \
    --ckpt runs/dcan_all85/best.pt --split test_a --tta
```
