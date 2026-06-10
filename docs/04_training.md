# 4 · Training — loss, schedule, and the loop

← [`03_model_architecture.md`](03_model_architecture.md) · code: `src/dcan/losses.py`, `scripts/train.py`

## 4.1 The loss — `losses.py` (`DCANLoss`)

```
L = L_obj + L_cnt + w_a · ( Σ aux_obj + Σ aux_cnt )
```
- **Main terms** `L_obj`, `L_cnt`: weighted cross-entropy on the **fused** full-resolution object
  and contour logits — the outputs actually used at inference. Separate class weights per branch.
- **Auxiliary terms** (deep supervision): the same CE applied to **every per-head score map**,
  summed. Each encoder depth therefore gets a direct gradient, which stabilises early training.
- **`w_a`** discounts the auxiliary terms so the fused main output dominates **later** in training.

### Class weights — `compute_class_weights` (in `train.py`)
Foreground (and especially contour) pixels are rare, so we use capped inverse-frequency weights
(background normalised to 1.0; contour cap 5, object cap 10). The cap is important: an uncapped
contour weight makes the contour branch over-fire and over-split glands.

### ⚠️ The `w_a` schedule caveat — `AuxWeightScheduler`
`w_a` decays ×0.1 every `step_iters` **iterations**. The default 10000 assumes a long run; on this
small dataset (~40 iters/epoch) a 100-epoch run is only ~4000 iters, so `w_a` never actually decays
and the auxiliary losses dominate the whole run. Fixing this (decay on the real scale) is a key item
in [`../ACCURACY_ANALYSIS.md`](../ACCURACY_ANALYSIS.md) §1.2.

## 4.2 Optimizer & schedule — `train.py`
SGD (lr 1e-3, momentum 0.9, weight decay 5e-4), AMP mixed precision, `ReduceLROnPlateau` on
validation loss, optional gradient accumulation (`accum_steps`) to simulate a larger batch.

## 4.3 The loop (input → optimized weights)
Per epoch:
1. **Train**: for each batch `(image, obj_target, cnt_target)` → forward → `DCANLoss` → scaled
   backward → SGD step (every `accum_steps` micro-batches).
2. **Validate**: mean validation loss (logged) **and**, every `eval_every` epochs, the full
   inference+post-processing pipeline on the val fold to get mean **object-Dice**.
3. **Checkpoint**: `last.pt` every epoch; `best.pt` whenever val **object-Dice** improves.

> Selection uses object-Dice, **not** val cross-entropy: CE bottoms out / overfits early and would
> pick an under-segmenting model, while object-Dice tracks the metric we report.

Run it: `uv run python scripts/train.py --config configs/dcan_all85.yaml` (configs hold all
hyper-parameters; `run_cv.py` does the 2-fold cross-validation).

Next → [`05_inference_postprocessing.md`](05_inference_postprocessing.md)
