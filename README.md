# DCAN Re-Implementation for Colon Gland Segmentation

Re-implementation of **DCAN: Deep Contour-Aware Networks** (Chen et al., CVPR 2016)
for gland instance segmentation on the **GlaS@MICCAI'2015** dataset, with the
official GlaS evaluation metrics and an object-only FCN baseline (CUMedVision1-style
ablation).

> **Verification & results:** see [`REPORT.md`](REPORT.md) for the full fidelity
> check against the paper, the mismatches we found and fixed, and the results with
> interpretation. This README is the usage guide.

## What's implemented

- **Multi-level contour-aware network** — shared VGG-16 encoder + two upsampling
  branches (gland **object** + gland **contour**), each fusing predictions from
  three encoder depths (different receptive fields), with deep supervision.
- **Fusion + post-processing** — `m(x)=1 iff p_o ≥ t_o and p_c < t_c` (paper Eq. 3),
  then smoothing, hole filling, small-area removal, connected-component labelling.
- **Official GlaS metrics** — detection F1 (one-to-one matching), object-level Dice,
  object-level Hausdorff, and rank-sum scoring.
- **FCN baseline** — identical encoder + single object branch, for ablation.

**Backbone:** VGG-16 converted to a **DeepLab-LargeFOV atrous encoder (output stride 8)**
— the last two max-pools are made stride-1 and the following convolutions are dilated, so
deep features stay at 1/8 resolution for sharp boundaries (`output_stride: 8`). The
ImageNet-pretrained VGG-16-BN weights are a defensible stand-in for the paper's
PASCAL-VOC-pretrained DeepLab initialisation. Set `output_stride: 32` for the legacy
low-memory backbone.

## Architecture (multi-level contextual fusion)

Mirrors the paper's Fig. 2. The shared encoder produces features at increasing
depth; each branch attaches a classifier head to three of them and **sums** the
upsampled score maps:

```
Input ─▶ enc1(/2) ─▶ enc2(/4) ─▶ enc3(/8,256) ─▶ enc4(/16,512) ─▶ enc5(/32) ─▶ bottleneck(/32,1024)
                                     │                  │                              │
                                  head U1            head U2                        head U3      (per branch)
                                     └──────── bilinear-upsample to input size ──────┘
                                                       ▼  sum-fusion
                                          object logits     contour logits
                                                       ▼  softmax + fusion rule + morphology
                                                 instance label map  m(x)
```

Each head: `Conv3×3 → GroupNorm → ReLU → Dropout2d(0.5) → Conv1×1`. New layers use
GroupNorm (stable at batch size 1); the pretrained encoder's BatchNorm is frozen
during training. The object branch is in `models/dcan.py`; the FCN baseline keeps
only that branch (`models/fcn_baseline.py`).

## Setup

```bash
cd impl
uv sync
```

**GPU note (HPC):** on a Blackwell GPU (e.g. **RTX 5090**, sm_120) you need **PyTorch ≥ 2.7
with CUDA 12.8** (or newer) — older builds fail with "no kernel image". Verify with
`python -c "import torch; print(torch.cuda.get_device_name(0))"`.

## Quick start

### 1. Prepare data
Convert the DatasetNinja GlaS release into PNGs and generate splits:
```bash
uv run python scripts/prepare_data.py --dataset ../Dataset --out data
```
Creates `data/{train,test_a,test_b}/images/` (RGB PNGs),
`.../instances/` (uint16 instance masks), and `data/splits/` (fold + fixed splits).

### 2. Train
```bash
uv run python scripts/train.py --config configs/dcan.yaml
uv run python scripts/train.py --config configs/fcn_baseline.yaml   # baseline (optional)
```
Logging is **append-safe**: an existing non-empty `train_log.csv` is rotated to
`train_log.prev.<ts>.csv` instead of being overwritten.

### 3. Inference
```bash
uv run python scripts/infer.py --config configs/dcan.yaml --ckpt runs/dcan/best.pt --split train
# add --dump-probs to also save p_obj/p_cnt .npy maps for offline threshold sweeping
```

### 4. Evaluate (official metrics)
```bash
uv run python scripts/evaluate.py \
    --pred runs/dcan/pred/train --gt data/train/instances \
    --names data/splits/fixed_val.txt --out runs/dcan/results_val.json
```

### 5. Demonstration artifacts
```bash
uv run python scripts/compare.py --paper        # metrics + rank-sum table (runs/comparison.{md,csv})
uv run python scripts/visualize.py              # Input|GT|FCN|DCAN panels (runs/figures/)
uv run python scripts/export_curves.py          # training curves + recovered CSV
```

### 6. High-accuracy HPC run (headline: 2-fold cross-validation)
```bash
# Stratified 2-fold CV (trains + evaluates both folds, pools over all 85 images)
uv run python scripts/run_cv.py --config configs/dcan.yaml
uv run python scripts/run_cv.py --config configs/fcn_baseline.yaml
uv run python scripts/compare.py --paper \
    --runs DCAN=runs/dcan_cv/results_cv.json FCN=runs/fcn_baseline_cv/results_cv.json \
    --out runs/comparison_cv

# Direct paper benchmark: train all 85, evaluate official Part A / Part B
uv run python scripts/train.py --config configs/dcan_all85.yaml
uv run python scripts/infer.py    --config configs/dcan_all85.yaml --ckpt runs/dcan_all85/best.pt --split test_a
uv run python scripts/evaluate.py --pred runs/dcan_all85/pred/test_a --gt data/test_a/instances --out runs/dcan_all85/results_test_a.json
# repeat with --split test_b / data/test_b/instances ; same for configs/fcn_all85.yaml
```

### 6. High-accuracy HPC run (headline: 2-fold cross-validation)
```bash
# Stratified 2-fold CV (trains + evaluates both folds, pools over all 85 images)
uv run python scripts/run_cv.py --config configs/dcan.yaml
uv run python scripts/run_cv.py --config configs/fcn_baseline.yaml
uv run python scripts/compare.py --paper \
    --runs DCAN=runs/dcan_cv/results_cv.json FCN=runs/fcn_baseline_cv/results_cv.json \
    --out runs/comparison_cv

# Direct paper benchmark: train all 85, evaluate official Part A / Part B
uv run python scripts/train.py --config configs/dcan_all85.yaml
uv run python scripts/infer.py    --config configs/dcan_all85.yaml --ckpt runs/dcan_all85/best.pt --split test_a
uv run python scripts/evaluate.py --pred runs/dcan_all85/pred/test_a --gt data/test_a/instances --out runs/dcan_all85/results_test_a.json
# repeat with --split test_b / data/test_b/instances ; same for configs/fcn_all85.yaml
```

## Results (held-out validation, corrected metric)

Fixed 71/14 train/val split. Rank-sum over our two methods (lower = better);
paper rows are the hidden-test reference (different split, context only).

| Method | F1 ↑ | Object-Dice ↑ | Object-Hausdorff ↓ | Rank-sum ↓ |
|---|---|---|---|---|
| **DCAN** | 0.476 | **0.637** | **143.3** | **4 (best)** |
| FCN baseline | 0.499 | 0.554 | 272.6 | 5 |
| *DCAN paper (Part A)* | *0.912* | *0.897* | *45.4* | *—* |
| *DCAN paper (Part B)* | *0.716* | *0.781* | *160.3* | *—* |

DCAN wins overall and is markedly better on boundary quality (Dice, Hausdorff) —
the contour-awareness benefit. See `REPORT.md` for the before/after of the metric
fix and the overfitting analysis. Numbers will be refreshed after retraining with
the fixed architecture.

## Configuration

Hyperparameters live in `configs/dcan.yaml` and `configs/fcn_baseline.yaml`.

| Parameter | DCAN | Description |
|---|---|---|
| output_stride | 8 | 8 = DeepLab atrous backbone (HPC); 32 = legacy 2 GB backbone |
| crop_size | 480 | random crop for training |
| batch_size | 8 | real batch on the 32 GB GPU (raise if VRAM allows) |
| accum_steps | 1 | gradient accumulation (1 = off; was 4 on the 2 GB GPU) |
| freeze_encoder_bn | false | real BatchNorm trains fine at batch ≥ 8 |
| eval_every | 5 | run val-Dice checkpoint selection every N epochs |
| max_epochs | 100 | maximum training epochs |
| lr / momentum / weight_decay | 1e-3 / 0.9 / 5e-4 | SGD; lr ↓×0.1 on plateau → 1e-7 |
| contour_radius | 3 | dilation radius for contour targets |
| obj_weight_cap / cnt_weight_cap | 10 / 5 | caps on inverse-frequency class weights |
| w_a_init / w_a_decay / w_a_step / w_a_min | 1.0 / 0.1 / 10000 / 1e-3 | aux-loss discount schedule |
| t_o, t_c | 0.5, 0.5 | fusion thresholds (object, contour) |
| smooth_radius / min_area | 3 / 120 | morphological opening radius / min instance size |

## Metrics

Official GlaS challenge metrics (`src/dcan/metrics.py`):
- **Detection F1** — a prediction overlapping ≥50% of a GT gland's area is a TP;
  matching is **one-to-one** so a merged blob cannot count as multiple TPs.
- **Object-level Dice** — area-weighted, bidirectional (paper Eq. 5).
- **Object-level Hausdorff** — area-weighted, bidirectional (paper Eq. 7).
- **Rank-sum** — sum of per-metric ranks across methods (lower = better);
  applied by `scripts/compare.py`.

## Project structure

```
impl/
├── configs/              YAML configs (dcan, fcn_baseline)
├── scripts/
│   ├── prepare_data.py   dataset conversion + splits
│   ├── train.py          training (BN-freeze, grad-accum, capped weights, safe logging)
│   ├── infer.py          overlap-tile inference (+ --dump-probs)
│   ├── evaluate.py       official GlaS metrics
│   ├── compare.py        metrics + rank-sum comparison table
│   ├── visualize.py      Input|GT|FCN|DCAN qualitative panels
│   ├── export_curves.py  TensorBoard → training curves + recovered CSV
│   └── run_cv.py         stratified 2-fold cross-validation driver
├── src/dcan/
│   ├── data/             dataset parsing, contour/object targets, augmentation
│   ├── models/           DCAN + FCN baseline (multi-level contextual fusion)
│   ├── losses.py         multi-task loss + auxiliary weight scheduler
│   ├── infer.py          overlap-tile prediction
│   ├── postprocess.py    fusion rule + morphology
│   ├── metrics.py        official GlaS metrics + rank-sum
│   └── utils.py          seeding, config I/O
├── REPORT.md             verification, mismatches, results interpretation
└── runs/                 checkpoints, logs, predictions, figures, results
```

## Reference

Hao Chen, Xiaojuan Qi, Lequan Yu, Pheng-Ann Heng. *DCAN: Deep Contour-Aware
Networks for Accurate Gland Segmentation.* CVPR 2016. Winner of the 2015 MICCAI
Gland Segmentation Challenge.
