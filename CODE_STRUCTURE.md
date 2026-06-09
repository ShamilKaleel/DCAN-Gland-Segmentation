# Code Structure & Alignment with the DCAN Paper

This document explains the structure of the implementation and how every part maps to the
reference paper:

> **H. Chen, X. Qi, L. Yu, P.-A. Heng. "DCAN: Deep Contour-Aware Networks for Accurate Gland
> Segmentation." CVPR 2016.** (Winner of the 2015 MICCAI Gland Segmentation Challenge.)

Companion docs: `README.md` (quick usage), `HPC_RUN_GUIDE.md` (how to run), `REPORT.md`
(verification findings & results).

---

## 1. Overview

The project re-implements **DCAN**, a multi-task fully-convolutional network that segments
glands **and** their contours simultaneously, then fuses the two to separate touching glands.
It is trained and evaluated on the **GlaS @ MICCAI'2015 (Warwick-QU)** dataset.

The three goals (from the project proposal) are:
1. **Faithful DCAN implementation** — dual object+contour branches, multi-level contextual
   features, deep supervision, contour-based fusion.
2. **Official GlaS evaluation** — F1 detection, object-level Dice, object-level Hausdorff,
   and rank-sum scoring.
3. **Reproducible benchmark** — DCAN vs. an object-only FCN baseline, reported with 2-fold
   cross-validation and the official Part A / Part B test sets.

---

## 2. Repository tree

```
impl/
├── configs/                  Experiment configuration (YAML)
│   ├── dcan.yaml             DCAN, fixed-split training (HPC defaults)
│   ├── fcn_baseline.yaml     Object-only FCN ablation
│   ├── dcan_all85.yaml       DCAN trained on all 85 images (paper benchmark)
│   └── fcn_all85.yaml        FCN baseline, all 85 images
│
├── scripts/                  Entry points (each is `uv run python scripts/<x>.py`)
│   ├── prepare_data.py       GlaS → PNG + uint16 instance masks + splits
│   ├── train.py              Training loop (multi-task loss, Dice checkpointing)
│   ├── infer.py              Overlap-tile inference → instance maps
│   ├── evaluate.py           Official GlaS metrics on predictions
│   ├── compare.py            DCAN-vs-FCN table + rank-sum
│   ├── visualize.py          Input | GT | FCN | DCAN qualitative panels
│   ├── export_curves.py      TensorBoard → training-curve PNGs + CSV
│   └── run_cv.py             Stratified 2-fold cross-validation driver
│
├── src/dcan/                 Core library
│   ├── models/
│   │   ├── dcan.py           DCAN network (atrous encoder + dual branches)
│   │   └── fcn_baseline.py   Object-only baseline (same encoder, one branch)
│   ├── data/
│   │   ├── ninja_parse.py    Decode DatasetNinja/Supervisely annotations
│   │   ├── targets.py        Instance map → object mask + contour mask
│   │   ├── dataset.py        PyTorch Dataset (crop, augment, build targets)
│   │   └── augment.py        Albumentations train/val transforms
│   ├── losses.py             Multi-task loss + aux-weight scheduler
│   ├── infer.py              Overlap-tile prediction
│   ├── postprocess.py        Fusion rule + morphology + connected components
│   ├── metrics.py            F1 / object-Dice / object-Hausdorff / rank-sum
│   └── utils.py              Seeding, config I/O
│
└── runs/                     Outputs: checkpoints, logs, predictions, figures, results
```

---

## 3. Paper → code map

| Paper concept (section / equation) | Where it lives in the code |
|---|---|
| FCN with **multi-level contextual features** (§2.1, Fig. 2) | `models/dcan.py` → `_MultiLevelBranch` (heads tap 3 encoder depths, sum-fusion) + atrous `_build_encoder` |
| **Deep contour-aware** dual branches: object + contour, shared encoder (§2.2, Fig. 3) | `models/dcan.py` → `DCAN.obj_branch`, `DCAN.contour_branch` |
| **Deep supervision** + auxiliary classifiers C1–C3 with discount weight `w_a` (§2.1, Eq. 1) | `losses.py` → `DCANLoss` (aux terms), `AuxWeightScheduler` |
| **Multi-task loss** = data loss (object + contour) + weighted aux (Eq. 1–2) | `losses.py` → `DCANLoss.forward` |
| **Fusion rule** `m(x)=1 iff p_o ≥ t_o and p_c < t_c`, `t_o=t_c=0.5` (Eq. 3) | `postprocess.py` → `fuse_predictions` |
| **Transfer learning** from a DeepLab atrous backbone (§2.3) | `models/dcan.py` → `_build_encoder` (atrous /8, ImageNet VGG-16) |
| **Contour labels** = connected-component boundaries dilated by disk r3 (§3.2) | `data/targets.py` → `instance_to_contour_mask` |
| **Post-processing**: smooth (disk r3) + fill holes + remove small + label (§2.2) | `postprocess.py` → `postprocess_mask` |
| **Overlap-tile** inference for whole images (§3.2) | `infer.py` → `overlap_tile_predict` |
| **480×480 crops, SGD, lr 1e-3 ↓×10, dropout 0.5** (§3.2) | `configs/*.yaml` + `scripts/train.py` |
| **Augmentation**: translation, rotation, elastic distortion (§3.1) | `data/augment.py` → `get_train_transform` |
| **F1 detection** (50% overlap, one-to-one) (§3.4, Eq. 4) | `metrics.py` → `_match_instances`, `detection_f1` |
| **Object-level Dice** (§3.4, Eq. 5) | `metrics.py` → `object_dice` |
| **Object-level Hausdorff** (§3.4, Eq. 7) | `metrics.py` → `object_hausdorff` |
| **Rank-sum** final ranking (§3.4, Table 4) | `metrics.py` → `rank_sum` (used by `compare.py`) |
| Dataset: GlaS, train 85 / Part A 60 / Part B 20 (§3.1) | `data/ninja_parse.py`, `scripts/prepare_data.py`, `data/splits/` |

---

## 4. Architecture at a glance

```
                          shared atrous VGG-16 encoder (output stride 8)
 Input 3×480×480 ─▶ enc1(/2) ─▶ enc2(/4) ─▶ enc3(/8,256) ─▶ enc4(/8,512) ─▶ enc5(/8) ─▶ bottleneck(/8,1024)
                                                  │              │                              │
                                                  └──────────────┴──────────────┬───────────────┘
                                                                                │  (three depths = three receptive fields)
                              ┌───────────────────────────────────────┐  ┌──────┴───────────────────────────────┐
                              │            OBJECT branch               │  │           CONTOUR branch              │
                              │  head(enc3) head(enc4) head(bneck)     │  │  head(enc3) head(enc4) head(bneck)    │
                              │      └── sum-fusion ──┘ → learned ↑8    │  │      └── sum-fusion ──┘ → learned ↑8  │
                              │     p_o (object prob, full res)        │  │     p_c (contour prob, full res)      │
                              └───────────────────────────┬───────────┘  └───────────────┬──────────────────────┘
                                                          └──────── fusion: m = (p_o≥t_o) & (p_c<t_c) ──────────┘
                                                                                │
                                              smooth (disk r3) → fill holes → remove small → label
                                                                                │
                                                                  instance map  m(x)  (one id per gland)
```

Each **head** = `Conv3×3 → GroupNorm → ReLU → Dropout2d(0.5) → Conv1×1(→2)`. The three head
score maps (all at /8 in the atrous backbone) are **summed** (the paper's sum-fusion of
multi-level predictions) and upsampled to full resolution; each head is also a deeply-supervised
auxiliary output.

---

## 5. Per-file explanation

### 5.1 Models

#### `src/dcan/models/dcan.py` — the DCAN network
**Purpose:** defines the contour-aware network. **Paper alignment:** §2.1–2.3, Fig. 2–3.

- **`_build_encoder(pretrained, output_stride)`** — splits an ImageNet `vgg16_bn` into 5
  stages. For `output_stride=8` (DeepLab-LargeFOV) it makes pool4/pool5 stride-1 and dilates
  block-5 convs (dilation 2), keeping deep features at 1/8 resolution for sharp boundaries
  (paper §2.3 used a DeepLab backbone). `output_stride=32` keeps the original low-memory net.
- **`_Head`** — one per-level classifier (`Conv3×3 → GroupNorm → ReLU → Dropout → Conv1×1`).
- **`_MultiLevelBranch`** — the paper's *multi-level contextual feature* fusion: runs the three
  heads on features from three encoder depths, **sums** their score maps, and upsamples to full
  resolution; returns `(main, [aux1,aux2,aux3])` for deep supervision.
- **`_LearnedUpsampler`** — stacked bilinear-initialised `ConvTranspose2d` (×8).
- **`DCAN`** — shared encoder + `bottleneck` (LargeFOV fc6, dilation 12) + two
  `_MultiLevelBranch` (object & contour). `forward` returns `obj_main, obj_aux, cnt_main,
  cnt_aux`. `freeze_encoder_bn()` keeps pretrained BN in eval for tiny batches.

```python
self.obj_branch     = _MultiLevelBranch((256, 512, 1024), deep_stride, head_dim, num_classes)
self.contour_branch = _MultiLevelBranch((256, 512, 1024), deep_stride, head_dim, num_classes)
# forward → {"obj_main", "obj_aux", "cnt_main", "cnt_aux"}
```

#### `src/dcan/models/fcn_baseline.py` — object-only ablation
**Purpose:** identical encoder + a **single** object branch (no contour). Isolates the
contribution of the contour branch (the paper's *CUMedVision1* ablation, §3.3). Reuses
`_build_encoder` and `_MultiLevelBranch` from `dcan.py`.

### 5.2 Loss

#### `src/dcan/losses.py`
**Purpose:** the multi-task objective. **Paper alignment:** Eq. 1–2.

- **`DCANLoss`** — `L = L_obj + L_cnt + w_a·(Σ aux_obj + Σ aux_cnt)`, each term a
  (class-weighted) cross-entropy. `L_obj/L_cnt` are the data-error terms; the aux terms are the
  deeply-supervised auxiliary classifiers.
- **`FCNBaselineLoss`** — object-only version.
- **`AuxWeightScheduler`** — decays `w_a` ×0.1 every 10 000 iterations down to 1e-3 (paper:
  "discount weights are decreased until marginal values").

```python
total = l_obj + l_cnt + w_a * (aux_obj + aux_cnt)
```

### 5.3 Inference & post-processing

#### `src/dcan/infer.py` — `overlap_tile_predict`
Slides a 480×480 window (stride 240) over the full image, averages softmax probabilities in the
overlaps, and returns `(p_obj, p_cnt)` (paper's overlap-tile strategy, §3.2). `normalize_image`
applies ImageNet mean/std.

#### `src/dcan/postprocess.py`
**Paper alignment:** Eq. 3 + §2.2.
- **`fuse_predictions(p_obj, p_cnt, t_o, t_c)`** — the fusion rule `m = (p_obj≥t_o) & (p_cnt<t_c)`.
- **`postprocess_mask`** — morphological opening (disk r3) → fill holes → connected-component
  labelling → remove small objects → relabel. Produces the final uint16 instance map.
- **`postprocess_fcn`** — object-only variant (no contour subtraction) for the baseline.

### 5.4 Metrics

#### `src/dcan/metrics.py` — official GlaS evaluation
**Paper alignment:** §3.4, Eq. 4–7.
- **`_match_instances`** — **one-to-one** greedy matching at ≥50% GT overlap (each predicted
  object can be a true positive for at most one gland — prevents a merged blob from being
  counted multiple times).
- **`detection_f1`** — precision/recall/F1 from the matching (Eq. 4).
- **`object_dice`** — area-weighted bidirectional object Dice (Eq. 5).
- **`object_hausdorff`** — area-weighted bidirectional object Hausdorff (Eq. 7).
- **`compute_all_metrics`** — convenience wrapper used by `evaluate.py`.
- **`rank_sum`** — sums per-metric ranks across methods (Table 4 ranking), used by `compare.py`.

### 5.5 Data pipeline

#### `src/dcan/data/ninja_parse.py`
Decodes the DatasetNinja/Supervisely annotation JSON: `decode_bitmap` (base64+zlib+PNG →
binary mask), `parse_annotation` (→ `(H,W)` and a uint16 **instance** map, one id per gland),
`get_image_tags` (benign/malignant tags for stratified splits).

#### `src/dcan/data/targets.py`
**Paper alignment:** §3.2 (contour generation).
- **`instance_to_object_mask`** — `instance > 0` (foreground).
- **`instance_to_contour_mask`** — computes each gland's boundary **per instance** (dilation−
  erosion) then dilates with a **disk radius 3**. Because boundaries are per-instance, two
  touching glands get a contour *between* them — exactly what lets the fusion split them.

#### `src/dcan/data/dataset.py` — `GlaSDataset`
PyTorch dataset: loads an image + instance map, (train) reflect-pads and random-crops to
480×480 with augmentation, then builds the **object** and **contour** target masks on the fly.
Returns `(image, obj_target, contour_target)`. `crops_per_image` virtually enlarges the epoch.

#### `src/dcan/data/augment.py`
Albumentations pipelines. Train: random crop, flips, `RandomRotate90`, `ShiftScaleRotate`,
`ElasticTransform` (the paper's distortions) + H&E **stain/illumination** jitter
(`RandomBrightnessContrast`, `HueSaturationValue`) + ImageNet normalisation. Val: normalise only.

### 5.6 Utilities

#### `src/dcan/utils.py`
`seed_everything` (reproducibility), `load_config` / `save_config` (YAML).

### 5.7 Scripts (entry points)

| Script | What it does | Key flags |
|---|---|---|
| `prepare_data.py` | GlaS BMP+JSON → PNG images + uint16 instance masks; generates **stratified 2-fold** and **fixed 70/15** splits | `--dataset --out` |
| `train.py` | Training loop: class weights, multi-task loss, AMP, grad-accum, **checkpoint by val object-Dice** (`evaluate_val_dice` every `eval_every`), append-safe logging, TensorBoard | `--config` |
| `infer.py` | Overlap-tile inference → instance PNGs (`pred/<split>/`); `--dump-probs` saves prob maps; `--names` restricts to a list | `--config --ckpt --split [--names --dump-probs]` |
| `evaluate.py` | Computes F1/Dice/Hausdorff per image + aggregate, writes results JSON | `--pred --gt [--names --out]` |
| `compare.py` | Builds a DCAN-vs-FCN markdown+CSV table and **rank-sum**; `--paper` adds reference rows | `--runs --paper --out` |
| `visualize.py` | Colorised **Input \| GT \| FCN \| DCAN** panels + a montage | `--names --out` (auto-picks busiest val images) |
| `export_curves.py` | Reads TensorBoard events → training/lr/`w_a` curves + recovers a complete CSV | `--runs --out` |
| `run_cv.py` | **2-fold CV driver**: trains/infers/evaluates both folds and pools metrics over all 85 images | `--config --folds` |

### 5.8 Configs

All hyperparameters live in YAML (`utils.load_config`). Key keys: `output_stride` (8 = atrous),
`batch_size`, `accum_steps`, `freeze_encoder_bn`, `eval_every`, `lr/momentum/weight_decay`,
`w_a_*` (aux schedule), `obj_weight_cap/cnt_weight_cap`, `t_o/t_c`, `smooth_radius/min_area`,
`train_split/val_split/run_dir`.
- `dcan.yaml` / `fcn_baseline.yaml` — fixed-split training.
- `dcan_all85.yaml` / `fcn_all85.yaml` — train on all 85 for the **direct paper benchmark**
  (evaluate on Part A / Part B).

---

## 6. Documented deviations from the paper

These are deliberate, justified differences (not defects); each is recorded in `REPORT.md`.

| Deviation | Reason |
|---|---|
| **ImageNet** VGG-16 init vs the paper's **PASCAL-VOC DeepLab** | The original 2015 Caffe DeepLab-VOC weights are not available in PyTorch; the *atrous architecture* (the important part) is reproduced, only the pretraining source differs. |
| Headline = **2-fold CV** over the 85 train images | Matches the project proposal; the paper used the fixed challenge split (also provided via the `all85` configs → Part A/B). |
| **Capped** inverse-frequency class weights | An uncapped contour weight makes the contour branch over-fire and over-split glands. |
| **GroupNorm** in the new heads/bottleneck | Stable under small batches; encoder keeps its pretrained BatchNorm. |
| **Bilinear-initialised** learned upsampling | FCN-style score-map upsampling; with the /8 atrous backbone only an ×8 upsample is needed. |

---

## 7. End-to-end data flow

```
prepare_data.py ─▶ data/{train,test_a,test_b}/{images,instances}/  +  data/splits/
        │
        ▼  (GlaSDataset + targets.py)
   image 480² ──▶ object target (fg/bg)  +  contour target (disk-r3 boundaries)
        │
        ▼  (models/dcan.py)
   DCAN forward ─▶ obj_main/obj_aux, cnt_main/cnt_aux
        │
        ▼  (losses.py)   L = L_obj + L_cnt + w_a·(aux)   ── SGD ──▶ best.pt (by val object-Dice)
        │
        ▼  (infer.py overlap-tile)        p_obj , p_cnt
        │
        ▼  (postprocess.py)  fuse (Eq.3) → morphology → connected components → instance map
        │
        ▼  (metrics.py)  F1 / object-Dice / object-Hausdorff  ─▶ results JSON
        │
        ▼  (compare.py / visualize.py / export_curves.py)  tables · panels · curves
```
