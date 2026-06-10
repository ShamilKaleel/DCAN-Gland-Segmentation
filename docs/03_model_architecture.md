# 3 · Model architecture — image to object & contour maps

← [`02_preprocessing.md`](02_preprocessing.md) · code: `src/dcan/models/dcan.py`

![DCAN architecture](../diagrams/01_model_architecture.png)

`DCAN.forward` is the whole forward pass: **shared encoder → two decoder branches**.

## 3.1 Atrous encoder (downsampling path) — `_build_encoder`

A `vgg16_bn` is split into 5 stages. Normally VGG downsamples ×32; we convert it to a
**DeepLab-LargeFOV atrous** backbone with **output stride 8** so deep features keep high resolution
(sharp gland boundaries):

| stage | source | stride (OS=8) | channels | tapped? |
|-------|--------|---------------|----------|---------|
| enc1 | VGG block1–2 | /2 | 64 | |
| enc2 | VGG block3 | /4 | 128 | |
| enc3 | VGG block4 | **/8** | 256 | ✅ |
| enc4 | VGG block5 convs | **/8** (pool4 stride 1) | 512 | ✅ |
| enc5 | VGG block5 tail | **/8** (dilation 2, pool5 stride 1) | 512 | |

How /8 is enforced: the last two pooling strides are set to 1 and the following convolutions are
**dilated** (rate 2) to keep the same receptive field without shrinking the map.

## 3.2 Bottleneck (fc6) — `DCAN.__init__`

A single 3×3 conv 512→1024 with **dilation 12** (LargeFOV), GroupNorm, ReLU — the high-level
context map, still at /8. Tapped as the third feature.

## 3.3 Multi-level contextual fusion — `_MultiLevelBranch`, `_Head`

Each branch attaches **three heads**, one per tapped depth (enc3 256, enc4 512, bottleneck 1024).
A head is `3×3 conv → GroupNorm → ReLU → Dropout 0.5 → 1×1 conv → 2 classes` (`_Head`), producing a
score map at /8. The three /8 score maps are **summed** (sum-fusion) and then upsampled. Tapping
different depths fuses different receptive fields — the paper's "multi-level contextual features".

> GroupNorm (not BatchNorm) is used in the new layers so the heads are stable at small batch sizes.

## 3.4 Learned upsampling — `_LearnedUpsampler`, `_bilinear_weights`

The fused /8 map is upsampled ×8 by a stack of three ×2 `ConvTranspose2d` layers, each
**initialised to bilinear interpolation** (`_bilinear_weights`) — a clean starting point the network
refines. A final bilinear `interpolate` snaps to the exact input size if needed.

## 3.5 Two branches + deep supervision

`obj_branch` and `contour_branch` are identical modules with separate weights, fed the same three
features. Each returns `(main, aux)`: `main` is the fused full-res logits (used at inference); `aux`
is the list of per-head maps, upsampled, used only for **deep supervision** in training (next doc).

`DCAN.forward` returns `{obj_main, obj_aux, cnt_main, cnt_aux}`. The **FCN baseline**
(`fcn_baseline.py`) is the same minus `contour_branch` → `{obj_main, obj_aux}`.

Next → [`04_training.md`](04_training.md)
