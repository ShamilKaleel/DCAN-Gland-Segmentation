# 2 · Preprocessing — from image + GT to training targets

← [`01_overview.md`](01_overview.md) · code: `src/dcan/data/`

The model is trained on three things per sample: the **image**, an **object target**, and a
**contour target**. The object/contour targets are *derived* from the ground-truth instance map.

## 2.1 The two targets — `data/targets.py`

Given the GT instance label map (each gland a distinct integer id):

- **Object mask** = `instance > 0` — a simple binary "gland vs background"
  (`instance_to_object_mask`).
- **Contour mask** = the **per-instance** boundary, thickened. For each gland id separately we take
  a 1-px ring (`dilate(1) − erode(1)`) and dilate it by a disk of radius 3, then union all rings
  (`instance_to_contour_mask`).

> **Why per-instance and not on the merged foreground?** Where two glands touch, computing each
> gland's boundary separately means the shared seam is labelled as contour from *both* sides. A
> boundary on the merged foreground would skip that seam, and the glands could never be split — this
> is the single detail that makes the whole contour-aware idea work.

## 2.2 The dataset — `data/dataset.py`

`GlaSDataset.__getitem__`:
1. Loads the image + instance map for one of the split's names.
2. (train) reflect-pads up to the crop size (`pad_to_crop`).
3. Applies the augmentation pipeline to **image and instance map together** (so they stay aligned).
4. Builds the object and contour targets **from the augmented instance map** (so targets match the
   transformed image exactly).
5. Returns `(image_tensor, object_target, contour_target)` — the targets are `long` class indices
   for cross-entropy.

Each image is sampled `crops_per_image` times per epoch (`__len__ = len(names) × crops_per_image`),
so the model sees several random 480² views of each of the few labelled images.

## 2.3 Augmentation — `data/augment.py`

Heavy augmentation matters because there are only ~85 labelled images. Two groups:
- **Geometric** (crop, H/V flip, rot90, shift-scale-rotate, elastic) — applied to **both** image
  and mask. `additional_targets={"instance": "mask"}` makes Albumentations resize the instance map
  with nearest-neighbour so ids are preserved.
- **Photometric** (brightness/contrast, hue/sat — H&E stain variation) and **Normalize** (ImageNet
  mean/std, to match the pretrained encoder) — **image only** (a mask has no colour).

`get_val_transform` keeps only Normalize + ToTensor (no random ops) for deterministic validation.

Next → [`03_model_architecture.md`](03_model_architecture.md)
