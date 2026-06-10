"""PyTorch Dataset for GlaS gland segmentation with DCAN targets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dcan.data.augment import get_train_transform, get_val_transform, pad_to_crop
from dcan.data.targets import instance_to_contour_mask, instance_to_object_mask


class GlaSDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        names: list[str] | None = None,
        crop_size: int = 480,
        contour_radius: int = 3,
        is_train: bool = True,
        crops_per_image: int = 4,
    ):
        self.data_dir = Path(data_dir) / split
        self.crop_size = crop_size
        self.contour_radius = contour_radius
        self.is_train = is_train
        self.crops_per_image = crops_per_image if is_train else 1

        img_dir = self.data_dir / "images"
        all_names = sorted(p.stem for p in img_dir.glob("*.png"))
        self.names = names if names is not None else all_names

        self.transform = get_train_transform(crop_size) if is_train else get_val_transform()

    def __len__(self) -> int:
        # Each image yields `crops_per_image` random crops per epoch, so the model
        # sees several augmented 480² views of each of the few labelled images.
        return len(self.names) * self.crops_per_image

    def _load(self, name: str):
        img_path = self.data_dir / "images" / f"{name}.png"
        inst_path = self.data_dir / "instances" / f"{name}.png"
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instance = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        return image, instance

    def __getitem__(self, idx: int):
        # idx runs over names×crops; map back to the source image (crops differ
        # because the random augmentation is re-sampled each call).
        name = self.names[idx % len(self.names)]
        image, instance = self._load(name)

        if self.is_train:
            image, instance = pad_to_crop(image, instance, self.crop_size)

        # Augment the image and the INSTANCE label map together (same geometric
        # transform), so pixels stay aligned. `instance` is carried as a mask
        # (nearest-neighbour, no interpolation) — see get_train_transform.
        transformed = self.transform(image=image, instance=instance)
        image_t = transformed["image"]  # (3, H, W) float, normalized
        instance_t = transformed["instance"]  # (H, W) uint16, still instance ids

        # Build the two training targets from the AUGMENTED instance map, so the
        # object/contour masks match the transformed image exactly.
        obj_mask = instance_to_object_mask(instance_t.numpy() if isinstance(instance_t, torch.Tensor) else instance_t)
        contour_mask = instance_to_contour_mask(
            instance_t.numpy() if isinstance(instance_t, torch.Tensor) else instance_t,
            self.contour_radius,
        )

        # (image, object target, contour target) — long tensors are class indices for CE loss.
        return (
            image_t,
            torch.from_numpy(obj_mask).long(),
            torch.from_numpy(contour_mask).long(),
        )
