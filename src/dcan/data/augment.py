"""Albumentations augmentation pipeline matching DCAN paper."""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(crop_size: int = 480) -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=45,
                border_mode=0, p=0.5,
            ),
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, p=0.5,
                border_mode=0,
            ),
            # H&E stain / illumination variation (image-only) to improve
            # generalization across slides -- within the paper's "distortion" aug.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"instance": "mask"},
    )


def get_val_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"instance": "mask"},
    )


def pad_to_crop(image: np.ndarray, instance: np.ndarray, crop_size: int):
    """Reflect-pad image and mask so they are at least crop_size in each dim."""
    h, w = image.shape[:2]
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
        instance = np.pad(instance, ((0, pad_h), (0, pad_w)), mode="reflect")
    return image, instance
