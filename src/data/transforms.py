"""Image preprocessing + augmentation for HHD-Ethiopic.

Two pipelines:
  * ``base_transform(h, w)``: deterministic resize + ImageNet normalize, used for val/test.
  * ``train_transform(h, w)``: same + conservative augmentation tuned for handwriting.

We deliberately avoid horizontal flips (text is directional) and keep rotation
small (≤ 2°). Augmentations are toggleable via ``aug_level`` to support ablations.
"""

from __future__ import annotations

import random
from typing import Callable

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad_to_aspect(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize preserving aspect ratio, then right-pad to (target_h, target_w) with white."""
    h, w = image.shape[:2]
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)
    if new_w >= target_w:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pad = np.full((target_h, target_w - new_w, 3), 255, dtype=image.dtype)
    return np.concatenate([image, pad], axis=1)


def _aspect_preserving_op(target_h: int, target_w: int) -> A.Lambda:
    def _fn(image: np.ndarray, **kwargs):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return _pad_to_aspect(image, target_h, target_w)

    return A.Lambda(image=_fn, name="aspect_pad")


def base_transform(height: int = 48, width: int = 368, preserve_aspect: bool = True):
    if preserve_aspect:
        return A.Compose([
            _aspect_preserving_op(height, width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height, width, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def train_transform(height: int = 48, width: int = 368, aug_level: str = "medium", preserve_aspect: bool = True):
    """Augmentation pipeline for training.

    aug_level:
      * 'none':   just resize/normalize.
      * 'light':  mild geometric + photometric.
      * 'medium': adds blur, noise, ink-bleed style erosion.
      * 'heavy':  adds paper-texture overlay + stronger erasing.
    """
    ops: list = []
    if preserve_aspect:
        ops.append(_aspect_preserving_op(height, width))
    else:
        ops.append(A.Resize(height, width, interpolation=cv2.INTER_AREA))

    if aug_level != "none":
        ops += [
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
                rotate=(-2.0, 2.0),
                shear={"x": (-2.0, 2.0)},
                cval=255,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ]
    if aug_level in {"medium", "heavy"}:
        ops += [
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), mean=0, p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
            # Coarse dropout simulates ink loss / paper holes
            A.CoarseDropout(max_holes=4, max_height=6, max_width=12, fill_value=255, p=0.2),
        ]
    if aug_level == "heavy":
        ops += [
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.ToGray(p=0.05),
        ]

    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


__all__ = ["base_transform", "train_transform", "IMAGENET_MEAN", "IMAGENET_STD"]
