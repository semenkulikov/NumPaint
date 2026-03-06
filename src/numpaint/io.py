from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


ArrayLike = np.ndarray


def load_image(source: Union[str, Path, ArrayLike]) -> ArrayLike:
    """
    Загружает изображение в формате BGR (как в OpenCV) с dtype uint8.
    """
    if isinstance(source, np.ndarray):
        img = source
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    path = str(Path(source))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def resize_for_processing(img_bgr: ArrayLike, max_size: int) -> ArrayLike:
    """
    Уменьшает изображение так, чтобы максимальная сторона была <= max_size.
    При небольших исходниках ничего не делает.
    """
    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_size:
        return img_bgr

    scale = max_size / float(max_dim)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def resize_label_map_to_original(
    label_map_small: np.ndarray,
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """
    Масштабирует карту меток (label_map) до исходного размера (H, W).
    Использует ближайшего соседа, чтобы сохранить индексы цветов.
    """
    orig_h, orig_w = orig_shape
    resized = cv2.resize(
        label_map_small.astype(np.int32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int32)

