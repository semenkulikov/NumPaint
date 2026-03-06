from __future__ import annotations

from typing import Tuple, List

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


ArrayLike = np.ndarray


def _image_to_lab(img_bgr: ArrayLike) -> ArrayLike:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab


def _lab_palette_to_bgr(palette_lab: ArrayLike) -> ArrayLike:
    """
    palette_lab: (K, 3) в пространстве Lab, 0-255
    """
    palette_lab_uint8 = np.clip(palette_lab, 0, 255).astype(np.uint8)
    lab_reshaped = palette_lab_uint8.reshape(1, -1, 3)
    bgr = cv2.cvtColor(lab_reshaped, cv2.COLOR_Lab2BGR)
    return bgr.reshape(-1, 3)


def quantize_lab_kmeans(
    img_bgr: ArrayLike,
    n_colors: int,
    sample_size: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Квантование цветов через MiniBatchKMeans в пространстве Lab.

    :param img_bgr: исходное BGR-изображение (H, W, 3), uint8.
    :param n_colors: количество кластеров.
    :param sample_size: сколько пикселей использовать для обучения.
    :return: (label_map, palette_bgr)
             label_map: (H, W) с индексами [0..n_colors-1]
             palette_bgr: (n_colors, 3) uint8
    """
    lab = _image_to_lab(img_bgr)
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3).astype(np.float32)

    total_pixels = flat.shape[0]
    if total_pixels > sample_size:
        idx = np.random.choice(total_pixels, sample_size, replace=False)
        sample = flat[idx]
    else:
        sample = flat

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=2048,
        n_init="auto",
    )
    kmeans.fit(sample)

    labels = kmeans.predict(flat)
    label_map = labels.reshape(h, w).astype(np.int32)

    palette_lab = kmeans.cluster_centers_
    palette_bgr = _lab_palette_to_bgr(palette_lab)

    return label_map, palette_bgr.astype(np.uint8)


def quantize_slic_then_kmeans(
    img_bgr: ArrayLike,
    n_colors: int,
    n_segments: int = 800,
    compactness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Region-first: SLIC суперпиксели → средний цвет сегмента → KMeans по сегментам.
    Даёт более связные регионы, меньше шума по сравнению с KMeans по пикселям.
    """
    from skimage.segmentation import slic

    rgb = np.asarray(img_bgr, dtype=np.uint8)
    rgb = rgb[:, :, ::-1].copy()
    lab = _image_to_lab(img_bgr)
    h, w = lab.shape[:2]

    segments = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0)
    segments = np.asarray(segments, dtype=np.int32)

    unique_seg = np.unique(segments)
    num_seg = len(unique_seg)
    segment_mean_lab = np.zeros((num_seg, 3), dtype=np.float64)
    for i, sid in enumerate(unique_seg):
        mask = segments == sid
        segment_mean_lab[i] = lab[mask].mean(axis=0)

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=min(256, num_seg),
        n_init="auto",
    )
    kmeans.fit(segment_mean_lab)
    color_labels = kmeans.predict(segment_mean_lab)

    seg_to_color = np.zeros(segments.max() + 1, dtype=np.int32)
    for i, sid in enumerate(unique_seg):
        seg_to_color[sid] = color_labels[i]
    label_map = seg_to_color[segments]

    palette_lab = kmeans.cluster_centers_
    palette_bgr = _lab_palette_to_bgr(palette_lab)
    return label_map, palette_bgr.astype(np.uint8)


def _mse_lab(original_lab: ArrayLike, quantized_lab: ArrayLike) -> float:
    diff = original_lab.astype(np.float32) - quantized_lab.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return mse


def choose_k_lab(
    img_bgr: ArrayLike,
    min_k: int,
    max_k: int,
    target_mse: float | None = None,
    sample_size: int = 20000,
    max_trials: int = 6,
) -> int:
    """
    Грубый подбор числа кластеров k в Lab по MSE.

    Возвращает k из [min_k, max_k], дающий наименьшую ошибку
    (или первое k, удовлетворяющее target_mse, если задан).
    """
    min_k = max(2, int(min_k))
    max_k = max(min_k, int(max_k))

    lab = _image_to_lab(img_bgr)
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3).astype(np.float32)

    total_pixels = flat.shape[0]
    if total_pixels > sample_size:
        idx = np.random.choice(total_pixels, sample_size, replace=False)
        sample = flat[idx]
    else:
        sample = flat

    if max_trials <= 0 or max_k == min_k:
        candidate_ks: List[int] = [min_k]
    else:
        candidate_ks = sorted(
            {int(x) for x in np.linspace(min_k, max_k, num=max_trials)}
        )

    best_k = candidate_ks[0]
    best_mse = float("inf")

    for k in candidate_ks:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=2048,
            n_init="auto",
        )
        kmeans.fit(sample)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        quantized = centers[labels]
        mse = _mse_lab(sample, quantized)

        if target_mse is not None and mse <= target_mse:
            return k

        if mse < best_mse:
            best_mse = mse
            best_k = k

    return int(best_k)


