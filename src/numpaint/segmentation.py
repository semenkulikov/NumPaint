from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


ArrayLike = np.ndarray


def mode_filter(label_map: np.ndarray, k: int) -> np.ndarray:
    """
    Каждый пиксель заменяется на наиболее частую метку в окне k×k (мода).
    Для категориальной карты меток — корректное сглаживание без артефактов median.
    """
    h, w = label_map.shape[:2]
    if k < 3 or k % 2 == 0:
        k = 3 if k < 3 else k + 1
    pad = k // 2
    lab = np.asarray(label_map, dtype=np.int32)
    padded = np.pad(lab, pad, mode="edge")
    out = np.zeros_like(lab)
    max_label = int(lab.max())
    for y in range(h):
        for x in range(w):
            window = padded[y : y + k, x : x + k].ravel()
            counts = np.bincount(window, minlength=max_label + 2)
            out[y, x] = np.argmax(counts)
    return out


def build_region_contours(
    label_map: np.ndarray,
    min_region_area: int,
    morph_kernel: int = 3,
    contour_simplify_eps: float = 0.01,
    merge_small_regions: bool = True,
    palette_bgr: Optional[np.ndarray] = None,
) -> Dict[int, List[np.ndarray]]:
    """
    Строит сглаженные контуры для каждой метки цвета.

    :param label_map: (H, W) int32, индексы цветов. Может быть модифицирован при merge_small_regions.
    :param min_region_area: минимальная площадь контура (в пикселях).
    :param morph_kernel: размер структурного элемента для морфологии (0/1 = выкл).
    :param contour_simplify_eps: доля от периметра для approxPolyDP (0 = без упрощения).
    :param merge_small_regions: если True, мелкие области перекрашиваются в цвет соседа по границе.
    :param palette_bgr: не используется, зарезервировано под учёт близости цвета при слиянии.
    :return: словарь color_index -> список контуров (np.ndarray Nx1x2).
    """
    h, w = label_map.shape[:2]
    unique_labels = np.unique(label_map.reshape(-1))

    kernel = None
    if morph_kernel and morph_kernel > 1:
        k = int(morph_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # При необходимости сливаем мелкие области с соседями (без морфологии по маскам)
    if merge_small_regions and min_region_area > 0:
        for color_idx in unique_labels:
            mask = (label_map == int(color_idx)).astype(np.uint8) * 255
            if not np.any(mask):
                continue

            contours, hierarchy = cv2.findContours(
                mask,
                mode=cv2.RETR_CCOMP,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )
            if hierarchy is None:
                hierarchy = np.zeros((1, 0, 4), dtype=np.int32)

            for i, cnt in enumerate(contours):
                if hierarchy.shape[1] > i and hierarchy[0][i][3] >= 0:
                    continue
                area = cv2.contourArea(cnt)
                if area >= float(min_region_area):
                    continue

                small_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(small_mask, [cnt], contourIdx=-1, color=255, thickness=-1)

                dilate_kernel = kernel if kernel is not None else np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(small_mask, dilate_kernel, iterations=1)
                border = (dilated == 255) & (small_mask == 0)
                neighbor_labels = label_map[border]
                neighbor_labels = neighbor_labels[neighbor_labels != int(color_idx)]
                if neighbor_labels.size == 0:
                    continue

                # Присоединяем к наиболее частому соседу
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                target_label = int(unique[np.argmax(counts)])
                label_map[small_mask == 255] = target_label

        # пересчёт уникальных меток после слияния
        unique_labels = np.unique(label_map.reshape(-1))

    contours_by_color: Dict[int, List[np.ndarray]] = {}

    for color_idx in unique_labels:
        mask = (label_map == int(color_idx)).astype(np.uint8) * 255
        if not np.any(mask):
            continue

        contours, _ = cv2.findContours(
            mask,
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        good_contours: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < float(min_region_area):
                continue

            if contour_simplify_eps > 0.0:
                peri = cv2.arcLength(cnt, closed=True)
                eps = float(contour_simplify_eps) * peri
                if eps > 0:
                    cnt = cv2.approxPolyDP(cnt, eps, closed=True)

            if cnt.shape[0] >= 3:
                good_contours.append(cnt)

        if good_contours:
            contours_by_color[int(color_idx)] = good_contours

    return contours_by_color


def scale_contours(
    contours_by_color: Dict[int, List[np.ndarray]],
    scale_x: float,
    scale_y: float,
) -> Dict[int, List[np.ndarray]]:
    """
    Масштабирует координаты контуров (sx, sy). Формат контуров (N, 1, 2) сохраняется.
    """
    out: Dict[int, List[np.ndarray]] = {}
    scale = np.array([[scale_x, scale_y]], dtype=np.float64)
    for color_idx, contours in contours_by_color.items():
        scaled = []
        for cnt in contours:
            c = cnt.astype(np.float64) * scale
            scaled.append(np.round(c).astype(np.int32))
        out[color_idx] = scaled
    return out


def scale_contour_list(
    contours: List[np.ndarray],
    scale_x: float,
    scale_y: float,
) -> List[np.ndarray]:
    """Масштабирует список контуров (для границ)."""
    scale = np.array([[scale_x, scale_y]], dtype=np.float64)
    return [np.round(c.astype(np.float64) * scale).astype(np.int32) for c in contours]

