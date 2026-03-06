from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


ArrayLike = np.ndarray


def contour_center(contour: ArrayLike) -> Tuple[int, int]:
    """
    Возвращает (cx, cy) для контура.
    Сначала по моментам, при неудаче — по среднему координат.
    """
    M = cv2.moments(contour)
    m00 = M.get("m00", 0.0)
    if m00:
        cx = int(M["m10"] / m00)
        cy = int(M["m01"] / m00)
        return cx, cy

    pts = contour.reshape(-1, 2)
    mean = pts.mean(axis=0)
    cx = int(mean[0])
    cy = int(mean[1])
    return cx, cy


def contour_widest_point(contour: ArrayLike, canvas_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Точка внутри контура, в которой область наиболее «широкая» (максимум distance transform).
    Подходит для размещения номера, чтобы не залезать на границы.
    """
    h, w = canvas_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [np.asarray(contour)], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def rect_fits_in_contour(
    contour: ArrayLike,
    cx: int,
    cy: int,
    tw: int,
    th: int,
) -> bool:
    """Проверяет, что прямоугольник (cx - tw/2, cy - th/2, tw, th) целиком внутри контура."""
    x1 = cx - tw // 2
    y1 = cy - th // 2
    corners = [
        (x1, y1),
        (x1 + tw, y1),
        (x1 + tw, y1 + th),
        (x1, y1 + th),
    ]
    for (px, py) in corners:
        if cv2.pointPolygonTest(contour, (px, py), False) <= 0:
            return False
    return True

