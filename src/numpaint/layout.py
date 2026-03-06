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

