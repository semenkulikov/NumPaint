# -*- coding: utf-8 -*-
"""
Векторизация границ через potrace (potracer — чистый Python): маска → кривые Безье → гладкие контуры.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

import logging

try:
    import potrace
    _POTRACE_AVAILABLE = True
except ImportError:
    potrace = None
    _POTRACE_AVAILABLE = False

_log = logging.getLogger(__name__)


def is_potrace_available() -> bool:
    return _POTRACE_AVAILABLE


def _curve_to_points_potracer(curve, n_steps: int = 16) -> np.ndarray:
    """
    Семплирование кривой из potracer (Curve = list of segments): Безье и углы → массив точек Nx2.
    """
    points: List[Tuple[float, float]] = []
    start = curve.start_point
    if start is None:
        return np.zeros((0, 2), dtype=np.float32)
    points.append((start.x, start.y))
    for seg in curve.segments:
        end = seg.end_point
        ex, ey = end.x, end.y
        if seg.is_corner:
            points.append((ex, ey))
        else:
            c1, c2 = seg.c1, seg.c2
            for i in range(1, n_steps + 1):
                t = i / n_steps
                s = 1.0 - t
                x = s * s * s * points[-1][0] + 3 * s * s * t * c1.x + 3 * s * t * t * c2.x + t * t * t * ex
                y = s * s * s * points[-1][1] + 3 * s * s * t * c1.y + 3 * s * t * t * c2.y + t * t * t * ey
                points.append((x, y))
    return np.array(points, dtype=np.float32)


def trace_boundary_potrace(
    boundary_mask: np.ndarray,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
) -> Optional[List[np.ndarray]]:
    """
    Трассирует бинарную маску границ через potrace (potracer), возвращает список кривых (каждая — массив точек Nx2).
    boundary_mask: (H, W), 0 или 255 (255 = граница).
    """
    if not _POTRACE_AVAILABLE or boundary_mask.size == 0:
        return None
    try:
        # potracer: Bitmap инвертирует; чтобы трассировать границу (255), передаём 255 - mask
        data = np.asarray(255 - boundary_mask, dtype=np.uint8)
        bmp = potrace.Bitmap(data)
        path = bmp.trace(
            turdsize=turdsize,
            alphamax=alphamax,
            opttolerance=opttolerance,
        )
        curves_pts: List[np.ndarray] = []
        for curve in path.curves:
            if hasattr(curve, "tesselate"):
                pts = curve.tesselate()
                if pts is not None and len(pts) >= 2:
                    pts = np.asarray(pts, dtype=np.float32)
                    if pts.ndim == 2 and pts.shape[1] == 2:
                        curves_pts.append(pts)
                        continue
            pts = _curve_to_points_potracer(curve)
            if len(pts) >= 2:
                curves_pts.append(pts)
        return curves_pts if curves_pts else None
    except Exception as e:
        _log.debug("potrace trace failed: %s", e)
        return None


def rasterize_potrace_curves(
    curves: List[np.ndarray],
    h: int,
    w: int,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Рисует список кривых (каждая Nx2 float) на белом холсте (H, W, 3) с заданной толщиной линии.
    """
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    for pts in curves:
        if pts is None or len(pts) < 2:
            continue
        pts_img = np.asarray(pts, dtype=np.float32)
        if pts_img.ndim != 2 or pts_img.shape[1] != 2:
            continue
        if pts_img[:, 0].max() <= 1.0 and pts_img[:, 1].max() <= 1.0:
            pts_img = pts_img.copy()
            pts_img[:, 0] *= w
            pts_img[:, 1] *= h
        # potracer: (0,0)=top-left, y вниз — как в OpenCV, не отражаем по Y
        pts_int = np.clip(np.round(pts_img), 0, [w - 1, h - 1]).astype(np.int32)
        cv2.polylines(
            out,
            [pts_int.reshape((-1, 1, 2))],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return out
