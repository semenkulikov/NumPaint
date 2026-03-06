# -*- coding: utf-8 -*-
"""
Модуль для получения раскрашенной картинки по контурам и палитре.
Используется для сравнения с исходником: области заполняются цветами палитры.
"""
from __future__ import annotations

import numpy as np

from . import render


def render_colorized(
    label_map: np.ndarray,
    palette_bgr: np.ndarray,
) -> np.ndarray:
    """
    Строит изображение, где каждая область залита своим цветом из палитры.
    По сути — постеризованное изображение по карте меток (без контуров и номеров).
    Удобно для сравнения с исходником: видно, как пайплайн разбил картинку на цвета.
    """
    return render.render_quantized_preview(label_map, palette_bgr)
