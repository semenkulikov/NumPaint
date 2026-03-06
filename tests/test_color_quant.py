from __future__ import annotations

import numpy as np

from numpaint import color_quant


def test_quantize_lab_kmeans_basic():
    """Квантование двухцветной картинки даёт label_map и палитру из 2 цветов."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :5] = (0, 0, 255)
    img[:, 5:] = (0, 255, 0)

    label_map, palette = color_quant.quantize_lab_kmeans(img, n_colors=2, sample_size=100)

    assert label_map.shape == img.shape[:2]
    assert palette.shape == (2, 3)
    assert np.unique(label_map).size <= 2


def test_choose_k_lab_range():
    """choose_k_lab возвращает k в заданном диапазоне [min_k, max_k]."""
    img = np.full((20, 20, 3), (10, 20, 30), dtype=np.uint8)
    k = color_quant.choose_k_lab(img, min_k=2, max_k=6, target_mse=None, sample_size=100)
    assert 2 <= k <= 6

