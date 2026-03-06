from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from numpaint import Config, generate_paint_by_numbers


def test_pipeline_small_image(tmp_path: Path):
    """Полный прогон пайплайна на маленьком изображении: есть outline, preview, палитра."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = (0, 0, 255)
    img[:, 32:] = (0, 255, 0)

    input_path = tmp_path / "test.png"
    cv2.imwrite(str(input_path), img)

    cfg = Config(colors=4, max_size=64)
    result = generate_paint_by_numbers(str(input_path), cfg)

    assert result.label_map.shape == img.shape[:2]
    assert result.palette_bgr.shape[1] == 3
    assert result.outline_image is not None
    assert result.preview_image is not None

