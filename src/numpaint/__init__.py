from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from loguru import logger

from . import color_quant
from . import io
from . import render
from . import segmentation


ArrayLike = np.ndarray


@dataclass
class Config:
    colors: int = 24
    max_size: int = 1024
    min_region_area: int | None = None
    morph_kernel: int = 5
    contour_simplify_eps: float = 0.01
    draw_preview: bool = True
    draw_outline: bool = True
    auto_colors: bool = False
    min_colors: int = 8
    max_colors: int = 40
    target_mse: float | None = None
    merge_small_regions: bool = True
    # Псевдо-векторизация: чёткие контуры, меньше шума
    smooth_input: bool = True
    smooth_label_map_kernel: int = 0
    boundary_smooth_eps: float = 2.0
    use_slic: bool = False
    slic_n_segments: int = 800
    slic_compactness: float = 10.0


@dataclass
class ResultBundle:
    outline_image: Optional[np.ndarray]
    preview_image: Optional[np.ndarray]
    palette_bgr: np.ndarray
    palette_hex: list[str]
    label_map: np.ndarray
    contours_by_color: Optional[dict[int, list[np.ndarray]]]


def _auto_min_region_area(h: int, w: int) -> int:
    """Порог площади области: меньше значение — сохраняются тонкие детали (ноги, усики)."""
    area = h * w
    rough = area // 600
    return int(max(100, min(rough, 2500)))


def generate_paint_by_numbers(
    image: Union[str, Path, ArrayLike],
    config: Optional[Config] = None,
) -> ResultBundle:
    """
    Основной API: по изображению строит раскраску по номерам и возвращает ResultBundle
    (контур с номерами, превью, палитра, карта меток).
    """
    if config is None:
        config = Config()

    bgr_orig = io.load_image(image)
    orig_h, orig_w = bgr_orig.shape[:2]
    logger.info("Загружено изображение {}x{}", orig_w, orig_h)

    bgr_work = io.resize_for_processing(bgr_orig, config.max_size)
    work_h, work_w = bgr_work.shape[:2]
    logger.info("Рабочее разрешение: {}x{}", work_w, work_h)

    if config.smooth_input:
        bgr_work = cv2.bilateralFilter(bgr_work, d=9, sigmaColor=75, sigmaSpace=75)
        logger.debug("Применён bilateralFilter к входу (сглаживание без размытия границ)")

    if config.auto_colors:
        k = color_quant.choose_k_lab(
            bgr_work,
            min_k=config.min_colors,
            max_k=config.max_colors,
            target_mse=config.target_mse,
        )
        logger.info("Автовыбор числа цветов: {}", k)
    else:
        k = config.colors
        logger.info("Используется фиксированное число цветов: {}", k)

    t0 = time.perf_counter()
    if config.use_slic:
        label_map_small, palette_bgr = color_quant.quantize_slic_then_kmeans(
            bgr_work,
            n_colors=k,
            n_segments=config.slic_n_segments,
            compactness=config.slic_compactness,
        )
        logger.info("Квантование SLIC+KMeans заняло {:.2f} с", time.perf_counter() - t0)
    else:
        label_map_small, palette_bgr = color_quant.quantize_lab_kmeans(
            bgr_work,
            n_colors=k,
        )
        logger.info("Квантование заняло {:.2f} с", time.perf_counter() - t0)

    n_unique_before = len(np.unique(label_map_small.ravel()))
    if config.morph_kernel and config.morph_kernel >= 3:
        k = config.morph_kernel if config.morph_kernel % 2 else config.morph_kernel + 1
        label_map_small = segmentation.mode_filter(label_map_small, k)
        logger.debug("Применён mode_filter к карте меток (окно {}x{}), уникальных меток до/после: {}/{}", k, k, n_unique_before, len(np.unique(label_map_small.ravel())))

    palette_hex = render.build_palette_hex(palette_bgr)

    if config.min_region_area is None:
        min_region_area_full = _auto_min_region_area(orig_h, orig_w)
        logger.info("Автовыбор min_region_area: {}", min_region_area_full)
    else:
        min_region_area_full = int(config.min_region_area)
        logger.info("Используется фиксированный min_region_area: {}", min_region_area_full)

    # Минимальная площадь в пикселях рабочего разрешения (пропорционально)
    work_pixels = work_h * work_w
    orig_pixels = orig_h * orig_w
    min_region_area_work = max(20, int(min_region_area_full * work_pixels / orig_pixels))

    outline_image: Optional[np.ndarray] = None
    preview_image: Optional[np.ndarray] = None
    contours_by_color: Optional[dict[int, list[np.ndarray]]] = None

    if config.draw_outline or config.draw_preview:
        logger.debug(
            "Параметры сегментации: min_region_area_work={}, morph_kernel={}, contour_simplify_eps={}, merge_small={}",
            min_region_area_work,
            config.morph_kernel,
            config.contour_simplify_eps,
            config.merge_small_regions,
        )
        t0 = time.perf_counter()
        contours_small = segmentation.build_region_contours(
            label_map_small,
            min_region_area=min_region_area_work,
            morph_kernel=config.morph_kernel,
            contour_simplify_eps=config.contour_simplify_eps,
            merge_small_regions=config.merge_small_regions,
            palette_bgr=palette_bgr,
        )
        seg_time = time.perf_counter() - t0
        total_contours = sum(len(cs) for cs in contours_small.values()) if contours_small else 0
        logger.info(
            "Построены контуры для {} цветов (рабочее разрешение), всего областей: {}, за {:.2f} с",
            len(contours_small) if contours_small else 0,
            total_contours,
            seg_time,
        )
        logger.debug(
            "Областей по цветам: {}",
            {int(k) + 1: len(v) for k, v in sorted(contours_small.items())} if contours_small else {},
        )
        boundary_pixels = np.sum(render._build_boundary_mask(label_map_small) > 0)
        logger.debug("Пикселей границы на рабочем разрешении: {}", boundary_pixels)

        scale_x = orig_w / work_w
        scale_y = orig_h / work_h
        contours_by_color = segmentation.scale_contours(contours_small, scale_x, scale_y)
        label_map_full = io.resize_label_map_to_original(
            label_map_small,
            (orig_h, orig_w),
        )

        if config.draw_outline:
            logger.debug(
                "Рендер контура: границы из label_map (разрешение {}x{}), областей для номеров: {}",
                orig_h, orig_w, total_contours,
            )
            outline_image = render.render_outline_with_numbers(
                label_map_full,
                contours_by_color,
                palette_bgr,
            )

        if config.draw_preview:
            preview_image = render.render_quantized_preview(
                label_map_full,
                palette_bgr,
            )
    else:
        label_map_full = io.resize_label_map_to_original(
            label_map_small,
            (orig_h, orig_w),
        )

    return ResultBundle(
        outline_image=outline_image,
        preview_image=preview_image,
        palette_bgr=palette_bgr,
        palette_hex=palette_hex,
        label_map=label_map_full,
        contours_by_color=contours_by_color,
    )

