from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import colors_ru
from . import layout


ArrayLike = np.ndarray

# Минимальная площадь области (в пикселях), при которой рисуется номер; иначе — пропуск
MIN_AREA_TO_DRAW_NUMBER = 200
# Минимальное расстояние между центрами номеров (чтобы не накладывались вложенные области)
MIN_DISTANCE_BETWEEN_NUMBERS = 24


def build_palette_hex(palette_bgr: ArrayLike) -> list[str]:
    """
    Преобразует BGR-палитру в список hex-строк (#RRGGBB).
    """
    palette_bgr = np.asarray(palette_bgr, dtype=np.uint8)
    b = palette_bgr[:, 0].astype(int)
    g = palette_bgr[:, 1].astype(int)
    r = palette_bgr[:, 2].astype(int)
    hex_list = [f"#{r_i:02X}{g_i:02X}{b_i:02X}" for r_i, g_i, b_i in zip(r, g, b)]
    return hex_list


def render_quantized_preview(
    label_map: np.ndarray,
    palette_bgr: np.ndarray,
) -> np.ndarray:
    """
    Строит постеризованное цветное превью по карте меток и палитре.
    """
    h, w = label_map.shape[:2]
    palette = np.asarray(palette_bgr, dtype=np.uint8)
    flat_labels = label_map.reshape(-1).astype(np.int64)
    flat_colors = palette[flat_labels]
    img = flat_colors.reshape(h, w, 3)
    return img


def _compute_font_params(h: int, w: int) -> tuple[float, int]:
    base = min(h, w) / 600.0
    font_scale = max(0.4, min(1.4, base * 0.9))
    thickness = max(1, int(round(font_scale * 2.0)))
    return font_scale, thickness


MIN_LEGEND_ROW_HEIGHT = 28
LEGEND_WIDTH = 300


def _render_legend(
    height: int,
    palette_bgr: np.ndarray,
    hex_colors: list[str],
    color_names: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Рисует вертикальную легенду: цветной прямоугольник + номер + название цвета (рус.) или hex.
    """
    n = len(palette_bgr)
    legend = np.full((height, LEGEND_WIDTH, 3), 255, dtype=np.uint8)

    if n == 0:
        return legend

    row_h = max(MIN_LEGEND_ROW_HEIGHT, height // n)
    font_scale = 0.5
    thickness = 1

    for idx, (bgr, hex_str) in enumerate(zip(palette_bgr, hex_colors), start=1):
        cy = int((idx - 1) * row_h + row_h // 2)
        y1 = max(4, cy - row_h // 2 + 4)
        y2 = min(height - 4, y1 + row_h - 8)
        x1 = 16
        x2 = 16 + 40
        color = tuple(int(c) for c in bgr.tolist())
        cv2.rectangle(legend, (x1, y1), (x2, y2), color, thickness=-1)

        text_num = str(idx)
        cv2.putText(
            legend,
            text_num,
            (x2 + 12, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

        label_text = (color_names[idx - 1] if color_names and idx <= len(color_names) else hex_str)
        tx_label = x2 + 40
        ty_label = y2 - 6
        if color_names and any(ord(c) > 127 for c in label_text):
            font_pil = _get_cyrillic_font(size=14)
            if font_pil is not None:
                x1_r, y1_r = tx_label, max(0, ty_label - 18)
                x2_r, y2_r = min(LEGEND_WIDTH, tx_label + 240), min(height, ty_label + 4)
                roi = legend[y1_r:y2_r, x1_r:x2_r].copy()
                pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_roi)
                draw.text((0, 0), label_text, font=font_pil, fill=(0, 0, 0))
                legend[y1_r:y2_r, x1_r:x2_r] = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(legend, hex_str, (tx_label, ty_label), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        else:
            cv2.putText(legend, label_text, (tx_label, ty_label), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return legend


def _get_cyrillic_font(size: int = 14) -> Optional[ImageFont.FreeTypeFont]:
    """Шрифт с поддержкой кириллицы для подписей в палитре."""
    candidates = []
    if sys.platform == "win32":
        candidates.extend([
            os.path.expandvars(r"%WINDIR%\Fonts\arial.ttf"),
            os.path.expandvars(r"%WINDIR%\Fonts\calibri.ttf"),
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ])
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def _build_boundary_mask(label_map: np.ndarray) -> np.ndarray:
    """
    Единая маска границ: пиксель = 255, если у него есть 4-сосед с другой меткой.
    """
    h, w = label_map.shape[:2]
    lab = np.asarray(label_map, dtype=np.int32)
    top = lab[:-1, :] != lab[1:, :]
    left = lab[:, :-1] != lab[:, 1:]
    boundary = np.zeros((h, w), dtype=np.uint8)
    boundary[:-1, :] |= top
    boundary[:, :-1] |= left
    return boundary


def get_boundary_contours(
    label_map: np.ndarray,
    smooth_eps: float = 0.0,
) -> List[np.ndarray]:
    """
    Контуры границ между регионами по карте меток (в координатах этой карты).
    smooth_eps > 0: упрощение контуров (approxPolyDP) для гладких вектороподобных линий.
    """
    boundary = _build_boundary_mask(label_map)
    contours, _ = cv2.findContours(
        boundary,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )
    contours = list(contours)
    if smooth_eps > 0:
        contours = _smooth_contours(contours, smooth_eps)
    return contours


def _smooth_contours(contours: List[np.ndarray], eps: float) -> List[np.ndarray]:
    """Упрощение контуров (Douglas–Peucker) для гладких линий."""
    out: List[np.ndarray] = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        peri = cv2.arcLength(cnt, closed=True)
        if peri <= 0:
            out.append(cnt)
            continue
        # Не схлопывать мелкие контуры: eps не больше 5% периметра
        effective_eps = min(eps, 0.05 * peri)
        approx = cv2.approxPolyDP(cnt, effective_eps, closed=True)
        if approx.shape[0] >= 2:
            out.append(approx)
    return out


def render_outline_with_numbers(
    label_map: np.ndarray,
    contours_by_color: Dict[int, List[np.ndarray]],
    palette_bgr: np.ndarray,
    boundary_contours: Optional[List[np.ndarray]] = None,
    boundary_line_thickness: int = 2,
) -> np.ndarray:
    """
    Рисует границы по единой маске (без пересечений), номера одного размера без наложения, легенду.
    Размер цифр — по самой маленькой вписывающейся области. Вложенные области не перекрывают номера.
    """
    h, w = label_map.shape[:2]
    outline = np.full((h, w, 3), 255, dtype=np.uint8)

    font_scale_base, thickness_base = _compute_font_params(h, w)
    line_th = max(2, boundary_line_thickness)

    # Границы по единой маске label_map (пиксельная маска + эллиптическое расширение)
    boundary = _build_boundary_mask(label_map)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_th, line_th))
    boundary_thick = cv2.dilate(boundary, kernel)
    outline[boundary_thick.astype(bool)] = (0, 0, 0)

    # Рамка по размеру изображения
    cv2.rectangle(outline, (0, 0), (w - 1, h - 1), (0, 0, 0), thickness=line_th)

    # Один размер цифр: по самой маленькой вписывающейся области
    scale_candidates = [1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.28]
    global_scale = float("inf")
    for color_idx, contours in contours_by_color.items():
        label_str = str(int(color_idx) + 1)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_TO_DRAW_NUMBER:
                continue
            cnt_arr = np.asarray(cnt, dtype=np.int32)
            cx, cy = layout.contour_widest_point(cnt_arr, (h, w))
            if not (0 <= cx < w and 0 <= cy < h):
                cx, cy = layout.contour_center(cnt)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            for scale in scale_candidates:
                fs = max(0.22, font_scale_base * scale)
                th = max(1, int(round(thickness_base * scale)))
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                if layout.rect_fits_in_contour(cnt_arr, cx, cy, tw, th):
                    global_scale = min(global_scale, fs)
                    break

    font_scale_uniform = max(0.22, global_scale) if global_scale != float("inf") else font_scale_base * 0.35
    thickness_uniform = max(1, int(round(thickness_base * font_scale_uniform / font_scale_base)))
    (tw_uni, th_uni), _ = cv2.getTextSize(
        "24", cv2.FONT_HERSHEY_SIMPLEX, font_scale_uniform, thickness_uniform
    )

    # Собираем кандидатов для номеров, сортируем по площади (сначала крупные)
    candidates: List[Tuple[int, int, float, str, int]] = []
    for color_idx, contours in contours_by_color.items():
        label_str = str(int(color_idx) + 1)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_TO_DRAW_NUMBER:
                continue
            cnt_arr = np.asarray(cnt, dtype=np.int32)
            cx, cy = layout.contour_widest_point(cnt_arr, (h, w))
            if not (0 <= cx < w and 0 <= cy < h):
                cx, cy = layout.contour_center(cnt)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if not layout.rect_fits_in_contour(cnt_arr, cx, cy, tw_uni, th_uni):
                continue
            candidates.append((cx, cy, area, label_str, color_idx))
    candidates.sort(key=lambda x: -x[2])

    # Рисуем номера без наложения: если центр слишком близко к уже нарисованному — пропускаем
    placed: List[Tuple[int, int]] = []
    for cx, cy, _area, label_str, _color_idx in candidates:
        too_close = any(
            (cx - px) ** 2 + (cy - py) ** 2 < MIN_DISTANCE_BETWEEN_NUMBERS ** 2
            for px, py in placed
        )
        if too_close:
            continue
        placed.append((cx, cy))
        (tw, th), _ = cv2.getTextSize(
            label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale_uniform, thickness_uniform
        )
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(
            outline,
            label_str,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_uniform,
            (0, 0, 0),
            thickness_uniform,
            lineType=cv2.LINE_AA,
        )

    n_colors = len(palette_bgr)
    legend_height = max(h, n_colors * MIN_LEGEND_ROW_HEIGHT)
    hex_colors = build_palette_hex(palette_bgr)
    color_names = colors_ru.build_palette_russian_names(palette_bgr)
    legend = _render_legend(legend_height, palette_bgr, hex_colors, color_names=color_names)
    combined = np.full((legend_height, w + LEGEND_WIDTH, 3), 255, dtype=np.uint8)
    combined[:h, :w] = outline
    combined[:legend_height, w : w + LEGEND_WIDTH] = legend
    return combined


def export_outline_svg(
    contours_by_color: Dict[int, List[np.ndarray]],
    image_shape: Tuple[int, int],
    out_path: Path,
    draw_numbers: bool = True,
) -> None:
    """
    Экспорт контуров в SVG. При draw_numbers=True в центр каждой области добавляется номер цвета.
    Под печать можно использовать как векторный контур.
    """
    h, w = image_shape
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<g fill="none" stroke="#000000" stroke-width="1">',
    ]

    for color_idx, contours in contours_by_color.items():
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            if pts.shape[0] < 2:
                continue
            d_parts = [f"M {float(pts[0,0]):.2f},{float(pts[0,1]):.2f}"]
            for x, y in pts[1:]:
                d_parts.append(f"L {float(x):.2f},{float(y):.2f}")
            d_parts.append("Z")
            d = " ".join(d_parts)
            lines.append(f'<path d="{d}" />')

            if draw_numbers:
                cx, cy = layout.contour_center(cnt)
                label_str = str(int(color_idx) + 1)
                lines.append(
                    f'<text x="{float(cx):.2f}" y="{float(cy):.2f}" '
                    f'font-size="8" text-anchor="middle" '
                    f'dominant-baseline="central">{label_str}</text>'
                )

    lines.append("</g>")
    lines.append("</svg>")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_outline_pdf(
    outline_image: np.ndarray,
    out_path: Path,
) -> None:
    """
    Экспорт контура в PDF через Pillow.
    """
    bgr = outline_image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    out_path = Path(out_path)
    img.save(out_path, format="PDF")

