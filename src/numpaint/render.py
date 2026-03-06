from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import layout


ArrayLike = np.ndarray


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


def _render_legend(
    height: int,
    palette_bgr: np.ndarray,
    hex_colors: list[str],
) -> np.ndarray:
    """
    Рисует вертикальную легенду: цветной прямоугольник + номер + hex.
    Высота должна быть не меньше n * MIN_LEGEND_ROW_HEIGHT, чтобы поместились все цвета.
    """
    n = len(palette_bgr)
    legend_width = 260
    legend = np.full((height, legend_width, 3), 255, dtype=np.uint8)

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

        cv2.putText(
            legend,
            hex_str,
            (x2 + 40, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    return legend


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
    Рисует контуры, номера внутри областей и легенду сбоку.
    Границы строятся по карте меток (один пиксель границы на ребро), затем
    рисуются чёрным; findContours по маске границ не используется, т.к. даёт
    только внешние контуры связных компонент и теряет внутренние линии.
    """
    h, w = label_map.shape[:2]
    outline = np.full((h, w, 3), 255, dtype=np.uint8)

    font_scale, thickness = _compute_font_params(h, w)

    boundary = _build_boundary_mask(label_map)
    if boundary_line_thickness <= 1:
        outline[boundary.astype(bool)] = (0, 0, 0)
    else:
        kernel = np.ones((boundary_line_thickness, boundary_line_thickness), np.uint8)
        boundary_thick = cv2.dilate(boundary, kernel)
        outline[boundary_thick.astype(bool)] = (0, 0, 0)

    # Номера цветов по центрам областей (из переданных контуров)
    for color_idx, contours in contours_by_color.items():
        label_str = str(int(color_idx) + 1)
        for cnt in contours:
            cx, cy = layout.contour_center(cnt)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            cv2.putText(
                outline,
                label_str,
                (cx - 8, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )

    n_colors = len(palette_bgr)
    legend_height = max(h, n_colors * MIN_LEGEND_ROW_HEIGHT)
    legend_width = 260
    hex_colors = build_palette_hex(palette_bgr)
    legend = _render_legend(legend_height, palette_bgr, hex_colors)
    combined = np.full((legend_height, w + legend_width, 3), 255, dtype=np.uint8)
    combined[:h, :w] = outline
    combined[:legend_height, w : w + legend_width] = legend
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

