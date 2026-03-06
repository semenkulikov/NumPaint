from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from loguru import logger

from . import Config, generate_paint_by_numbers, render
from .logging_config import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpaint",
        description="Преобразование изображений в раскраски по номерам.",
    )
    parser.add_argument("input", help="Путь к входному изображению.")
    parser.add_argument(
        "--colors",
        type=int,
        default=24,
        help="Количество цветов (кластеров) в палитре (если не включен --auto-colors).",
    )
    parser.add_argument(
        "--auto-colors",
        action="store_true",
        help="Автоматически подбирать количество цветов по ошибке квантования.",
    )
    parser.add_argument(
        "--min-colors",
        type=int,
        default=8,
        help="Минимальное число цветов при авто-подборе.",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=40,
        help="Максимальное число цветов при авто-подборе.",
    )
    parser.add_argument(
        "--target-mse",
        type=float,
        default=None,
        help="Целевая ошибка в Lab для авто-подбора цветов (опция).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Максимальная сторона изображения для внутренней обработки.",
    )
    parser.add_argument(
        "--min-region",
        type=int,
        default=None,
        help="Минимальная площадь области (в пикселях). По умолчанию подбирается автоматически.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Базовый каталог для сохранения результатов (по умолчанию result/<basename>).",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Экспортировать также векторный контур в SVG (с номерами).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Экспортировать контур в PDF.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Не сохранять цветное постеризованное превью.",
    )
    parser.add_argument(
        "--no-outline",
        action="store_true",
        help="Не сохранять контур с номерами.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level.upper())

    cfg = Config(
        colors=args.colors,
        max_size=args.max_size,
        min_region_area=args.min_region,
        draw_preview=not args.no_preview,
        draw_outline=not args.no_outline,
        auto_colors=args.auto_colors,
        min_colors=args.min_colors,
        max_colors=args.max_colors,
        target_mse=args.target_mse,
    )

    logger.info("Запуск numpaint для входа {}", args.input)
    result = generate_paint_by_numbers(args.input, cfg)

    base_name = Path(args.input).stem
    if args.out_dir is None:
        out_dir = Path("result") / base_name
    else:
        out_dir = Path(args.out_dir) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Запись результатов в {}", out_dir)

    if result.outline_image is not None and not args.no_outline:
        outline_path = out_dir / f"{base_name}_outline.png"
        cv2.imwrite(str(outline_path), result.outline_image)
        logger.info("Сохранён контур: {}", outline_path)

        if args.pdf:
            pdf_path = out_dir / f"{base_name}_outline.pdf"
            render.export_outline_pdf(result.outline_image, pdf_path)
            logger.info("Сохранён PDF контура: {}", pdf_path)

    if result.preview_image is not None and not args.no_preview:
        preview_path = out_dir / f"{base_name}_preview.png"
        cv2.imwrite(str(preview_path), result.preview_image)
        logger.info("Сохранено превью: {}", preview_path)

    palette_records = []
    for idx, (bgr, hex_str) in enumerate(
        zip(result.palette_bgr, result.palette_hex),
        start=1,
    ):
        b, g, r = [int(x) for x in bgr.tolist()]
        palette_records.append(
            {
                "index": idx,
                "bgr": [b, g, r],
                "rgb": [r, g, b],
                "hex": hex_str,
            }
        )

    palette_path = out_dir / f"{base_name}_palette.json"
    palette_path.write_text(
        json.dumps(palette_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Сохранена палитра: {}", palette_path)

    if args.svg and result.contours_by_color is not None:
        svg_path = out_dir / f"{base_name}_outline.svg"
        h, w = result.label_map.shape[:2]
        render.export_outline_svg(
            result.contours_by_color,
            (h, w),
            svg_path,
            draw_numbers=True,
        )
        logger.info("Сохранён SVG: {}", svg_path)


if __name__ == "__main__":
    main()

