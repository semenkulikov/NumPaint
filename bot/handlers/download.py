from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Set

import cv2
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from loguru import logger

from numpaint import Config, generate_paint_by_numbers


router = Router()

_pending_download_users: Set[int] = set()


@router.message(Command("download"))
async def cmd_download(message: Message) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id
    _pending_download_users.add(user_id)
    logger.info("Пользователь {} начал сценарий /download", user_id)
    await message.answer("Пришли фото или файл изображения, я сделаю из него раскраску по номерам.")


@router.message(F.photo | F.document)
async def handle_image(message: Message) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id

    if user_id not in _pending_download_users:
        return

    _pending_download_users.discard(user_id)

    if message.photo:
        file = message.photo[-1]
        suffix = ".jpg"
        logger.info("Получено фото от пользователя {}", user_id)
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        file = message.document
        suffix = Path(message.document.file_name or "image").suffix or ".jpg"
        logger.info("Получен файл-изображение от пользователя {}", user_id)
    else:
        await message.answer("Это не похоже на изображение. Попробуй ещё раз с /download.")
        return

    base_dir = Path("result")
    ts = time.strftime("%Y%m%d_%H%M%S")
    user_dir = base_dir / f"user_{user_id}_{ts}"
    user_dir.mkdir(parents=True, exist_ok=True)

    input_path = user_dir / f"input{suffix}"

    await message.answer("Обрабатываю изображение, подожди немного...")

    if message.photo:
        await file.download(destination=str(input_path))
    else:
        # Для документов используем загрузку через бота
        assert message.document is not None
        await message.bot.download(message.document, destination=str(input_path))

    loop = asyncio.get_running_loop()

    def _run_pipeline() -> tuple[Path | None, Path | None]:
        cfg = Config()
        try:
            result = generate_paint_by_numbers(str(input_path), cfg)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка пайплайна для пользователя {}: {}", user_id, exc)
            return None, None

        outline_path = user_dir / "outline.png"
        colorized_path = user_dir / "colorized.png"
        if result.outline_image is not None:
            cv2.imwrite(str(outline_path), result.outline_image)
        if result.preview_image is not None:
            cv2.imwrite(str(colorized_path), result.preview_image)
        return outline_path, colorized_path

    outline_path, colorized_path = await loop.run_in_executor(None, _run_pipeline)

    if outline_path is None or not outline_path.exists():
        await message.answer("Не получилось обработать изображение. Попробуй другое или позже.")
        return

    file_outline = FSInputFile(str(outline_path))
    await message.answer_photo(file_outline, caption="Вот твоя раскраска по номерам (контур с номерами).")

    if colorized_path is not None and colorized_path.exists():
        file_colorized = FSInputFile(str(colorized_path))
        await message.answer_photo(file_colorized, caption="Для сравнения — раскрашенный вариант по палитре.")

