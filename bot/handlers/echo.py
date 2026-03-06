from __future__ import annotations

from aiogram import F, Router
from aiogram.types import Message
from loguru import logger


router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def unknown_text(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else None
    logger.info("Неизвестное текстовое сообщение от пользователя {}: {}", user_id, message.text)
    await message.answer(
        "Я тебя не понял.\n"
        "Доступные команды:\n"
        "/start — краткая справка\n"
        "/download — прислать картинку и получить раскраску"
    )

