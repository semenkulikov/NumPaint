from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from loguru import logger


router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    logger.info("Получена команда /start от пользователя {}", message.from_user.id if message.from_user else None)
    text = (
        "Привет! Я бот NumPaint.\n\n"
        "Принимаю изображение и превращаю его в раскраску по номерам.\n\n"
        "Команды:\n"
        "/download — прислать картинку и получить раскраску\n"
    )
    await message.answer(text)

