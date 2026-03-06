from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from loguru import logger
from pathlib import Path
import sys

# Добавляем src/ в sys.path, чтобы можно было импортировать numpaint при запуске из корня проекта
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from numpaint.logging_config import setup_logging
from .config import load_config
from .handlers import download, echo, start


async def _run() -> None:
    """Загрузка конфига, настройка логов, регистрация роутеров и запуск polling."""
    cfg = load_config()
    setup_logging(cfg.log_level)

    bot = Bot(
        token=cfg.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    dp.include_router(start.router)
    dp.include_router(download.router)
    dp.include_router(echo.router)

    logger.info("Запуск опроса (polling)")
    await dp.start_polling(bot)


def main() -> None:
    """Точка входа: запуск бота через asyncio."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()

