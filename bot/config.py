from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class BotConfig:
    """Конфигурация бота: токен, уровень логов, каталог результатов, параметры numpaint."""

    bot_token: str
    log_level: str = "INFO"
    result_dir: Path = Path("result")
    max_size: int = 1024
    colors: int = 24


def load_config() -> BotConfig:
    """Читает конфиг из .env (BOT_TOKEN, LOG_LEVEL, RESULT_DIR, NUMPAINT_*)."""
    load_dotenv()

    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN не задан в .env")

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    result_dir = Path(os.getenv("RESULT_DIR", "result"))
    max_size = int(os.getenv("NUMPAINT_MAX_SIZE", "1024"))
    colors = int(os.getenv("NUMPAINT_DEFAULT_COLORS", "24"))

    return BotConfig(
        bot_token=token,
        log_level=log_level,
        result_dir=result_dir,
        max_size=max_size,
        colors=colors,
    )

