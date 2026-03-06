from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(level: LogLevel = "INFO") -> None:
    """
    Базовая настройка loguru:
    - вывод в stdout;
    - файловый лог в logs/numpaint.log.
    """
    logger.remove()

    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        backtrace=True,
        diagnose=False,
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_path = logs_dir / "numpaint.log"

    logger.add(
        file_path,
        level=level,
        rotation="10 MB",
        retention=10,  # хранить последние 10 файлов лога
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

