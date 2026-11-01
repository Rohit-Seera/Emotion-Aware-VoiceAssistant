"""Application logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.settings import settings


LOG_DIRECTORY = Path("logs")
LOG_FILE = LOG_DIRECTORY / "backend.log"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """Configure application logging."""

    LOG_DIRECTORY.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    root_logger.setLevel(
        logging.DEBUG if settings.app.debug else logging.INFO
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""

    return logging.getLogger(name)


configure_logging()

logger = get_logger("emotion_aware_voice_assistant")
