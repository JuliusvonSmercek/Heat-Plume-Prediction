import logging
import sys
from pathlib import Path
from typing import Any


def debug(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.debug(message)


def info(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.info(message)


def warning(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.warning(message)


def error(*messages: Any) -> None:
    message = " ".join([str(msg) for msg in messages])
    logging.error(message)
    exit(1)


def set_log_level_debug() -> None:
    logging.getLogger().setLevel(logging.DEBUG)


def set_log_level_info() -> None:
    logging.getLogger().setLevel(logging.INFO)


def configure_logging(
    log_path: Path | None = None,
    level: int = logging.INFO,
    clear_handlers: bool = False,
    fmt: str = "%(asctime)s %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure root logger with console output and an optional timestamped file."""
    logger = logging.getLogger()
    logger.setLevel(level)

    if clear_handlers and logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
