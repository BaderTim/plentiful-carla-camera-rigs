"""
Shared logging utilities for the PCCR pipeline.

All pipeline entry points (``core/``, ``debug/``) use the same
``scenario_runner`` logger so that log output is consolidated into a single
file when multiple modules are active in the same process.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path


def setup_logging(output_dir: str, debug: bool = False) -> tuple[logging.Logger, Path]:
    """Configure the ``scenario_runner`` logger to write to a timestamped file
    and to the console.

    Args:
        output_dir: Directory where the log file will be written.  Created if
            it does not exist.
        debug: When ``True`` the console handler also emits ``DEBUG``-level
            messages.  The file handler always captures ``DEBUG`` regardless.

    Returns:
        A tuple of ``(logger, log_file_path)``.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"scenario_run_{timestamp}.log"

    logger = logging.getLogger("scenario_runner")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger, log_file


def log_print(message: str, level: str = "INFO") -> None:
    """Emit *message* through the ``scenario_runner`` logger.

    A convenience wrapper that mirrors the ``print``-then-log pattern used
    throughout the pipeline.  Falls back to ``print`` when no handlers are
    configured (e.g. during unit tests).

    Args:
        message: Text to log.
        level: One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
            Case-insensitive.  Defaults to ``"INFO"``.
    """
    logger = logging.getLogger("scenario_runner")
    lvl = level.upper()
    if lvl == "ERROR":
        logger.error(message)
    elif lvl == "WARNING":
        logger.warning(message)
    elif lvl == "DEBUG":
        logger.debug(message)
    else:
        logger.info(message)
