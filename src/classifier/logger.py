"""Lightweight logger wrapping Python's logging module."""

import logging
import sys
from typing import Optional


class Logger:
    """Simple configurable logger with console and optional file output."""

    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
    ):
        self._level = getattr(logging, log_level.upper(), logging.INFO)
        self._logger = logging.getLogger("classifier")
        self._logger.setLevel(self._level)
        self._logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if console_output:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self._level)
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

        if log_file:
            from pathlib import Path
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(self._level)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

    # Convenience methods matching the old engine's Logger API
    def log(self, message: str, level: str = "INFO"):
        getattr(self._logger, level.lower())(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)
