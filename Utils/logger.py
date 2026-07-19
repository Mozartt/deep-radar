import logging
import sys
from pathlib import Path
from typing import Optional, Union


class Logger:
    """Simple logger that writes to both the console and a log file.

    Usage:
        logger = Logger(log_file="runs/train_radar_DETR.log")
        logger.info("Starting training...")
        logger.info(f"Epoch {epoch:03d} | train loss: {train_loss:.6f}")
    """

    def __init__(
        self,
        log_file: Union[str, Path],
        name: str = "radar",
        level: int = logging.INFO,
        mode: str = "a",
    ):
        """
        Args:
            log_file: Path to the output log file. Parent directories are
                created automatically if they do not exist.
            name: Logger name (useful if you create multiple loggers).
            level: Logging level (e.g. logging.INFO, logging.DEBUG).
            mode: File open mode, "a" to append or "w" to overwrite.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        # Avoid duplicate handlers if a Logger with the same name is created twice.
        if self._logger.handlers:
            self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        file_handler = logging.FileHandler(self.log_file, mode=mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def close(self):
        """Flush and close all handlers (call at the end of training if needed)."""
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
