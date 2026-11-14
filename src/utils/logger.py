"""
src/utils/logger.py
===================
Centralised logging for the pipeline.

Public API
----------
  get_logger(name) -> logging.Logger
"""

import logging
import sys

from config.settings import PATHS

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = PATHS["logs"] / "pipeline.log"


class _SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler that silently swallows flush/write errors (e.g. TimeoutError
    when stdout is backed by a slow network filesystem like OneDrive).
    The file handler always gets the record regardless.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.stream.flush()
        except Exception:
            pass  # never let a stdout flush error surface to the user


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger named *name* that writes to stdout and logs/pipeline.log.

    Handlers are attached only once per logger name, so repeated calls
    do not produce duplicate log lines.
    """
    logger = logging.getLogger(name)

    if getattr(logger, "_pipeline_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate output via the root logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Use the safe handler so stdout flush timeouts don't print tracebacks
    stream_handler = _SafeStreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger._pipeline_configured = True
    return logger
