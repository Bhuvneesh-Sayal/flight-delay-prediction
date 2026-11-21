"""
src/utils/logger.py
===================
Centralised logging for the pipeline.

Security hardening:
  - Log directory created with mode 0o700 (owner-only)
  - Log file restricted to 0o600 after creation (owner read/write only)
  - _SafeStreamHandler swallows flush errors (OneDrive/network stdout)
  - _NewlineEscapingFormatter prevents log injection via server error strings

Public API
----------
  get_logger(name) -> logging.Logger
"""

import logging
import os
import sys

from config.settings import PATHS

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = PATHS["logs"] / "pipeline.log"


class _NewlineEscapingFormatter(logging.Formatter):
    """
    Formatter that escapes \\n and \\r in the final message string.

    Prevents log injection: a server error body containing embedded newlines
    cannot forge additional log lines.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return msg.replace("\n", "\\n").replace("\r", "\\r")


class _SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler that silently swallows ALL stdout write/flush errors.

    On OneDrive-backed paths stdout occasionally times out. Python's default
    Handler.handleError() prints a '--- Logging error ---' traceback to
    stderr even when the exception is caught inside emit(). We override
    handleError() as well so that secondary traceback never appears.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.stream.flush()
        except Exception:
            pass  # swallow write/flush errors silently

    def handleError(self, record: logging.LogRecord) -> None:
        pass  # suppress the '--- Logging error ---' traceback


def _suppress_logging_errors() -> None:
    """
    Monkey-patch logging.Handler.handleError globally so that no handler
    anywhere in the process prints '--- Logging error ---' tracebacks.
    This is needed because Python calls handleError on the *base class*
    in some internal paths, bypassing our subclass override.
    """
    logging.Handler.handleError = lambda self, record: None


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger named *name* that writes to stdout and logs/pipeline.log.

    Handlers are attached only once per logger name, so repeated calls
    do not produce duplicate log lines.

    Security:
      - logs/ directory created with mode 0o700 (owner-only access)
      - pipeline.log restricted to 0o600 after creation
      - All log messages have newlines escaped to prevent log injection
    """
    logger = logging.getLogger(name)

    if getattr(logger, "_pipeline_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate output via the root logger

    formatter = _NewlineEscapingFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Safe stream handler: swallows flush timeouts from OneDrive-backed stdout
    stream_handler = _SafeStreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create log directory with restrictive permissions (owner-only)
    log_dir = LOG_FILE.parent
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Tighten existing directory permissions if it was created before this fix
    try:
        os.chmod(log_dir, 0o700)
    except OSError:
        pass

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Restrict log file to owner read/write only (0o600)
    try:
        os.chmod(LOG_FILE, 0o600)
    except OSError:
        pass

    logger._pipeline_configured = True
    return logger


# Apply globally at module import time — before any handler is created
_suppress_logging_errors()
