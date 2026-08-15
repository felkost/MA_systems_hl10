"""Process-local observability. Logging half only (spec Sec12, stage 2); the
OTel/Langfuse half joins this module at stage 9, when its consumer exists.

Rotating file logs are the post-mortem artefact for a crash: OTel's
`BatchSpanProcessor` buffers spans in memory, and a hard crash loses the
unbuffered tail. A file handler flushes per line and survives it.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import paths

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(
    service: str,
    *,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Attach a rotating file handler on `paths.log_path(service, log_dir)`
    to the logger named `service`, and return it.

    Parameters
    ----------
    service : str
        Process name; also the logger name and the log file's stem.
    log_dir : str or Path, default "logs"
        Forwarded to `paths.log_path` -- explicit so a test can redirect it.
    level : int, default logging.INFO
        Logger level.
    max_bytes, backup_count : int
        `RotatingFileHandler` rollover parameters.

    Returns
    -------
    logging.Logger
        Idempotent: a logger that already carries a `RotatingFileHandler`
        pointed at this same resolved path is returned unchanged, not given
        a second handler -- otherwise importing a module that calls this
        more than once would duplicate every line it writes.
    """
    path = paths.log_path(service, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(service)
    logger.setLevel(level)

    target = str(path.resolve())
    already_attached = any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", None) == target
        for handler in logger.handlers
    )
    if not already_attached:
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)

    return logger
