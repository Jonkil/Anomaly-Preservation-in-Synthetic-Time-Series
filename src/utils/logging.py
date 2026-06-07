"""Lightweight logging helpers (extend for structured logs later)."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a basic stream handler."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        log.addHandler(h)
    log.setLevel(level)
    return log
