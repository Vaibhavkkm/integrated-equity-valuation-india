"""
Centralised logging.

Why this module exists:

The codebase had `print(...)` calls scattered through it for progress
chatter ("→ Valuing TCS.NS ...", "  ! peer X skipped"). That was fine for
a CLI demo but hostile to anyone embedding the library — Streamlit would
spam its own stdout, tests would deadlock on captured output, and there
was no way to silence one module without silencing the whole engine.

This module gives every other module a `logging.Logger` configured to:

  * Write to stderr by default (stdout is reserved for *results*).
  * Honour the `EQUITY_LOG_LEVEL` env var (DEBUG, INFO, WARNING, ERROR).
  * Use a compact, prefixable format so the pipeline stages line up.
  * Work without setup — `get_logger(__name__)` is enough.

Usage in any module:

    from src.logging_setup import get_logger
    log = get_logger(__name__)
    log.info("fetching %s", ticker)
"""
from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"

_LEVEL = os.environ.get("EQUITY_LOG_LEVEL", "INFO").upper()


def _configure_root() -> None:
    root = logging.getLogger("equity_valuation")
    if root.handlers:
        return  # already configured (idempotent on repeated import)
    root.setLevel(_LEVEL)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the engine's namespace."""
    _configure_root()
    # Strip the leading "src." so log lines read "ddm_models" not "src.ddm_models".
    short = name.split(".")[-1] if name.startswith("src.") else name
    return logging.getLogger("equity_valuation").getChild(short)


def set_level(level: str) -> None:
    """Override the log level at runtime (e.g. from a CLI flag)."""
    _configure_root()
    logging.getLogger("equity_valuation").setLevel(level.upper())
