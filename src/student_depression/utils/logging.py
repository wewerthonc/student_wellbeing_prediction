"""Consistent logging configuration."""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger without duplicating handlers."""
    logger = logging.getLogger("student_depression")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
