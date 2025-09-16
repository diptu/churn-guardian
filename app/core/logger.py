"""
FILE: app/core/logger.py
Reusable logging setup for the entire app with color-coded output.
"""

import logging

from colorlog import ColoredFormatter

from app.core.config import get_settings

settings = get_settings()

LOG_FORMAT = (
    "%(log_color)s[%(levelname)s][%(asctime)s][%(name)s][%(funcName)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

COLOR_MAP = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def get_logger(name: str) -> logging.LoggerAdapter:
    """
    Returns a color-coded logger with environment info.

    Args:
        name (str): Name of the logger, usually __name__ of the module.

    Returns:
        logging.LoggerAdapter: Configured logger with color output.
    """
    logger = logging.getLogger(name)
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT, log_colors=COLOR_MAP)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Return LoggerAdapter with environment context
    return logging.LoggerAdapter(logger, {"env": settings.env})
