"""
Core package for shared configuration and logging.
"""

from .config import get_settings
from .logger import get_logger

__all__ = ["get_settings", "get_logger"]
