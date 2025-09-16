# app/utils/decorators.py
from functools import wraps, cache
from typing import Callable
from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def log_and_cache(label: str):
    """
    Decorator to log function calls with label and environment,
    and cache the result.
    """

    def decorator(func: Callable):
        cached_func = cache(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("[%s][%s] Calling %s", settings.env, label, func.__name__)
            result = cached_func(*args, **kwargs)
            logger.info("[%s][%s] Completed %s", settings.env, label, func.__name__)
            return result

        return wrapper

    return decorator
