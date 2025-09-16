"""
FILE: app/service/model_loader.py
Utilities for saving and loading machine learning models.
Includes caching and logging.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from app.core import get_logger, get_settings
from app.utils.decorators import log_and_cache

# --------------------------
# Settings and Logger
# --------------------------
settings = get_settings()
logger = get_logger(__name__)

MODEL_DIR = Path(settings.model_path).parent
_model_cache: dict[str, object] = {}


# --------------------------
# Model Load/Save
# --------------------------
@log_and_cache("MODEL_LOAD")
def load_model(model_name: str) -> object:
    """
    Load a model from disk, with in-memory caching.

    Args:
        model_name (str): Name of the model file without extension.

    Returns:
        object: Loaded model.
    """
    if model_name in _model_cache:
        logger.info(
            "[%s][MODEL_LOAD] Retrieved '%s' from cache", settings.env, model_name
        )
        return _model_cache[model_name]

    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    _model_cache[model_name] = model
    logger.info(
        "[%s][MODEL_LOAD] Loaded model '%s' from disk", settings.env, model_name
    )
    return model


def save_model(model: object, model_name: str):
    """
    Save a model to disk and update cache.

    Args:
        model (object): Model object to save.
        model_name (str): Name of the model file without extension.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    _model_cache[model_name] = model
    logger.info(
        "[%s][MODEL_SAVE] Saved model '%s' to %s", settings.env, model_name, model_path
    )
