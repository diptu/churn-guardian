"""
FILE : app/service/data_loader.py
Data loading utilities for training, evaluation, and EDA.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core import get_logger, get_settings
from app.utils.decorators import log_and_cache

# --------------------------
# Settings and Logger
# --------------------------
settings = get_settings()
logger = get_logger(__name__)

RAW_DATA_DIR = Path(settings.default_csv_path).parent
PROCESSED_DATA_DIR = Path(settings.processed_csv_path)
DEFAULT_PROCESSED_PATH = PROCESSED_DATA_DIR
DEFAULT_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------
# Loaders
# --------------------------
@log_and_cache("DATA_LOAD")
def load_train() -> pd.DataFrame:
    """Load training dataset with labels."""
    path = settings.default_csv_path
    df = pd.read_csv(path)
    logger.info("Loaded train data: %s", df.shape)
    return df


@log_and_cache("DATA_LOAD")
def load_test() -> pd.DataFrame:
    """Load test dataset without labels."""
    path = RAW_DATA_DIR / "test.csv"
    df = pd.read_csv(path)
    logger.info("Loaded test data: %s", df.shape)
    return df


@log_and_cache("DATA_LOAD")
def load_sample_submission() -> pd.DataFrame:
    """Load sample submission template."""
    path = RAW_DATA_DIR / "sampleSubmission.csv"
    df = pd.read_csv(path)
    logger.info("Loaded sample submission: %s", df.shape)
    return df


# --------------------------
# Helpers
# --------------------------
def get_splits(
    df: pd.DataFrame,
    target: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/validation splits from a labeled dataset."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    x_features = df.drop(columns=[target])
    y_target = df[target]
    logger.info("[%s][DATA_SPLIT] Splitting %d samples", settings.env, len(df))
    return train_test_split(
        x_features,
        y_target,
        test_size=test_size,
        stratify=y_target,
        random_state=random_state,
    )


# --------------------------
# Optional processed data helpers
# --------------------------
def save_processed_train(df: pd.DataFrame, filename: str = "train_processed.csv"):
    """Save processed training dataset."""
    path = DEFAULT_PROCESSED_PATH / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed train data to: %s", path)


def load_processed_train(filename: str = "train_processed.csv") -> pd.DataFrame:
    """Load processed training dataset."""
    path = DEFAULT_PROCESSED_PATH / filename
    df = pd.read_csv(path)
    logger.info("Loaded processed train data: %s", df.shape)
    return df
