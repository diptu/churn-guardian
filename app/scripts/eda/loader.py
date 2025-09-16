"""
Load dataset for EDA.
"""

from app.services import load_train
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Load training dataset for EDA."""
    return load_train()
