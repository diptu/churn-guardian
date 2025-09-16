"""
Minimal unit tests for app.service.data_loader
Tests basic data loading and train/validation splitting.
"""

import pandas as pd
import pytest

from app.core import get_settings
from app.services.data_loader import get_splits, load_test, load_train

settings = get_settings()


@pytest.fixture
def sample_train_csv(tmp_path):
    """Temporary train CSV for testing."""
    df = pd.DataFrame(
        {
            "feature1": range(4),
            "feature2": range(4, 8),
            "churn": ["no", "yes", "no", "yes"],
        }
    )
    file_path = tmp_path / "train.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_test_csv(tmp_path):
    """Temporary test CSV for testing."""
    df = pd.DataFrame(
        {
            "feature1": range(2),
            "feature2": range(2, 4),
        }
    )
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_load_train(monkeypatch, sample_train_csv):
    monkeypatch.setattr(settings, "default_csv_path", sample_train_csv)
    df = load_train()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4


def test_load_test(monkeypatch, sample_test_csv):
    monkeypatch.setattr("app.service.data_loader.RAW_DATA_DIR", sample_test_csv.parent)
    df = load_test()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2


def test_get_splits(monkeypatch, sample_train_csv):
    monkeypatch.setattr(settings, "default_csv_path", sample_train_csv)
    df = load_train()
    X_train, X_val, y_train, y_val = get_splits(df, test_size=0.5, random_state=42)
    # Check split sizes
    assert X_train.shape[0] + X_val.shape[0] == df.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == df.shape[0]
