# FILE: app / core / config.py
"""
Application configuration settings.

This module defines the Settings class for managing environment variables,
paths, server, training, and logging configuration.
"""

from functools import cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.

    Includes server, training, paths, and logging configuration.
    """

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True
    env: str = "dev"  # dev, prod, staging

    # Training / ML
    epochs: int = 10
    batch_size: int = 32

    # Paths
    model_path: str = "app/models/weights/mlp_churn_model.h5"
    default_csv_path: str = "app/data/raw/customer-churn-prediction-2020/train.csv"
    processed_csv_path: str = "app/data/processed"

    # Logging
    log_level: str = "INFO"  # default log level

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
    )

    def __repr__(self) -> str:
        return (
            f"<Settings(env={self.env}, host={self.host}, port={self.port}, "
            f"debug={self.debug}, epochs={self.epochs}, batch_size={self.batch_size}, "
            f"log_level={self.log_level})>"
        )


@cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
