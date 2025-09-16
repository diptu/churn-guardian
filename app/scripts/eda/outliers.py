"""
Detect outliers in numeric columns using IQR method.
"""

import pandas as pd


def detect_outliers(df: pd.DataFrame) -> dict:
    """Return dictionary with count of outliers per numeric column."""
    outliers = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
        outliers[col] = int(mask.sum())
    return outliers
