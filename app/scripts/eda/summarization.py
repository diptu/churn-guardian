"""
Summarize dataset and detect unnecessary features.
"""

import pandas as pd


def summarize(df: pd.DataFrame, target: str) -> dict:
    """Return dataset summary: shape, dtypes, missing, target distribution."""
    summary = {}
    summary["n_rows"], summary["n_cols"] = df.shape
    summary["dtypes"] = df.dtypes.astype(str).to_dict()
    summary["missing_ratio"] = df.isna().mean().sort_values(ascending=False).to_dict()

    if target in df.columns:
        counts = df[target].value_counts().to_dict()
        pct = df[target].value_counts(normalize=True).to_dict()
        summary["target_distribution"] = {"counts": counts, "percent": pct}

    return summary


def find_unnecessary_columns(df: pd.DataFrame, target: str, missing_thresh=0.5) -> dict:
    """Detect ID-like, constant, high-missing, and high-cardinality categorical columns."""
    recs = {}
    nunique = df.nunique()
    missing_ratio = df.isna().mean()

    recs["id_like"] = [c for c in df.columns if nunique[c] == len(df) and c != target]
    recs["constant"] = [c for c in df.columns if nunique[c] == 1]
    recs["high_missing"] = [c for c in df.columns if missing_ratio[c] > missing_thresh]
    recs["high_cardinality"] = [
        c
        for c in df.select_dtypes(include="object").columns
        if nunique[c] > 100 and c != target
    ]
    return recs
