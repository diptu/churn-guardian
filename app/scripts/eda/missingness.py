# scripts/eda/missingness.py
"""
Missingness analysis utilities.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# --------------------------
# Missingness Summary
# --------------------------
def missingness_summary(df: pd.DataFrame) -> dict:
    """Return missingness ratio per column."""
    return df.isna().mean().sort_values(ascending=False).to_dict()


# --------------------------
# Missingness Heatmap
# --------------------------
def missingness_heatmap(df: pd.DataFrame, outdir: Path | None = None):
    """Save a heatmap visualizing missing values per feature."""
    outdir = Path(outdir) if outdir else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "missingness_heatmap.png", dpi=300)
    plt.close()
