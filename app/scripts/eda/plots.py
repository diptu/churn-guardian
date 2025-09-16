"""
Plotting utilities for EDA: univariate, correlations, pairwise interactions.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_univariate(df: pd.DataFrame, target: str, outdir: Path):
    """Save univariate plots for numeric and categorical features."""
    outdir.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    for col in num_cols[:10]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(outdir / f"{col}_dist.png", dpi=300)
        plt.close()

    for col in cat_cols[:10]:
        plt.figure(figsize=(6, 4))
        df[col].value_counts().head(20).plot(kind="bar")
        plt.title(f"Top categories of {col}")
        plt.tight_layout()
        plt.savefig(outdir / f"{col}_bar.png", dpi=300)
        plt.close()

    if target in df.columns:
        plt.figure(figsize=(6, 4))
        df[target].value_counts().plot(kind="bar")
        plt.title(f"{target} distribution")
        plt.tight_layout()
        plt.savefig(outdir / f"{target}_distribution.png", dpi=300)
        plt.close()


def plot_correlations(df: pd.DataFrame, outdir: Path):
    """Plot correlation heatmap for numeric features."""
    outdir.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(outdir / "correlations.png", dpi=300)
    plt.close()


def plot_pairwise_interactions(
    df: pd.DataFrame, target: str, outdir: Path, sample_size: int = 500
):
    """Plot pair plots to detect interaction effects."""
    outdir.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cols_to_plot = num_cols[:5]  # limit to first 5 numeric for clarity
    if target in df.columns:
        cols_to_plot.append(target)

    df_sampled = df[cols_to_plot].sample(n=min(sample_size, len(df)), random_state=42)
    sns.pairplot(df_sampled, hue=target, corner=True)
    plt.tight_layout()
    plt.savefig(outdir / "pairwise_interactions.png", dpi=300)
    plt.close()
