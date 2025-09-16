"""
Exploratory Data Analysis (EDA) for churn dataset.

Features:
- Summarize dataset
- Detect unnecessary columns (ID-like, constant, high missingness, high cardinality)
- Handle outliers
- Plot univariate distributions, correlations, pairwise interactions
- Save high-quality plots and JSON reports
- Advanced dimensionality reduction: PCA, t-SNE, UMAP
- Business-oriented analysis: churn drivers, CLV-based segmentation, executive summary

Usage:
    uv run python -m app.scripts.eda --target churn
"""

import json
from pathlib import Path

from app.scripts.eda import (
    churn_driver_waterfall,
    clv_based_analysis,
    detect_outliers,
    executive_summary,
    find_unnecessary_columns,
    load_dataset,
    missingness_heatmap,
    missingness_summary,
    plot_pairwise_interactions,
    plot_pca,
    plot_tsne,
    plot_umap,
    plot_univariate,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Separate subfolders for plots
UNIVARIATE_DIR = REPORTS_DIR / "plots" / "univariate"
PAIRWISE_DIR = REPORTS_DIR / "plots" / "pairwise"
DIMENSIONALITY_DIR = REPORTS_DIR / "plots" / "dimensionality"
BUSINESS_DIR = REPORTS_DIR / "plots" / "business"

for d in [UNIVARIATE_DIR, PAIRWISE_DIR, DIMENSIONALITY_DIR, BUSINESS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _to_json_safe(obj):
    """Convert non-JSON-serializable objects to string."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def main(target: str = "churn"):
    """Perform EDA"""
    df = load_dataset()

    # Summary
    summary = summarize(df, target=target)
    summary["outliers"] = detect_outliers(df)
    summary["missingness"] = missingness_summary(df)
    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(summary), f, indent=2)

    # Feature screening
    recs = find_unnecessary_columns(df, target=target)
    with open(REPORTS_DIR / "eda_feature_screening.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(recs), f, indent=2)

    # Plots
    plot_univariate(df, target=target, outdir=UNIVARIATE_DIR)
    plot_pairwise_interactions(df, target=target, outdir=PAIRWISE_DIR)
    missingness_heatmap(df, outdir=REPORTS_DIR / "plots" / "missingness")

    # Dimensionality reduction
    plot_pca(df, target=target, outdir=DIMENSIONALITY_DIR / "pca")
    plot_tsne(df, target=target, outdir=DIMENSIONALITY_DIR / "tsne")
    plot_umap(df, target=target, outdir=DIMENSIONALITY_DIR / "umap")

    # Business reports
    churn_driver_waterfall(df, target=target, outdir=BUSINESS_DIR)
    clv_based_analysis(
        df, clv_col="monthly_charges", target=target, outdir=BUSINESS_DIR
    )
    executive_summary(summary_dict=summary, recs_dict=recs, outdir=BUSINESS_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="churn")
    args = parser.parse_args()
    main(target=args.target)
