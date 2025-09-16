"""
Business-oriented visualization and reporting for churn analysis.

Includes:
- Waterfall chart of churn drivers
- CLV-based churn analysis
- Executive summary generation
- Cohort, funnel, and segmentation analysis
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Business-Oriented Visualizations
# -----------------------------


def churn_driver_waterfall(df: pd.DataFrame, target: str, outdir: Path):
    """Plot a waterfall chart of positive and negative churn drivers using only numeric columns."""
    outdir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)  # exclude target from correlation features

    if not numeric_cols:
        print("[WARN] No numeric columns available for churn driver waterfall.")
        return

    # Compute correlation with target
    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        corr = df[numeric_cols + [target]].corr()[target].drop(target).sort_values()
    else:
        print(
            f"[WARN] Target '{target}' is non-numeric, using numeric columns only for plot."
        )
        corr = df[numeric_cols].sum().sort_values()  # fallback numeric proxy

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.bar(
        corr.index,
        corr.values,
        color=["red" if x < 0 else "green" for x in corr.values],
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Churn Driver Waterfall")
    plt.ylabel("Correlation with churn (numeric proxy)")
    plt.tight_layout()
    plt.savefig(outdir / "churn_driver_waterfall.png", dpi=300)
    plt.close()


def clv_based_analysis(df: pd.DataFrame, clv_col: str, target: str, outdir: Path):
    """Analyze churn by customer lifetime value segments with fallback column."""
    outdir.mkdir(parents=True, exist_ok=True)

    if clv_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns
        if not numeric_cols.any():
            print("[WARN] No numeric columns available for CLV analysis, skipping.")
            return
        clv_col = numeric_cols[0]
        print(f"[INFO] '{clv_col}' used for CLV analysis instead of missing column.")

    df = df.copy()
    df["clv_segment"] = pd.qcut(
        df[clv_col], q=4, labels=["Low", "Medium", "High", "Top"]
    )

    # Ensure target column exists and is numeric-friendly
    if target in df.columns:
        df[target] = df[target].apply(lambda x: 1 if str(x).lower() == "yes" else 0)

    summary = df.groupby("clv_segment")[target].mean()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    summary.plot(kind="bar", color="skyblue")
    plt.title("Churn by CLV Segment")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(outdir / "clv_churn_analysis.png", dpi=300)
    plt.close()


def executive_summary(summary_dict: dict, recs_dict: dict, outdir: Path):
    """Generate concise executive summary JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    summary_text = {
        "key_findings": list(summary_dict.keys()),
        "feature_recommendations": recs_dict,
    }
    with open(outdir / "executive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_text, f, indent=2)


# -----------------------------
# Domain-specific / Behavioral Analysis
# -----------------------------


def cohort_analysis(
    df: pd.DataFrame,
    account_length_col: str = "account_length",
    _target: str = "churn",
    outdir: Optional[Path] = None,
):
    """Cohort analysis using account_length as a proxy for signup date."""
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    bins = [0, 12, 24, 48, 1000]
    labels = ["0-12m", "13-24m", "25-48m", "48m+"]
    df["cohort"] = pd.cut(df[account_length_col], bins=bins, labels=labels, right=False)
    cohort_data = df.groupby("cohort")[_target].apply(lambda x: (x == "yes").mean())

    if outdir:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=cohort_data.index, y=cohort_data.values, palette="coolwarm")
        plt.title("Cohort Analysis by Account Length")
        plt.ylabel("Churn Rate")
        plt.xlabel("Account Length Cohort")
        plt.tight_layout()
        plt.savefig(outdir / "cohort_analysis.png", dpi=300)
        plt.close()

    return cohort_data


def funnel_analysis(
    df: pd.DataFrame,
    steps_cols: Optional[List[str]] = None,
    _target: str = "churn",
    outdir: Optional[Path] = None,
):
    """Funnel analysis showing drop-offs from one stage to another."""
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    if steps_cols is None:
        steps_cols = [
            "international_plan",
            "voice_mail_plan",
            "number_customer_service_calls",
        ]

    funnel_counts = []
    total = len(df)
    current_df = df.copy()

    for step in steps_cols:
        if current_df[step].dtype == "object":
            current_df = current_df[current_df[step].astype(str).str.lower() == "yes"]
        else:
            current_df = current_df[current_df[step] > 0]
        funnel_counts.append(len(current_df))

    funnel_df = pd.DataFrame(
        {
            "step": steps_cols,
            "count": funnel_counts,
            "conversion_rate": [c / total for c in funnel_counts],
        }
    )

    if outdir:
        plt.figure(figsize=(6, 4))
        sns.barplot(x="conversion_rate", y="step", data=funnel_df, palette="viridis")
        plt.xlabel("Conversion Rate")
        plt.ylabel("Funnel Step")
        plt.title("Customer Funnel Analysis")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(outdir / "funnel_analysis.png", dpi=300)
        plt.close()

    print("[INFO] Funnel analysis completed")
    return funnel_df


def segmentation_analysis(
    df: pd.DataFrame,
    segment_cols: Optional[List[str]] = None,
    target: str = "churn",
    outdir: Optional[Path] = None,
    n_clusters: int = 4,
):
    """Customer segmentation using KMeans and churn analysis."""
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    if segment_cols is None:
        segment_cols = [
            "total_day_minutes",
            "total_eve_minutes",
            "total_night_minutes",
            "total_intl_minutes",
            "number_vmail_messages",
            "number_customer_service_calls",
            "total_day_charge",
            "total_eve_charge",
            "total_night_charge",
            "total_intl_charge",
        ]

    df_seg = df.copy()
    x_features = df_seg[segment_cols].values
    x_scaled_features = StandardScaler().fit_transform(x_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_seg["segment"] = kmeans.fit_predict(x_scaled_features)

    churn_summary = (
        df_seg.groupby("segment")[target]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    if outdir:
        plt.figure(figsize=(8, 5))
        churn_summary.plot(kind="bar", stacked=True, colormap="coolwarm", ax=plt.gca())
        plt.title("Churn Rate per Customer Segment")
        plt.ylabel("Proportion")
        plt.xlabel("Segment")
        plt.tight_layout()
        plt.savefig(outdir / "segmentation_churn_analysis.png", dpi=300)
        plt.close()

    print("[INFO] Segmentation analysis completed")
    return df_seg
