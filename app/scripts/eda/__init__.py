"""
EDA subpackage for churn analysis.

This package provides modules for:
- Loading datasets
- Summarization and feature screening
- Outlier detection
- Plotting (univariate, pairwise, correlations)
- Missingness analysis
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Business-oriented analysis and reporting
"""

from .business import (
    churn_driver_waterfall,
    clv_based_analysis,
    cohort_analysis,
    executive_summary,
    funnel_analysis,
    segmentation_analysis,
)
from .dimensionality import plot_pca, plot_tsne, plot_umap
from .loader import load_dataset
from .missingness import missingness_heatmap, missingness_summary
from .outliers import detect_outliers
from .plots import plot_correlations, plot_pairwise_interactions, plot_univariate
from .summarization import find_unnecessary_columns, summarize
