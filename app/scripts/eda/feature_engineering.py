"""
Advanced Feature Engineering for churn dataset.
"""

import pandas as pd


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral metrics from usage data.
    Example features:
    - average_daily_call_minutes
    - data_usage_std_dev
    - service_downgrade_flags
    """
    df = df.copy()
    if "total_day_minutes" in df.columns and "total_day_calls" in df.columns:
        df["avg_daily_call_minutes"] = df["total_day_minutes"] / (
            df["total_day_calls"].replace(0, 1)
        )

    if "data_usage" in df.columns:
        df["data_usage_std_dev"] = (
            df.groupby("customer_id")["data_usage"].transform("std").fillna(0)
        )

    if "service_downgrade" in df.columns:
        df["service_downgrade_flags"] = df["service_downgrade"].astype(int)

    return df


def create_price_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price and contract-related features.
    Example features:
    - price_sensitivity_index
    - months_until_contract_end
    """
    df = df.copy()
    if "monthly_charge" in df.columns and "total_day_charge" in df.columns:
        df["price_sensitivity_index"] = df["total_day_charge"] / df[
            "monthly_charge"
        ].replace(0, 1)

    if "contract_end_date" in df.columns and "signup_date" in df.columns:
        df["months_until_contract_end"] = (
            (
                pd.to_datetime(df["contract_end_date"])
                - pd.to_datetime(df["signup_date"])
            ).dt.days
            // 30
        ).clip(lower=0)

    return df
