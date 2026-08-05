import numpy as np
import pandas as pd


def build_feature_matrix(df):
    """
    Build features from daily sales data.

    Input:
        date
        total_sales

    Output:
        DataFrame with engineered features
    """

    df = df.copy()

    # -----------------------
    # Calendar Features
    # -----------------------

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # -----------------------
    # Weekend
    # -----------------------

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    # -----------------------
    # Month Cyclical Encoding
    # -----------------------

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # -----------------------
    # Lag Features
    # -----------------------

    df["lag_1"] = df["total_sales"].shift(1)
    df["lag_7"] = df["total_sales"].shift(7)
    df["lag_30"] = df["total_sales"].shift(30)

    # -----------------------
    # Rolling Statistics
    # -----------------------

    df["rolling_mean_7"] = (
        df["total_sales"]
        .rolling(7)
        .mean()
    )

    df["rolling_mean_30"] = (
        df["total_sales"]
        .rolling(30)
        .mean()
    )

    df["rolling_std_7"] = (
        df["total_sales"]
        .rolling(7)
        .std()
    )

    # -----------------------
    # Trend
    # -----------------------

    df["trend"] = np.arange(len(df))

    # -----------------------
    # Remove NaN rows
    # -----------------------

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "day",
        "month",
        "year",
        "dayofweek",
        "quarter",
        "dayofyear",
        "weekofyear",
        "is_weekend",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
        "trend",
    ]

    return df, feature_cols


if __name__ == "__main__":

    from preprocessing import load_and_clean, aggregate_daily

    df = load_and_clean()

    daily = aggregate_daily(df)

    feature_df, cols = build_feature_matrix(daily)

    print(feature_df.head())

    print("\nFeatures:")

    print(cols)