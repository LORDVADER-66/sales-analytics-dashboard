import numpy as np
import pandas as pd


def build_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time-series features from aggregated monthly sales data.
    Adds lag features, rolling averages, cyclical month encoding, and seasonal flags.
    Drops rows with NaN values introduced by lag/rolling windows.

    Args:
        monthly_df: DataFrame with columns [date, sales, orders, quantity, profit]

    Returns:
        Feature-engineered DataFrame ready for model training
    """
    df = monthly_df.copy().sort_values("date").reset_index(drop=True)

    # Basic time features
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # Cyclical encoding — tells the model Dec and Jan are close together
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    df["lag_1"] = df["sales"].shift(1)
    df["lag_3"] = df["sales"].shift(3)

    # Rolling averages
    df["rolling_mean_3"] = df["sales"].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df["sales"].shift(1).rolling(window=6).mean()

    # Seasonal flag
    df["is_q4"] = (df["quarter"] == 4).astype(int)

    # Linear trend index — helps XGBoost capture overall growth
    df["trend"] = np.arange(len(df))

    # Drop rows with NaN from lag/rolling (first 6 rows)
    df = df.dropna().reset_index(drop=True)

    print(f"Feature matrix shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Features: {[c for c in df.columns if c not in ['date', 'sales', 'orders', 'quantity', 'profit']]}")

    return df