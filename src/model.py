import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


FEATURE_COLS = [
    "month", "quarter", "year",
    "month_sin", "month_cos",
    "lag_1", "lag_3",
    "rolling_mean_3", "rolling_mean_6",
    "is_q4", "trend"
]


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """
    Trains an XGBoost regressor on the provided features and target.

    Args:
        X: feature DataFrame (use FEATURE_COLS)
        y: target Series (monthly sales)

    Returns:
        Fitted XGBRegressor model
    """
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        random_state=42
    )
    model.fit(X, y)
    print("Model trained successfully.")
    return model


def predict(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Generates predictions from a trained model.

    Args:
        model: fitted XGBRegressor
        X: feature DataFrame

    Returns:
        Array of predictions
    """
    return model.predict(X)


def save_model(model: XGBRegressor, filepath: str) -> None:
    """Saves the trained model to disk using joblib."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> XGBRegressor:
    """Loads a saved model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
