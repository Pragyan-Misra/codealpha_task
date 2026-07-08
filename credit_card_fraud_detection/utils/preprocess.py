"""
utils/preprocess.py
-------------------
Shared preprocessing helpers used by app.py at prediction time.
"""

import numpy as np
import pandas as pd


def prepare_upload(file, feature_names, scaler):
    """
    Accept a Werkzeug FileStorage object, a file-path string, or a DataFrame.
    Returns (X_scaled: np.ndarray, n_rows: int).

    FIX: Previously, any columns missing from the uploaded CSV were silently
    filled with 0.0. After scaling, all-zero rows map to the statistical mean
    of a non-fraud transaction, so every row looked maximally "normal" to the
    model → always 0 fraud predicted on any non-Kaggle dataset.

    The correct behaviour is to REJECT uploads whose columns don't match the
    training features, with a clear message explaining what's expected.
    """
    if isinstance(file, pd.DataFrame):
        df = file.copy()
    elif isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file.stream)

    # Drop label column if the user uploaded a labelled dataset
    df = df.drop(columns=["Class"], errors="ignore")

    uploaded_cols = set(df.columns)
    expected_cols = set(feature_names)

    missing = expected_cols - uploaded_cols

    # FIX: Hard reject if more than 10% of expected columns are missing.
    # Silently zeroing large numbers of columns produces garbage predictions
    # (always 0 fraud) with no warning to the user.
    if len(missing) > max(1, int(0.1 * len(feature_names))):
        missing_sample = sorted(missing)[:10]
        raise ValueError(
            f"Uploaded CSV is missing {len(missing)} expected feature columns "
            f"(e.g. {missing_sample}{'...' if len(missing) > 10 else ''}). "
            f"This model was trained on the Kaggle Credit Card Fraud dataset "
            f"(features: Time, V1–V28, Amount). Upload a CSV with those same "
            f"columns, or retrain the model on your dataset."
        )

    # For minor mismatches (≤10% of columns) fill with column median
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    extra = set(df.columns) - expected_cols
    if extra:
        df = df.drop(columns=list(extra))

    df = df[feature_names]
    df.fillna(df.median(numeric_only=True), inplace=True)

    n_rows   = len(df)
    X_scaled = scaler.transform(df.values)
    return X_scaled, n_rows


def iforest_predict(model, X_scaled):
    """
    Map IsolationForest output (+1 / -1) to fraud labels (0 / 1).
    +1 = normal  → 0 (legit)
    -1 = anomaly → 1 (fraud)
    """
    raw = model.predict(X_scaled)
    return np.where(raw == -1, 1, 0)