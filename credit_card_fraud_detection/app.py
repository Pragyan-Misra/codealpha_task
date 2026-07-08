import os
import sys
import pickle
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = "fraud-detection"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_models.pkl")


# ── Bundle loading ────────────────────────────────────────────────────────────
# FIX: Guard against missing pkl so the app gives a clear error instead of
#      crashing silently on startup.
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. "
            "Run train_model.py --data <csv> first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


BUNDLE = load_bundle()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def prepare_upload(file, feature_names, scaler):
    """
    Accept either a Werkzeug FileStorage object or a file path string.
    Returns (X_scaled: np.ndarray, n_rows: int).

    FIX: The original code sometimes received a DataFrame instead of a file
         object, causing 'DataFrame has no attribute read'.  This function
         now handles FileStorage, file paths, and DataFrames uniformly.
    """
    if isinstance(file, pd.DataFrame):
        df = file.copy()
    elif isinstance(file, str):
        df = pd.read_csv(file)
    else:
        # Werkzeug FileStorage — read from the stream
        df = pd.read_csv(file.stream)

    # Drop the label column if the user uploaded a labelled dataset
    df = df.drop(columns=["Class"], errors="ignore")

    # ── Column alignment ──────────────────────────────────────────────────
    # FIX: Uploaded CSV may have different columns from the training set.
    #      Add missing columns as zeros; drop extra columns; reorder to match.
    missing = set(feature_names) - set(df.columns)
    if missing:
        for col in missing:
            df[col] = 0.0

    extra = set(df.columns) - set(feature_names)
    if extra:
        df = df.drop(columns=list(extra))

    df = df[feature_names]   # enforce exact column order

    # Fill any NaNs introduced by missing columns or bad data
    df.fillna(df.median(numeric_only=True), inplace=True)

    n_rows = len(df)
    X_scaled = scaler.transform(df.values)
    return X_scaled, n_rows


def iforest_predict(model, X_scaled):
    """
    Map IsolationForest output to fraud labels.
    +1 (normal) → 0 (legit),  -1 (anomaly) → 1 (fraud)
    """
    raw = model.predict(X_scaled)
    return np.where(raw == -1, 1, 0)


# ── Prediction logic ──────────────────────────────────────────────────────────
def run_predictions(file):
    scaler        = BUNDLE["scaler"]
    feature_names = BUNDLE["feature_names"]
    models_meta   = BUNDLE["models"]

    # FIX: Always pass the raw file object — never pre-read it into a DataFrame
    #      here, because prepare_upload handles that internally.
    X_scaled, n_rows = prepare_upload(file, feature_names, scaler)

    # Clip to prevent extreme values from distorting model outputs
    X_scaled = np.clip(X_scaled, -5, 5)

    results = []
    primary_fraud = primary_legit = None   # FIX: use None sentinel, not 0

    for name, meta in models_meta.items():
        model = meta["model"]

        if name == "Isolation Forest":
            preds = iforest_predict(model, X_scaled)
        else:
            probs     = model.predict_proba(X_scaled)[:, 1]
            # FIX: Use the threshold that was tuned at training time.
            #      Falls back to 0.5 only if the bundle pre-dates this fix.
            threshold = meta.get("threshold", 0.5)
            preds     = (probs >= threshold).astype(int)

        fraud = int(np.sum(preds == 1))
        legit = int(np.sum(preds == 0))

        # FIX: Prefer Random Forest for the summary card; fall back to the
        #      first model that completes if RF is absent.
        if name == "Random Forest" or primary_fraud is None:
            primary_fraud = fraud
            primary_legit = legit

        results.append({
            "name":       name,
            "accuracy":   f"{meta['accuracy'] * 100:.2f}",
            "error_rate": f"{meta['error_rate'] * 100:.2f}",
            "fraud":      fraud,
            "legit":      legit,
        })

    return {
        "total":         n_rows,
        "fraud":         primary_fraud,
        "legit":         primary_legit,
        "model_results": results,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # FIX: Validate that a file was actually attached before processing
    if "file" not in request.files or request.files["file"].filename == "":
        flash("Please upload a CSV file.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.")
        return redirect(url_for("index"))

    # FIX: Wrap in try/except so column mismatches or bad data return a
    #      friendly error page instead of a raw 500 traceback
    try:
        results = run_predictions(file)
    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("index"))

    return render_template("result.html", **results)


if __name__ == "__main__":
    app.run(debug=True)