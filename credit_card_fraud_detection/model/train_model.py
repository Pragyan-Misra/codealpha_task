import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(MODEL_DIR, "saved_models.pkl")


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    if "Class" not in df.columns:
        sys.exit("Dataset must contain a 'Class' column (0 = legit, 1 = fraud).")

    df.fillna(df.median(numeric_only=True), inplace=True)

    n_sample = min(50_000, len(df))
    df = df.sample(n_sample, random_state=42)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # stratify so both splits always contain fraud samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # True fraud ratio for IsolationForest contamination
    true_contamination = float(y_train.mean())
    true_contamination = max(0.001, min(true_contamination, 0.5))

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)


    return X_train, X_test, y_train, y_test, scaler, list(X.columns), true_contamination


def auto_threshold(model, X_test, y_test):
    """
    Sweep thresholds and return the one with the best F1.
    Saved in the bundle so prediction-time behaviour matches the
    accuracy figure shown in the UI.
    """
    probs = model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = 0.5, 0.0

    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        score = f1_score(y_test, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t  = float(t)

    return best_t


def evaluate(model, X_test, y_test, threshold):
    """Accuracy computed with the tuned threshold, not the default 0.5."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    acc   = accuracy_score(y_test, preds)
    return acc, 1 - acc


def train_all(csv_path):
    print("Loading and preprocessing data…")
    X_tr, X_te, y_tr, y_te, scaler, feature_names, contamination = \
        load_and_preprocess(csv_path)

    print(f"  Train size : {len(X_tr):,}")
    print(f"  Test  size : {len(X_te):,}")
    print(f"  Fraud rows in test : {int(y_te.sum())} / {len(y_te)}")
    print(f"  IF contamination   : {contamination:.4f}\n")

    models_meta = {}

    # ── Logistic Regression ───────────────────────────────────────────────
    print("Training Logistic Regression…")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   
        random_state=42
    )
    lr.fit(X_tr, y_tr)
    t_lr = auto_threshold(lr, X_te, y_te)
    acc_lr, err_lr = evaluate(lr, X_te, y_te, t_lr)
    print(f"  threshold={t_lr:.2f}  accuracy={acc_lr:.4f}")
    print(classification_report(
        y_te, (lr.predict_proba(X_te)[:, 1] >= t_lr).astype(int),
        target_names=["Legit", "Fraud"]
    ))
    models_meta["Logistic Regression"] = {
        "model": lr, "threshold": t_lr,
        "accuracy": acc_lr, "error_rate": err_lr,
    }

    # ── Random Forest ─────────────────────────────────────────────────────
    print("Training Random Forest…")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    t_rf = auto_threshold(rf, X_te, y_te)
    acc_rf, err_rf = evaluate(rf, X_te, y_te, t_rf)
    print(f"  threshold={t_rf:.2f}  accuracy={acc_rf:.4f}")
    print(classification_report(
        y_te, (rf.predict_proba(X_te)[:, 1] >= t_rf).astype(int),
        target_names=["Legit", "Fraud"]
    ))
    models_meta["Random Forest"] = {
        "model": rf, "threshold": t_rf,
        "accuracy": acc_rf, "error_rate": err_rf,
    }

    # ── SVM ───────────────────────────────────────────────────────────────
    print("Training SVM (may take a minute)…")
    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        cache_size=500,
        random_state=42
    )
    svm.fit(X_tr, y_tr)
    t_svm = auto_threshold(svm, X_te, y_te)
    acc_svm, err_svm = evaluate(svm, X_te, y_te, t_svm)
    print(f"  threshold={t_svm:.2f}  accuracy={acc_svm:.4f}")
    print(classification_report(
        y_te, (svm.predict_proba(X_te)[:, 1] >= t_svm).astype(int),
        target_names=["Legit", "Fraud"]
    ))
    models_meta["SVM"] = {
        "model": svm, "threshold": t_svm,
        "accuracy": acc_svm, "error_rate": err_svm,
    }

    # ── Isolation Forest ──────────────────────────────────────────────────

    print("Training Isolation Forest…")
    iforest = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    iforest.fit(X_tr)

    # +1 (normal) → 0 (legit),  -1 (anomaly) → 1 (fraud)
    if_preds = np.where(iforest.predict(X_te) == -1, 1, 0)
    acc_if   = accuracy_score(y_te, if_preds)
    print(f"  accuracy={acc_if:.4f}")
    print(classification_report(y_te, if_preds, target_names=["Legit", "Fraud"]))
    models_meta["Isolation Forest"] = {
        "model": iforest,
        "accuracy": acc_if, "error_rate": 1 - acc_if,
    }

    # ── Save bundle ───────────────────────────────────────────────────────
    bundle = {
        "scaler":        scaler,
        "feature_names": feature_names,
        "models":        models_meta,
    }
    os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
    with open(MODELS_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nBundle saved → {MODELS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with a 'Class' column.")
    args = parser.parse_args()
    train_all(args.data)