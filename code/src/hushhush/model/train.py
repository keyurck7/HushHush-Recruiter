from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---- Labeled dataset produced AFTER clustering ----
LABELED_CSV = PROCESSED_DIR / "github_features_clustered.csv"
# Change this filename only if your clustering output has a different name.

# ---- Label convention (standard) ----
# label = 1  -> STRONG candidate
# label = 0  -> WEAK candidate

# ---- Curated, less-correlated feature set for classification ----
FEATURE_COLS = [
    "log_total_stars",
    "log_total_forks",
    "log_total_watchers",
    "log_followers",
    "follower_following_ratio",
    "events_PushEvent",
    "events_PullRequestEvent",
    "days_since_last_push",
    "days_since_last_event",
    "unique_languages",
    "top_language_share",
    "fork_ratio",
    "archived_ratio",
]


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf/-inf, fill NaNs, keep numeric data safe for ML.
    """
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X


def train_model() -> None:
    if not LABELED_CSV.exists():
        raise FileNotFoundError(
            f"Could not find labeled dataset: {LABELED_CSV}. "
            f"Run clustering first to generate it."
        )

    df = pd.read_csv(LABELED_CSV)

    if "label" not in df.columns:
        raise ValueError("Your labeled CSV does not contain a 'label' column. Generate labels first.")

    df["label"] = df["label"].astype(int)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLS].copy()
    X = clean_features(X)
    y = df["label"]

    # Stratify keeps class balance consistent across train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Pipeline: scale -> logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nLabel convention: 1 = STRONG, 0 = WEAK")
    print("\nLabel distribution (full dataset):")
    print(y.value_counts())

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    out_path = MODEL_DIR / "github_classifier.joblib"
    joblib.dump(model, out_path)
    print(f"\n✅ Model saved to {out_path}")


if __name__ == "__main__":
    train_model()
