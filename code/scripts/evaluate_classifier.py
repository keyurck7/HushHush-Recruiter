from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/github_classifier.joblib")

# Same feature list used in train.py
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


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(PROCESSED_DIR / "github_features_clustered.csv")

    X = df[FEATURE_COLS].copy()
    y = df["label"]

    # Clean numeric safety
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    # Train/test split (same logic as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Load trained model
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    # Get probability scores
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\nAUC Score: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - GitHub Candidate Classifier")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
