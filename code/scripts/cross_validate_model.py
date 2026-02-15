from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROCESSED_DIR = Path("data/processed")

# Same features used in training
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

def main():
    df = pd.read_csv(PROCESSED_DIR / "github_features_clustered.csv")

    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print("Cross-Validation Accuracy:", accuracy)
    print("Mean Accuracy:", accuracy.mean())

    print("\nCross-Validation AUC:", auc)
    print("Mean AUC:", auc.mean())


if __name__ == "__main__":
    main()
