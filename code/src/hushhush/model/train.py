from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weak supervision rule to label strong candidates.
    """
    df = df.copy()
    df["label"] = (
        (df["followers"] >= 50)
        & (df["public_repos_reported"] >= 5)
        & (df["total_stars"] >= 20)
    ).astype(int)
    return df

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf/-inf, fill NaNs, and keep all numeric columns safe for ML.
    """
    X = X.copy()

    # Replace infinite values (can happen with ratios)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with 0 (safe default for counts/flags/ratios in our case)
    X = X.fillna(0)

    return X



def train_model() -> None:
    df = pd.read_csv(PROCESSED_DIR / "github_features.csv")

    df = create_label(df)

    X = df.drop(columns=["username", "label"])
    X = clean_features(X)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_DIR / "github_model.joblib")
    print("\nModel saved to models/github_model.joblib")


if __name__ == "__main__":
    train_model()
