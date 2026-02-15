from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import joblib


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/github_classifier.joblib")


def main():
    print("Loading refined unseen dataset...")
    df = pd.read_csv(PROCESSED_DIR / "github_features_unseen_refined.csv")

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model does not contain feature_names_in_. Please confirm training pipeline.")

    feature_cols = list(model.feature_names_in_)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Probability of STRONG (label = 1)
    probs = model.predict_proba(X)[:, 1]

    df["strong_probability"] = probs
    df = df.sort_values("strong_probability", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    out_path = PROCESSED_DIR / "final_ranked_candidates.csv"
    df.to_csv(out_path, index=False)

    print("\nTop 10 Candidates:\n")
    print(df[["rank", "username", "strong_probability"]].head(10).to_string(index=False))

    print(f"\nSaved ranked results to: {out_path}")


if __name__ == "__main__":
    main()
