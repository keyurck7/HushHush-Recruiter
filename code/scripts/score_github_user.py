from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from hushhush.config import ensure_dirs
from hushhush.ingest.github_fetch import GithubFetcher
from hushhush.features.github_features import bundle_to_feature_row


DEFAULT_MODEL_PATH = Path("models/github_model.joblib")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a GitHub candidate using the trained model.")
    p.add_argument("--user", required=True, help="GitHub username to score")
    p.add_argument("--refresh", action="store_true", help="Refresh GitHub API data even if cached")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to trained model .joblib")
    p.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for SELECTED")
    p.add_argument("--json", action="store_true", help="Print output as JSON")
    return p.parse_args()


def predict_score(model, X: pd.DataFrame) -> float:
    # Prefer probability if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return float(proba[0][1])
    # Fallback: decision_function or predict
    if hasattr(model, "decision_function"):
        val = float(model.decision_function(X)[0])
        # Map decision value into (0,1) roughly (not perfect, but ok for fallback)
        return 1.0 / (1.0 + pow(2.718281828, -val))
    return float(model.predict(X)[0])


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Fetch raw bundle (cached unless refresh)
    fetcher = GithubFetcher()
    bundle = fetcher.fetch_user_bundle(args.user, refresh=args.refresh)

    # Build single-row feature dataframe
    X = bundle_to_feature_row(bundle)

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}. Train first (train_model.py).", file=sys.stderr)
        return 2

    model = joblib.load(model_path)

    # Score
    score = predict_score(model, X)
    decision = "SELECTED" if score >= args.threshold else "REJECTED"

    if args.json:
        print(json.dumps({
            "username": args.user,
            "score": round(score, 4),
            "threshold": args.threshold,
            "decision": decision,
        }, indent=2))
        return 0

    print("=== HushHush Recruiter: GitHub Score ===")
    print(f"User: {args.user}")
    print(f"Score (P[selected]): {score:.4f}")
    print(f"Decision (threshold={args.threshold:.2f}): {decision}")
    print("")
    print("Feature snapshot:")
    for col in list(X.columns)[:10]:  # show first 10 features only
        print(f"- {col}: {X.iloc[0][col]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
