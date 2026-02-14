from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = [
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
    parser = argparse.ArgumentParser(description="Cluster GitHub candidates and assign pseudo-labels.")
    parser.add_argument("--in-csv", type=str, required=True, help="Input features CSV (refined or refined_noisy).")
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV with cluster + label columns.")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (start with 3).")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Use only the selected columns
    X = df[CLUSTER_FEATURES].copy()

    # Make sure everything is numeric
    for c in CLUSTER_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Fill NaNs (simple + safe for KMeans)
    X = X.fillna(0)

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Cluster
    km = KMeans(n_clusters=args.k, random_state=args.random_state, n_init=10)
    clusters = km.fit_predict(Xs)

    df["cluster"] = clusters

    # OPTIONAL but useful for your pipeline:
    # Convert clusters to a binary "label" (strong vs not)
    # Rule: the "strong" cluster = the one with highest mean log_total_stars
    cluster_strength = (
        df.groupby("cluster")["log_total_stars"].mean().sort_values(ascending=False)
    )
    strong_cluster = int(cluster_strength.index[0])
    df["label"] = (df["cluster"] == strong_cluster).astype(int)

    df.to_csv(out_path, index=False)

    print("✅ Clustering complete.")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"k = {args.k}")
    print(f"Strong cluster = {strong_cluster} (highest avg log_total_stars)")


if __name__ == "__main__":
    main()
