# code/scripts/cluster_label.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Keep this aligned with your refined CSV column names.
# These are the same ones you showed in your screenshot and earlier messages.
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


def cluster_and_label(df: pd.DataFrame, k: int, strong_metric: str = "log_total_stars") -> pd.DataFrame:
    """
    Option 1 approach:
    - Select CLUSTER_FEATURES
    - Fill NaN/inf safely
    - StandardScaler
    - KMeans
    - Label = 1 for the 'strong cluster' (highest avg strong_metric)
    """
    df = df.copy()

    # Validate required columns
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    if strong_metric not in df.columns:
        raise ValueError(f"strong_metric '{strong_metric}' is missing from CSV.")

    # Prepare X
    X = df[CLUSTER_FEATURES].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    # Scale then cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Pick "strong" cluster by highest avg strong_metric
    cluster_strength = df.groupby("cluster")[strong_metric].mean().sort_values(ascending=False)
    strong_cluster = int(cluster_strength.index[0])

    # Label convention:
    # 1 = STRONG, 0 = WEAK
    df["label"] = (df["cluster"] == strong_cluster).astype(int)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="KMeans clustering + weak labels (Option 1: StandardScaler only).")
    parser.add_argument("--in-csv", required=True, help="Input CSV (refined features).")
    parser.add_argument("--out-csv", required=True, help="Output CSV with cluster + label columns.")
    parser.add_argument("-k", type=int, default=3, help="Number of clusters (default: 3).")
    parser.add_argument(
        "--strong-metric",
        default="log_total_stars",
        help="Metric used to select the 'strong' cluster (default: log_total_stars).",
    )

    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    df_out = cluster_and_label(df, k=args.k, strong_metric=args.strong_metric)

    # Print sanity summaries (super useful for debugging)
    print("✅ Clustering complete.")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"k = {args.k}")
    print(f"Strong metric: {args.strong_metric}")
    print("\nCluster counts:")
    print(df_out["cluster"].value_counts().sort_index().to_string())
    print("\nLabel counts (1=STRONG, 0=WEAK):")
    print(df_out["label"].value_counts().sort_index().to_string())

    # Also show which cluster was chosen as strong and its avg metric
    cluster_means = df_out.groupby("cluster")[args.strong_metric].mean().sort_values(ascending=False)
    strong_cluster = int(cluster_means.index[0])
    print(f"\nStrong cluster = {strong_cluster} (highest mean {args.strong_metric})")
    print("\nMean strong-metric per cluster:")
    print(cluster_means.to_string())

    df_out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
