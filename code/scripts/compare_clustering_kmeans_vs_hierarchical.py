from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Optional dendrogram (requires scipy)
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


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


def clean_X(df: pd.DataFrame) -> np.ndarray:
    X = df[CLUSTER_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.values


def compute_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    # Need at least 2 clusters for metrics
    if len(set(labels)) < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}

    return {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "calinski": float(calinski_harabasz_score(X_scaled, labels)),
        "davies": float(davies_bouldin_score(X_scaled, labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KMeans vs Hierarchical clustering with metrics + plots.")
    parser.add_argument("--in-csv", default="data/processed/github_features_refined.csv", help="Input refined features CSV")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--dendrogram", action="store_true", help="Plot hierarchical dendrogram (requires scipy)")
    parser.add_argument("--dendro-sample", type=int, default=250, help="Sample size for dendrogram (hierarchical is expensive)")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    df = pd.read_csv(in_path)

    # Validate columns
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Preprocess (same for both)
    X = clean_X(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows = []
    ks = list(range(args.k_min, args.k_max + 1))

    for k in ks:
        # ---- KMeans ----
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(X_scaled)
        km_time = time.perf_counter() - t0
        km_m = compute_metrics(X_scaled, km_labels)

        rows.append({
            "algo": "KMeans",
            "k": k,
            "silhouette": km_m["silhouette"],
            "calinski": km_m["calinski"],
            "davies": km_m["davies"],
            "runtime_s": km_time
        })

        # ---- Hierarchical (Agglomerative) ----
        t0 = time.perf_counter()
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        ag_labels = ag.fit_predict(X_scaled)
        ag_time = time.perf_counter() - t0
        ag_m = compute_metrics(X_scaled, ag_labels)

        rows.append({
            "algo": "Hierarchical(Ward)",
            "k": k,
            "silhouette": ag_m["silhouette"],
            "calinski": ag_m["calinski"],
            "davies": ag_m["davies"],
            "runtime_s": ag_time
        })

    res = pd.DataFrame(rows)
    out_csv = Path("data/processed/compare_clustering_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    print("\n✅ Saved metrics table to:", out_csv)
    print("\nTop configs by silhouette (desc):")
    print(res.sort_values("silhouette", ascending=False).head(10).to_string(index=False))

    # ---- Plot helpers ----
    def plot_metric(metric: str, title: str, y_label: str) -> None:
        plt.figure()
        for algo in res["algo"].unique():
            sub = res[res["algo"] == algo].sort_values("k")
            plt.plot(sub["k"], sub[metric], label=algo)
        plt.title(title)
        plt.xlabel("k (number of clusters)")
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    plot_metric("silhouette", "Silhouette Score vs k (Higher is better)", "Silhouette")
    plot_metric("calinski", "Calinski–Harabasz vs k (Higher is better)", "Calinski–Harabasz")
    plot_metric("davies", "Davies–Bouldin vs k (Lower is better)", "Davies–Bouldin")
    plot_metric("runtime_s", "Runtime vs k (Lower is faster)", "Seconds")

    # ---- Optional dendrogram ----
    if args.dendrogram:
        if not SCIPY_OK:
            print("\n⚠️ scipy not available. Install it: pip install scipy")
            return

        # Dendrogram on a sample (otherwise huge + slow)
        n = min(args.dendro_sample, len(df))
        sample = df.sample(n=n, random_state=42)
        Xs = scaler.fit_transform(clean_X(sample))

        plt.figure()
        Z = linkage(Xs, method="ward")
        dendrogram(Z, no_labels=True)
        plt.title(f"Hierarchical Dendrogram (Ward linkage) | sample={n}")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.show()


if __name__ == "__main__":
    main()
