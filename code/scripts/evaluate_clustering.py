from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


PROCESSED_DIR = Path("data/processed")

# Use the SAME columns you already have available
RAW_FEATURES = [
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


def make_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce correlation by collapsing highly-related columns into a few strong signals.
    This usually improves clustering quality and silhouette.
    """
    out = pd.DataFrame(index=df.index)

    # Popularity: collapse correlated popularity metrics into one axis
    out["popularity_score"] = (
        df["log_total_stars"].fillna(0)
        + df["log_total_forks"].fillna(0)
        + df["log_followers"].fillna(0)
        + 0.5 * df["log_total_watchers"].fillna(0)
    )

    # Activity: recent and visible work signals
    # lower days_since_last_* means MORE active, so we invert with minus
    out["activity_score"] = (
        df["events_PushEvent"].fillna(0)
        + 1.5 * df["events_PullRequestEvent"].fillna(0)
        - 0.02 * df["days_since_last_push"].fillna(0)
        - 0.02 * df["days_since_last_event"].fillna(0)
    )

    # Diversity / quality
    out["diversity_score"] = (
        df["unique_languages"].fillna(0)
        + 2.0 * df["top_language_share"].fillna(0)
        + 2.0 * df["fork_ratio"].fillna(0)
        - 2.0 * df["archived_ratio"].fillna(0)
    )

    # Influence balance
    out["influence_balance"] = df["follower_following_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out


def run_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return model.fit_predict(X)


def run_gmm(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    # full covariance lets clusters be elliptical, often better than KMeans
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    return model.fit_predict(X)


def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", default=str(PROCESSED_DIR / "github_features_refined.csv"))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # Ensure all required columns exist
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---- Two candidate feature sets ----
    X_raw = df[RAW_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_meta = make_meta_features(df)

    # ---- Two candidate scalers ----
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),  # more resistant to outliers
    }

    # ---- Two candidate representation spaces ----
    # no PCA vs PCA(2/3/4)
    pca_dims = [None, 2, 3, 4]

    # ---- Two clustering algorithms ----
    algos = {
        "kmeans": run_kmeans,
        "gmm": run_gmm,
    }

    results = []

    for feat_name, X_df in [("raw", X_raw), ("meta", X_meta)]:
        for scaler_name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X_df)

            for p in pca_dims:
                if p is None:
                    X_rep = X_scaled
                    rep_name = "no_pca"
                else:
                    pca = PCA(n_components=p, random_state=args.seed)
                    X_rep = pca.fit_transform(X_scaled)
                    rep_name = f"pca_{p}"

                for algo_name, algo_fn in algos.items():
                    for k in range(args.min_k, args.max_k + 1):
                        labels = algo_fn(X_rep, k, args.seed)

                        # if all labels same, skip
                        if len(set(labels)) < 2:
                            continue

                        metrics = evaluate(X_rep, labels)

                        results.append({
                            "features": feat_name,
                            "scaler": scaler_name,
                            "representation": rep_name,
                            "algo": algo_name,
                            "k": k,
                            **metrics
                        })

    res = pd.DataFrame(results)
    if res.empty:
        raise RuntimeError("No valid clustering results produced.")

    # Sort by best silhouette, then best calinski, then lowest davies (lower is better)
    res_sorted = res.sort_values(
        by=["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True]
    )

    print("\nTop 15 clustering configs (best first):\n")
    print(res_sorted.head(15).to_string(index=False))

    best = res_sorted.iloc[0]
    print("\n✅ BEST CONFIG FOUND:\n")
    print(best.to_string())

    if args.plot:
        # quick plot of silhouette vs k for the BEST pipeline family
        mask = (
            (res["features"] == best["features"]) &
            (res["scaler"] == best["scaler"]) &
            (res["representation"] == best["representation"]) &
            (res["algo"] == best["algo"])
        )
        sub = res[mask].sort_values("k")
        plt.plot(sub["k"], sub["silhouette"])
        plt.title("Silhouette vs k (best pipeline family)")
        plt.xlabel("k")
        plt.ylabel("silhouette")
        plt.show()


if __name__ == "__main__":
    main()
