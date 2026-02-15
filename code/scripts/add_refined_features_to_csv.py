from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def to_numeric_safe(df: pd.DataFrame, skip_cols: set[str]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in skip_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_refined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds refined features used by the model:
    - log transforms
    - ratios
    - (keeps everything else intact)
    """
    df = df.copy()

    # Ensure numeric safety
    df = to_numeric_safe(df, skip_cols={"username"})
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---- Log transforms (create only if raw exists) ----
    # NOTE: Use the raw totals in your CSV; names must match exactly.
    log_map = {
        "followers": "log_followers",
        "total_stars": "log_total_stars",
        "total_forks": "log_total_forks",
        "total_watchers": "log_total_watchers",
    }
    for raw, log_col in log_map.items():
        if raw in df.columns and log_col not in df.columns:
            df[log_col] = np.log1p(df[raw])

    # ---- Ratios ----
    if "followers" in df.columns and "following" in df.columns and "follower_following_ratio" not in df.columns:
        df["follower_following_ratio"] = df["followers"] / (df["following"] + 1)

    # Optional helpful ratios (won't hurt, and you can keep them)
    if "total_stars" in df.columns and "public_repos_reported" in df.columns and "stars_per_repo" not in df.columns:
        df["stars_per_repo"] = df["total_stars"] / (df["public_repos_reported"] + 1)

    if "total_forks" in df.columns and "public_repos_reported" in df.columns and "forks_per_repo" not in df.columns:
        df["forks_per_repo"] = df["total_forks"] / (df["public_repos_reported"] + 1)

    # ---- Final cleanup ----
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add refined features (log, ratios) to an existing CSV.")
    parser.add_argument("--in-csv", required=True, help="Input CSV path (existing features).")
    parser.add_argument("--out-csv", required=True, help="Output CSV path (with refined features).")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if "username" not in df.columns:
        raise ValueError("Input CSV must contain a 'username' column.")

    df2 = add_refined_features(df)
    df2.to_csv(out_path, index=False)

    added = [c for c in df2.columns if c not in df.columns]
    print(f"✅ Wrote: {out_path}")
    print(f"✅ Added {len(added)} new columns:")
    for c in added:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
