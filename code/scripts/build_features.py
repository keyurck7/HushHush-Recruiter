from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from hushhush.config import PROCESSED_DIR, ensure_dirs
from hushhush.features.github_features import GithubFeatureExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GitHub feature table from cached data.")
    p.add_argument(
        "--users",
        nargs="+",
        required=True,
        help="One or more GitHub usernames (space-separated). Example: --users torvalds octocat",
    )
    p.add_argument(
        "--out",
        default="github_features.csv",
        help="Output filename inside data/processed/ (default: github_features.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    extractor = GithubFeatureExtractor()

    rows: List[dict] = []
    for u in args.users:
        rows.append(extractor.extract(u))

    df = pd.DataFrame(rows)

    out_path: Path = PROCESSED_DIR / args.out
    df.to_csv(out_path, index=False)

    print(f"\n✅ Wrote features for {len(df)} user(s) to: {out_path}")
    print(df.head().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
