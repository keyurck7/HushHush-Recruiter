from __future__ import annotations

import argparse
import sys

from hushhush.ingest.github_fetch import GithubFetcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache GitHub data for a candidate."
    )
    parser.add_argument(
        "--user",
        required=True,
        help="GitHub username to fetch",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch from GitHub API",
    )

    args = parser.parse_args()

    fetcher = GithubFetcher()

    try:
        bundle = fetcher.fetch_user_bundle(
            args.user,
            refresh=args.refresh,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    user = bundle["user"]
    repos = bundle["repos"]
    events = bundle["events"]

    print("\n=== GitHub Candidate Summary ===")
    print(f"Username        : {user.get('login')}")
    print(f"Public Repos    : {user.get('public_repos')}")
    print(f"Followers       : {user.get('followers')}")
    print(f"Following       : {user.get('following')}")
    print(f"Repos fetched   : {len(repos)}")
    print(f"Events fetched  : {len(events)}")
    print("\nRaw data cached under: data/raw/github/")
    print("================================\n")


if __name__ == "__main__":
    main()
