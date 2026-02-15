from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Set

import requests


API = "https://api.github.com"


def gh_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def load_seen(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def backoff_sleep(resp: requests.Response) -> None:
    # Basic handling for rate limits
    if resp.status_code != 403:
        return
    # GitHub rate limit headers
    reset = resp.headers.get("X-RateLimit-Reset")
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining == "0" and reset:
        wait = int(reset) - int(time.time()) + 5
        wait = max(wait, 5)
        print(f"⏳ Rate limit hit. Sleeping {wait}s...")
        time.sleep(wait)


def fetch_users_page(since: int, per_page: int) -> List[dict]:
    url = f"{API}/users"
    params = {"since": since, "per_page": per_page}
    resp = requests.get(url, headers=gh_headers(), params=params, timeout=30)
    if resp.status_code == 403:
        backoff_sleep(resp)
        resp = requests.get(url, headers=gh_headers(), params=params, timeout=30)

    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect unseen GitHub usernames (filters out seen list).")
    parser.add_argument("--seen", default="code/data/usernames.txt", help="Seen usernames file (one per line)")
    parser.add_argument("--out", default="code/data/usernames_unseen.txt", help="Output file for unseen usernames")
    parser.add_argument("--target", type=int, default=1000, help="How many unseen usernames to collect")
    parser.add_argument("--since", type=int, default=0, help="Starting 'since' user id (0 = from beginning)")
    parser.add_argument("--per-page", type=int, default=100, help="Users per page (max 100)")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between API calls")
    args = parser.parse_args()

    seen_path = Path(args.seen)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = load_seen(seen_path)
    unseen: List[str] = []

    since = args.since
    print(f"Seen loaded: {len(seen)} usernames")
    print(f"Collecting {args.target} unseen usernames...")

    while len(unseen) < args.target:
        page = fetch_users_page(since=since, per_page=args.per_page)
        if not page:
            print("No more users returned by API. Stopping.")
            break

        for u in page:
            login = u.get("login")
            uid = u.get("id")
            if uid is not None:
                since = int(uid)
            if not login:
                continue

            if login not in seen:
                unseen.append(login)
                seen.add(login)

                if len(unseen) % 100 == 0:
                    print(f"✅ Unseen collected: {len(unseen)}")

                if len(unseen) >= args.target:
                    break

        time.sleep(args.sleep)

    out_path.write_text("\n".join(unseen) + "\n", encoding="utf-8")
    print(f"\n✅ Saved {len(unseen)} unseen usernames to: {out_path}")


if __name__ == "__main__":
    main()
