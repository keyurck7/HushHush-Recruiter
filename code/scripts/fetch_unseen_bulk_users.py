from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


API = "https://api.github.com"


def gh_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def backoff_sleep(resp: requests.Response) -> None:
    if resp.status_code != 403:
        return
    reset = resp.headers.get("X-RateLimit-Reset")
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining == "0" and reset:
        wait = int(reset) - int(time.time()) + 5
        wait = max(wait, 5)
        print(f"⏳ Rate limit hit. Sleeping {wait}s...")
        time.sleep(wait)


def get_json(url: str, params: Optional[dict] = None) -> dict | list:
    resp = requests.get(url, headers=gh_headers(), params=params, timeout=30)
    if resp.status_code == 403:
        backoff_sleep(resp)
        resp = requests.get(url, headers=gh_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_paginated(url: str, per_page: int, max_pages: int) -> List[dict]:
    out: List[dict] = []
    for page in range(1, max_pages + 1):
        data = get_json(url, params={"per_page": per_page, "page": page})
        if not data:
            break
        if isinstance(data, list):
            out.extend(data)
        else:
            # safety: some endpoints might return dict
            out.append(data)
        time.sleep(0.1)
    return out


def save_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch unseen GitHub user bundles into data/raw/github_unseen/")
    parser.add_argument("--in", dest="in_path", default="code/data/usernames_unseen.txt", help="Input usernames file")
    parser.add_argument("--raw-root", default="data/raw/github_unseen", help="Output raw root folder")
    parser.add_argument("--limit", type=int, default=3000, help="Max usernames to fetch from file")
    parser.add_argument("--max-repo-pages", type=int, default=3, help="Max pages of repos (100 per page)")
    parser.add_argument("--max-event-pages", type=int, default=3, help="Max pages of events (100 per page)")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep between users")
    parser.add_argument("--skip-existing", action="store_true", help="Skip user if folder already exists")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    raw_root = Path(args.raw_root)

    usernames = [u.strip() for u in in_path.read_text(encoding="utf-8").splitlines() if u.strip()]
    usernames = usernames[: args.limit]

    print(f"Usernames loaded: {len(usernames)}")
    print(f"Raw root: {raw_root}")

    ok = 0
    skipped = 0
    failed = 0

    for i, username in enumerate(usernames, start=1):
        user_dir = raw_root / username
        if args.skip_existing and user_dir.exists():
            skipped += 1
            continue

        try:
            user = get_json(f"{API}/users/{username}")
            repos = get_paginated(f"{API}/users/{username}/repos", per_page=100, max_pages=args.max_repo_pages)
            events = get_paginated(f"{API}/users/{username}/events/public", per_page=100, max_pages=args.max_event_pages)

            save_json(user_dir / "user.json", user)
            save_json(user_dir / "repos.json", repos)
            save_json(user_dir / "events_public.json", events)

            ok += 1
            if ok % 50 == 0:
                print(f"✅ fetched {ok} users (processed {i}/{len(usernames)})")

        except Exception as e:
            failed += 1
            print(f"❌ {username} failed: {e}")

        time.sleep(args.sleep)

    print("\nDone.")
    print(f"✅ ok={ok}  ⏭️ skipped={skipped}  ❌ failed={failed}")


if __name__ == "__main__":
    main()
