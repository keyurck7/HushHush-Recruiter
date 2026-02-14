from __future__ import annotations

import argparse
import os
import time
from typing import List, Set, Optional

import requests


GITHUB_API = "https://api.github.com"
SEARCH_ENDPOINT = f"{GITHUB_API}/search/users"


def github_headers() -> dict:
    """
    Uses GITHUB_TOKEN if available (recommended).
    """
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hushhush-collector",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def handle_rate_limit(resp: requests.Response) -> None:
    """
    If rate-limited, sleep until reset.
    """
    if resp.status_code != 403:
        return

    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")

    # If this 403 is rate limit, remaining will often be 0
    if remaining == "0" and reset:
        reset_ts = int(reset)
        sleep_for = max(1, reset_ts - int(time.time()) + 2)
        print(f"⏳ Rate limit hit. Sleeping {sleep_for}s until reset...")
        time.sleep(sleep_for)


def search_users_single_query(query: str, pages: int, per_page: int = 100, sleep_s: float = 0.3) -> List[str]:
    """
    Fetch up to pages*per_page usernames for ONE query.
    Stops early if GitHub says the 1000-result limit is reached.
    """
    usernames: List[str] = []
    headers = github_headers()

    for page in range(1, pages + 1):
        params = {
            "q": query,
            "per_page": per_page,
            "page": page,
        }

        resp = requests.get(SEARCH_ENDPOINT, headers=headers, params=params, timeout=30)

        # Handle rate limiting
        if resp.status_code == 403:
            handle_rate_limit(resp)
            # retry once after sleeping
            resp = requests.get(SEARCH_ENDPOINT, headers=headers, params=params, timeout=30)

        if resp.status_code == 422:
            # Common case: "Only the first 1000 search results are available"
            print(f"⚠️ 422 for query slice (1000 cap reached). Stopping pages for this slice.\n   Query: {query}")
            break

        if resp.status_code != 200:
            raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text}")

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        for it in items:
            login = it.get("login")
            if login:
                usernames.append(login)

        print(f"  - page {page}: +{len(items)} (slice total {len(usernames)})")

        # Be nice to the API
        time.sleep(sleep_s)

        # If fewer than per_page results, no more pages
        if len(items) < per_page:
            break

    return usernames


def build_created_slices(start_year: int, end_year: int) -> List[str]:
    """
    Build half-year slices: YYYY-01-01..YYYY-06-30 and YYYY-07-01..YYYY-12-31
    """
    slices: List[str] = []
    for y in range(start_year, end_year + 1):
        slices.append(f"type:user created:{y}-01-01..{y}-06-30")
        slices.append(f"type:user created:{y}-07-01..{y}-12-31")
    return slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GitHub usernames using Search API with date slicing.")
    parser.add_argument("--output", default="code/data/usernames.txt", help="Output file path")
    parser.add_argument("--target", type=int, default=3000, help="How many usernames to collect (approx).")
    parser.add_argument("--pages-per-slice", type=int, default=10, help="Max pages per slice (10 => max 1000 per slice).")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year for created-date slicing.")
    parser.add_argument("--end-year", type=int, default=2024, help="End year for created-date slicing.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep between requests (seconds).")
    args = parser.parse_args()

    # IMPORTANT: pages-per-slice must not exceed 10, otherwise you hit the 1000 cap anyway
    if args.pages_per_slice > 10:
        print("⚠️ Setting --pages-per-slice > 10 is pointless (Search API max 1000 results/query). Forcing to 10.")
        args.pages_per_slice = 10

    slices = build_created_slices(args.start_year, args.end_year)

    all_users: List[str] = []
    seen: Set[str] = set()

    print("🔎 Collecting GitHub usernames using created-date slices...")
    print(f"Target: {args.target} usernames")
    print(f"Slices: {len(slices)} (from {args.start_year} to {args.end_year})")
    print(f"Pages per slice: {args.pages_per_slice}\n")

    for i, q in enumerate(slices, start=1):
        if len(seen) >= args.target:
            break

        print(f"Slice {i}/{len(slices)}: {q}")
        batch = search_users_single_query(q, pages=args.pages_per_slice, per_page=100, sleep_s=args.sleep)

        new_count = 0
        for u in batch:
            if u not in seen:
                seen.add(u)
                all_users.append(u)
                new_count += 1

        print(f"✅ Slice added {new_count} new usernames. Total so far: {len(seen)}\n")

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for u in all_users[: args.target]:
            f.write(u + "\n")

    print(f"🎉 Saved {min(len(all_users), args.target)} usernames to: {out_path}")
    if len(all_users) < args.target:
        print(f"⚠️ Only collected {len(all_users)}. Increase year range or loosen slice strategy.")


if __name__ == "__main__":
    main()
