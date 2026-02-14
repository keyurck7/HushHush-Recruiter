from pathlib import Path
from hushhush.ingest.github_fetch import GithubFetcher
import json
import time

USER_FILE = Path("code/data/usernames.txt")
RAW_DIR = Path("data/raw/github")

RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    fetcher = GithubFetcher()

    with USER_FILE.open() as f:
        usernames = [line.strip() for line in f if line.strip()]

    print(f"Fetching data for {len(usernames)} users...")

    for i, username in enumerate(usernames, 1):
        try:
            bundle = fetcher.fetch_user_bundle(username)

            output_path = RAW_DIR / f"{username}.json"

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(bundle, f)

            print(f"[{i}/{len(usernames)}] Saved {username}")

            time.sleep(0.5)  # avoid rate limit

        except Exception as e:
            print(f"Failed for {username}: {e}")

if __name__ == "__main__":
    main()
