from __future__ import annotations

from typing import Any, Dict

from hushhush.ingest.github_client import GithubClient, GithubNotFoundError
from hushhush.storage.cache import load_from_cache, save_to_cache


class GithubFetcher:
    def __init__(self, client: GithubClient | None = None):
        self.client = client or GithubClient()

    def _get(
        self,
        username: str,
        endpoint: str,
        refresh: bool = False,
    ) -> Any:
        if not refresh:
            cached = load_from_cache(username, endpoint)
            if cached is not None:
                return cached

        data = self.client.get_json(endpoint)
        save_to_cache(username, endpoint, data)
        return data

    def fetch_user_bundle(
        self,
        username: str,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch core GitHub data for a user.
        """
        try:
            user = self._get(username, f"/users/{username}", refresh)
            repos = self._get(
                username,
                f"/users/{username}/repos",
                refresh,
            )
            events = self._get(
                username,
                f"/users/{username}/events/public",
                refresh,
            )
        except GithubNotFoundError:
            raise ValueError(f"GitHub user not found: {username}")

        return {
            "user": user,
            "repos": repos,
            "events": events,
        }
