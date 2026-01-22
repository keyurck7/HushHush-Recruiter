from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from hushhush.config import GITHUB_API_BASE, GITHUB_TOKEN


class GithubAPIError(Exception):
    """Base exception for GitHub API issues."""


class GithubNotFoundError(GithubAPIError):
    """Raised when a GitHub resource is not found (404)."""


@dataclass
class GithubClient:
    timeout_seconds: int = 20
    max_retries: int = 3
    backoff_seconds: float = 1.5

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hushhush-recruiter/1.0",
        }
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        return headers

    def get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Perform a GET request to GitHub API and return parsed JSON.
        endpoint example: "/users/torvalds"
        """
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        url = f"{GITHUB_API_BASE}{endpoint}"

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self.timeout_seconds,
                )

                # Handle common statuses
                if resp.status_code == 404:
                    raise GithubNotFoundError(f"Not found: {endpoint}")

                # Rate limit handling (often 403)
                if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers:
                    remaining = resp.headers.get("X-RateLimit-Remaining")
                    reset = resp.headers.get("X-RateLimit-Reset")
                    if remaining == "0" and reset:
                        reset_ts = int(reset)
                        sleep_for = max(0, reset_ts - int(time.time())) + 2
                        raise GithubAPIError(
                            f"Rate limit hit. Try again in ~{sleep_for} seconds."
                        )

                # Any other error
                if resp.status_code >= 400:
                    raise GithubAPIError(
                        f"GitHub API error {resp.status_code} for {endpoint}: {resp.text[:200]}"
                    )

                return resp.json()

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
                    continue
                raise GithubAPIError(f"Network error after retries: {e}") from e

            except GithubAPIError as e:
                # If it's rate-limit or API error, don't keep retrying blindly
                raise

        # Should never reach here, but just in case:
        raise GithubAPIError(f"Failed to GET {endpoint}. Last error: {last_err}")
