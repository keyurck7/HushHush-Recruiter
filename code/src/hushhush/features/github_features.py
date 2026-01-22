from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hushhush.storage.cache import load_from_cache


def _parse_github_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """
    GitHub timestamps look like: '2011-09-03T15:26:22Z'
    """
    if not dt_str:
        return None
    try:
        # Replace Z with +00:00 for Python parsing
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


@dataclass
class GithubFeatureExtractor:
    """
    Extracts ML-friendly numeric features from cached GitHub data for a username.
    Cached endpoints expected (from your ingestion pipeline):
      - /users/{username}
      - /users/{username}/repos
      - /users/{username}/events/public
    """

    def extract(self, username: str) -> Dict[str, Any]:
        user = load_from_cache(username, f"/users/{username}")
        repos = load_from_cache(username, f"/users/{username}/repos") or []
        events = load_from_cache(username, f"/users/{username}/events/public") or []

        if user is None:
            raise ValueError(
                f"No cached GitHub user data found for '{username}'. "
                f"Run: python code/scripts/fetch_github_user.py --user {username}"
            )

        now = datetime.now(timezone.utc)

        created_at = _parse_github_datetime(user.get("created_at"))
        updated_at = _parse_github_datetime(user.get("updated_at"))

        account_age_days = (now - created_at).days if created_at else None
        profile_updated_days_ago = (now - updated_at).days if updated_at else None

        # Repo aggregates
        repo_count = len(repos)
        total_stars = sum(int(r.get("stargazers_count", 0) or 0) for r in repos)
        total_forks = sum(int(r.get("forks_count", 0) or 0) for r in repos)
        total_watchers = sum(int(r.get("watchers_count", 0) or 0) for r in repos)
        total_open_issues = sum(int(r.get("open_issues_count", 0) or 0) for r in repos)

        avg_stars = _safe_div(total_stars, repo_count)
        avg_forks = _safe_div(total_forks, repo_count)

        forked_repo_count = sum(1 for r in repos if r.get("fork") is True)
        fork_ratio = _safe_div(forked_repo_count, repo_count)

        # Language diversity
        langs = [r.get("language") for r in repos if r.get("language")]
        unique_langs = len(set(langs))

        # Repo freshness: days since most recently pushed repo
        pushed_dates: List[datetime] = []
        for r in repos:
            pushed = _parse_github_datetime(r.get("pushed_at"))
            if pushed:
                pushed_dates.append(pushed)
        days_since_last_push = (now - max(pushed_dates)).days if pushed_dates else None

        # Event recency: days since last public event in cached page
        event_dates: List[datetime] = []
        for e in events:
            ed = _parse_github_datetime(e.get("created_at"))
            if ed:
                event_dates.append(ed)
        days_since_last_event = (now - max(event_dates)).days if event_dates else None

        events_count = len(events)

        followers = int(user.get("followers", 0) or 0)
        following = int(user.get("following", 0) or 0)

        followers_following_ratio = _safe_div(followers, following)

        # A few “profile completeness” signals
        has_bio = 1 if user.get("bio") else 0
        has_company = 1 if user.get("company") else 0
        has_blog = 1 if user.get("blog") else 0
        has_location = 1 if user.get("location") else 0

        return {
            "username": user.get("login") or username,
            "account_age_days": account_age_days,
            "profile_updated_days_ago": profile_updated_days_ago,
            "followers": followers,
            "following": following,
            "followers_following_ratio": followers_following_ratio,
            "public_repos_reported": int(user.get("public_repos", 0) or 0),
            "repos_fetched": repo_count,
            "total_stars": total_stars,
            "avg_stars": avg_stars,
            "total_forks": total_forks,
            "avg_forks": avg_forks,
            "total_watchers": total_watchers,
            "total_open_issues": total_open_issues,
            "fork_ratio": fork_ratio,
            "unique_languages": unique_langs,
            "events_fetched": events_count,
            "days_since_last_push": days_since_last_push,
            "days_since_last_event": days_since_last_event,
            "has_bio": has_bio,
            "has_company": has_company,
            "has_blog": has_blog,
            "has_location": has_location,
        }
