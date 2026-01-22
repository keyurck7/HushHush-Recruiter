from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hushhush.config import RAW_GITHUB_DIR, ensure_dirs


def _safe_endpoint_name(endpoint: str) -> str:
    """
    Convert an API endpoint into a filesystem-safe filename.
    Example: '/users/torvalds/repos' -> 'users_torvalds_repos.json'
    """
    return endpoint.strip("/").replace("/", "_") + ".json"


def load_from_cache(username: str, endpoint: str) -> Optional[Any]:
    """
    Load cached JSON response for a user and endpoint if it exists.
    """
    ensure_dirs()

    user_dir = RAW_GITHUB_DIR / username
    file_path = user_dir / _safe_endpoint_name(endpoint)

    if not file_path.exists():
        return None

    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("data")


def save_to_cache(username: str, endpoint: str, data: Any) -> Path:
    """
    Save JSON response to disk with metadata.
    """
    ensure_dirs()

    user_dir = RAW_GITHUB_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)

    file_path = user_dir / _safe_endpoint_name(endpoint)

    payload: Dict[str, Any] = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "endpoint": endpoint,
        "data": data,
    }

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return file_path
