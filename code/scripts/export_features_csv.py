from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


import pandas as pd


# ---------- helpers ----------
def _read_cached_json(path: Path) -> Any:
    """
    Your cached files look like:
      {"endpoint": "...", "fetched_at": "...", "data": <actual payload>}
    This function returns the "data" part if present, else the whole file.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # GitHub timestamps are ISO 8601 like "2024-01-01T12:34:56Z"
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _days_since(dt: Optional[datetime], now: datetime) -> Optional[float]:
    if not dt:
        return None
    return (now - dt).total_seconds() / 86400.0


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# ---------- feature extraction ----------
@dataclass
class Bundle:
    username: str
    user: Dict[str, Any]
    repos: List[Dict[str, Any]]
    events: List[Dict[str, Any]]


def extract_features(bundle: Bundle) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)

    user = bundle.user or {}
    repos = bundle.repos or []
    events = bundle.events or []

    # ---- user/profile features ----
    created_at = _parse_dt(user.get("created_at"))
    updated_at = _parse_dt(user.get("updated_at"))

    feats: Dict[str, Any] = {}
    feats["username"] = bundle.username

    feats["followers"] = user.get("followers", 0) or 0
    feats["following"] = user.get("following", 0) or 0
    feats["public_repos_reported"] = user.get("public_repos", 0) or 0
    feats["public_gists"] = user.get("public_gists", 0) or 0

    feats["account_age_days"] = _days_since(created_at, now)
    feats["profile_updated_days_ago"] = _days_since(updated_at, now)

    feats["has_bio"] = int(bool(user.get("bio")))
    feats["has_company"] = int(bool(user.get("company")))
    feats["has_blog"] = int(bool(user.get("blog")))
    feats["hireable"] = int(bool(user.get("hireable")))
    feats["site_admin"] = int(bool(user.get("site_admin")))

    # ---- repo aggregate features ----
    feats["repos_fetched"] = len(repos)

    stars = [r.get("stargazers_count") or 0 for r in repos]
    forks = [r.get("forks_count") or 0 for r in repos]
    watchers = [r.get("watchers_count") or r.get("watchers") or 0 for r in repos]
    open_issues = [r.get("open_issues_count") or 0 for r in repos]
    sizes = [r.get("size") or 0 for r in repos]  # KB

    feats["total_stars"] = sum(stars)
    feats["avg_stars"] = _safe_div(sum(stars), len(repos))
    feats["total_forks"] = sum(forks)
    feats["avg_forks"] = _safe_div(sum(forks), len(repos))
    feats["total_watchers"] = sum(watchers)
    feats["avg_watchers"] = _safe_div(sum(watchers), len(repos))
    feats["total_open_issues"] = sum(open_issues)
    feats["avg_open_issues"] = _safe_div(sum(open_issues), len(repos))

    feats["avg_repo_size_kb"] = _safe_div(sum(sizes), len(repos))
    feats["median_repo_size_kb"] = float(pd.Series(sizes).median()) if repos else 0.0

    fork_flags = [bool(r.get("fork")) for r in repos]
    archived_flags = [bool(r.get("archived")) for r in repos]
    disabled_flags = [bool(r.get("disabled")) for r in repos]
    has_pages_flags = [bool(r.get("has_pages")) for r in repos]
    has_issues_flags = [bool(r.get("has_issues")) for r in repos]

    feats["fork_ratio"] = _safe_div(sum(fork_flags), len(repos))
    feats["archived_ratio"] = _safe_div(sum(archived_flags), len(repos))
    feats["disabled_ratio"] = _safe_div(sum(disabled_flags), len(repos))
    feats["has_pages_ratio"] = _safe_div(sum(has_pages_flags), len(repos))
    feats["has_issues_ratio"] = _safe_div(sum(has_issues_flags), len(repos))

    # languages
    langs = [r.get("language") for r in repos if r.get("language")]
    feats["unique_languages"] = len(set(langs))
    if langs:
        top_lang = pd.Series(langs).value_counts().iloc[0]
        feats["top_language_share"] = float(top_lang) / float(len(langs))
    else:
        feats["top_language_share"] = 0.0

    # recency based on repos
    pushed_dates = [_parse_dt(r.get("pushed_at")) for r in repos]
    updated_dates = [_parse_dt(r.get("updated_at")) for r in repos]

    pushed_days = [d for d in (_days_since(dt, now) for dt in pushed_dates) if d is not None]
    updated_days = [d for d in (_days_since(dt, now) for dt in updated_dates) if d is not None]

    feats["days_since_last_push"] = min(pushed_days) if pushed_days else None
    feats["days_since_last_repo_update"] = min(updated_days) if updated_days else None

    # ---- events features ----
    feats["events_fetched"] = len(events)
    event_types = [e.get("type") for e in events if e.get("type")]
    vc = pd.Series(event_types).value_counts() if event_types else pd.Series(dtype=int)

    # A few common event types (add more if you like)
    for t in [
        "PushEvent",
        "PullRequestEvent",
        "PullRequestReviewEvent",
        "IssuesEvent",
        "IssueCommentEvent",
        "CreateEvent",
        "ForkEvent",
        "WatchEvent",
    ]:
        feats[f"events_{t}"] = int(vc.get(t, 0))

    last_event_dt = max((_parse_dt(e.get("created_at")) for e in events), default=None)
    feats["days_since_last_event"] = _days_since(last_event_dt, now)

    # ---- cleanup: numeric-friendly ----
    # Convert None -> NaN (pandas-friendly); later we can fillna(0) if training requires.
    return feats


# ---------- loading bundles ----------
def load_bundle_for_user(user_dir: Path, username: str) -> Optional[Bundle]:
    user_path = user_dir / f"users_{username}.json"
    repos_path = user_dir / f"users_{username}_repos.json"
    events_path = user_dir / f"users_{username}_events_public.json"

    if not user_path.exists() or not repos_path.exists():
        return None

    user = _read_cached_json(user_path) or {}
    repos = _read_cached_json(repos_path) or []
    events = _read_cached_json(events_path) if events_path.exists() else []

    # Ensure list types
    if isinstance(repos, dict):
        repos = repos.get("items", []) or []
    if isinstance(events, dict):
        events = events.get("items", []) or []

    return Bundle(username=username, user=user, repos=repos, events=events)


def discover_user_dirs(raw_root: Path) -> List[Path]:
    # expected: data/raw/github/<username>/
    return [p for p in raw_root.iterdir() if p.is_dir()]


def infer_username_from_dir(user_dir: Path) -> str:
    return user_dir.name

def refine_features(df: pd.DataFrame, inject_noise: bool = False, noise_level: float = 0.01) -> pd.DataFrame:
    """
    Add stronger, ML-friendly derived features:
    - log transforms for heavy-tailed vars
    - ratios (quality / intensity)
    - per-year normalization
    - optional Gaussian noise injection (for robustness testing)
    """
    df = df.copy()

    # Ensure numeric columns become numeric (turn bad values into NaN)
    numeric_cols = [c for c in df.columns if c != "username"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill NaNs before ratios/logs (safe defaults)
    df = df.fillna(0)

    # --------- Log transforms (stabilize heavy tails) ----------
    for col in ["followers", "total_stars", "total_forks", "total_repo_size", "total_watchers"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # --------- Ratios (quality signals) ----------
    # Avoid divide-by-zero by adding 1
    if "total_stars" in df.columns and "public_repos" in df.columns:
        df["stars_per_repo"] = df["total_stars"] / (df["public_repos"] + 1)

    if "total_forks" in df.columns and "public_repos" in df.columns:
        df["forks_per_repo"] = df["total_forks"] / (df["public_repos"] + 1)

    if "followers" in df.columns and "following" in df.columns:
        df["follower_following_ratio"] = df["followers"] / (df["following"] + 1)

    if "total_events" in df.columns and "public_repos" in df.columns:
        df["events_per_repo"] = df["total_events"] / (df["public_repos"] + 1)

    if "push_events" in df.columns and "total_events" in df.columns:
        df["push_event_share"] = df["push_events"] / (df["total_events"] + 1)

    # --------- Normalize by account age ----------
    if "account_age_days" in df.columns:
        age_years = (df["account_age_days"] / 365.0) + 1.0

        if "total_events" in df.columns:
            df["events_per_year"] = df["total_events"] / age_years

        if "commits_estimated" in df.columns:
            df["commits_per_year"] = df["commits_estimated"] / age_years

    # --------- Optional noise injection (robustness testing) ----------
    # Only apply to a shortlist of continuous features (never to username)
    if inject_noise:
        noise_targets = [c for c in df.columns if c.startswith("log_")] + [
            "stars_per_repo",
            "forks_per_repo",
            "events_per_year",
            "commits_per_year",
            "follower_following_ratio",
        ]
        noise_targets = [c for c in noise_targets if c in df.columns]

        # Add Gaussian noise proportional to each column's std
        for c in noise_targets:
            std = df[c].std()
            if std > 0:
                df[c] = df[c] + np.random.normal(0, noise_level * std, size=len(df))

    return df



# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Export GitHub cached JSON bundles into a features CSV.")
    ap.add_argument("--raw-root", default="data/raw/github", help="Folder that contains per-user subfolders.")
    ap.add_argument("--out", default="data/processed/github_features.csv", help="Output CSV path.")
    ap.add_argument("--fillna0", action="store_true", help="Fill NaN with 0 (useful before sklearn training).")
    ap.add_argument("--refine", action="store_true", help="Add refined features (log, ratios, per-year).")
    ap.add_argument("--inject-noise", action="store_true", help="Inject small Gaussian noise (for robustness testing).")
    ap.add_argument("--noise-level", type=float, default=0.01, help="Noise level multiplier (default 0.01).")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_path = Path(args.out)

    rows: List[Dict[str, Any]] = []
    skipped = 0

    if not raw_root.exists():
        raise SystemExit(f"Raw root does not exist: {raw_root.resolve()}")

    for user_dir in discover_user_dirs(raw_root):
        username = infer_username_from_dir(user_dir)
        bundle = load_bundle_for_user(user_dir, username)
        if not bundle:
            skipped += 1
            continue
        rows.append(extract_features(bundle))

    df = pd.DataFrame(rows)

# Optional refinement BEFORE fillna0 export
    if args.refine:
       df = refine_features(df, inject_noise=args.inject_noise, noise_level=args.noise_level)

# Optional: make sklearn happier
    if args.fillna0:
        for col in df.columns:
            if col == "username":
               continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0)

    df.to_csv(out_path, index=False)


    print(f"✅ Wrote {len(df)} rows to: {out_path}")
    if skipped:
        print(f"⚠️ Skipped {skipped} user folders (missing required files).")


if __name__ == "__main__":
    main()
