from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env in the repo root
# This allows us to use os.getenv("GITHUB_TOKEN")
load_dotenv()


def get_project_root() -> Path:
    """
    Returns the repository root path.
    Assumes this file lives in: code/src/hushhush/config.py
    So root is 3 levels up from this file.
    """
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT: Path = get_project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_GITHUB_DIR: Path = RAW_DIR / "github"
PROCESSED_DIR: Path = DATA_DIR / "processed"

GITHUB_API_BASE: str = "https://api.github.com"

GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")


def ensure_dirs() -> None:
    """
    Create required data directories if they don't exist.
    """
    RAW_GITHUB_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
