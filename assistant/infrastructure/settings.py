"""Centralized runtime settings for the assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Keep module imports side-effect free with respect to credentials.  This makes
# static checks, tests and CLI diagnostics possible on a fresh checkout while
# `validate_runtime_settings()` still rejects an actual unconfigured service.
APP_ID = os.getenv("LARK_APP_ID", "")
APP_SECRET = os.getenv("LARK_APP_SECRET", "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
RECENT_LIMIT = max(1, min(int(os.getenv("RECENT_MESSAGE_LIMIT", "30")), 50))
REPORT_DIR = Path(os.getenv("DAILY_REPORT_DIR", str(PROJECT_ROOT / "reports")))
SKILLS_DIR = Path(os.getenv("SKILLS_DIR", str(PROJECT_ROOT / "skills")))
DATA_DIR = Path(os.getenv("ASSISTANT_DATA_DIR", str(PROJECT_ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "assistant.db"
MEMORY_LIMIT = max(3, min(int(os.getenv("LONG_TERM_MEMORY_LIMIT", "8")), 15))
KNOWLEDGE_SPACES_PATH = Path(os.getenv("KNOWLEDGE_SPACES_PATH", str(PROJECT_ROOT / "knowledge_spaces.json")))
CLAUDE_MEM_URL = os.getenv("CLAUDE_MEM_URL", "").rstrip("/")
CLAUDE_MEM_PLATFORM = "feishu"


def validate_runtime_settings() -> None:
    """Fail clearly only when starting the real external service."""
    missing = [
        name
        for name, value in {
            "LARK_APP_ID": APP_ID,
            "LARK_APP_SECRET": APP_SECRET,
            "DEEPSEEK_API_KEY": DEEPSEEK_KEY,
            "TOKEN_ENCRYPTION_KEY": os.getenv("TOKEN_ENCRYPTION_KEY", ""),
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
