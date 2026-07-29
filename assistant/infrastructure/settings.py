"""Centralized runtime settings for the assistant."""

from __future__ import annotations

import os
from pathlib import Path


APP_ID = os.environ["LARK_APP_ID"]
APP_SECRET = os.environ["LARK_APP_SECRET"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
RECENT_LIMIT = max(1, min(int(os.getenv("RECENT_MESSAGE_LIMIT", "30")), 50))
REPORT_DIR = Path(os.getenv("DAILY_REPORT_DIR", "/reports"))
SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "/app/skills"))
DATA_DIR = Path(os.getenv("ASSISTANT_DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "assistant.db"
MEMORY_LIMIT = max(3, min(int(os.getenv("LONG_TERM_MEMORY_LIMIT", "8")), 15))
KNOWLEDGE_SPACES_PATH = Path(os.getenv("KNOWLEDGE_SPACES_PATH", "/app/knowledge_spaces.json"))
CLAUDE_MEM_URL = os.getenv("CLAUDE_MEM_URL", "").rstrip("/")
CLAUDE_MEM_PLATFORM = "feishu"

