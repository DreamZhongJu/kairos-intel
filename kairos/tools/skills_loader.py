"""Open skill reader client: loads external skill instructions at runtime.

The skill contents themselves are NOT part of this repository; they are
served by an independent skill-reader HTTP service (see deploy/skills_api.py)
configured via SKILL_API_URL. The agent can list available skills and load
one by name, then follow its instructions with the existing tools.
"""

from __future__ import annotations

import logging
import os

import requests
from langchain_core.tools import tool

LOG = logging.getLogger("kairos.tools.skills")

API_URL = os.getenv("SKILL_API_URL", "").rstrip("/")
API_TOKEN = os.getenv("SKILL_API_TOKEN", "").strip()
TIMEOUT = 15


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


@tool("skill_list")
def native_skill_list() -> str:
    """List external skills the assistant can load (names and what each does)."""
    try:
        if not API_URL:
            return "技能服务未配置（SKILL_API_URL 为空）。"
        payload = requests.get(f"{API_URL}/skills", headers=_headers(), timeout=TIMEOUT).json()
        skills = payload.get("skills") or []
        if not skills:
            return "技能服务在线，但当前没有可用技能。"
        return "\n".join(
            f"- {item.get('name', '?')}：{(item.get('description') or '')[:200]}" for item in skills
        )
    except Exception as exc:
        LOG.warning("skill_list unavailable: %s", type(exc).__name__)
        return "技能服务暂不可用，请稍后再试。"


@tool("skill_load")
def native_skill_load(name: str) -> str:
    """Load one skill's full instructions by its exact name (see skill_list)."""
    name = (name or "").strip()
    if not name:
        return "请提供技能名称，例如：skill_load(name=\"deep-research\")。"
    try:
        if not API_URL:
            return "技能服务未配置（SKILL_API_URL 为空）。"
        response = requests.get(f"{API_URL}/skills/{name}", headers=_headers(), timeout=TIMEOUT)
        if response.status_code == 404:
            return f"没有找到名为 {name} 的技能；先用 skill_list 查看可用技能。"
        response.raise_for_status()
        return response.text[:24000]
    except Exception as exc:
        LOG.warning("skill_load(%s) unavailable: %s", name, type(exc).__name__)
        return f"技能服务暂不可用，无法加载 {name}。"