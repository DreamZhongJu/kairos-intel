"""Agent-Reach backed internet tools exposed to the LangGraph agent.

The image contains the upstream command-line clients selected by Agent-Reach.
Each LangChain tool has a fixed purpose and never passes model output through a
shell, so untrusted webpage text cannot become an operating-system command.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import requests
from langchain_core.tools import tool


def _run(command: list[str], timeout: int = 45) -> str:
    """Run one installed client and return bounded, display-safe output."""
    if not command or not shutil.which(command[0]):
        return json.dumps({"error": f"{command[0] if command else 'tool'} is not installed"}, ensure_ascii=False)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "upstream tool timed out"}, ensure_ascii=False)
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode:
        return json.dumps({"error": "upstream tool failed", "detail": output[:2000]}, ensure_ascii=False)
    return output[:24_000] or json.dumps({"result": "no output"}, ensure_ascii=False)


@tool("agent_reach_health")
def agent_reach_health() -> str:
    """Check Agent-Reach channels and report which internet backends are available."""
    return _run(["agent-reach", "doctor", "--json"], timeout=75)


@tool("semantic_web_search")
def semantic_web_search(query: str, limit: int = 5) -> str:
    """Use Agent-Reach's Exa semantic search for current public web research."""
    query = query.strip()[:500]
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    limit = max(1, min(int(limit), 10))
    return _run(
        ["mcporter", "call", "exa.web_search_exa", f"query={query}", f"numResults={limit}"],
        timeout=75,
    )


@tool("github_research")
def github_research(query: str, limit: int = 5) -> str:
    """Search public GitHub repositories for projects, releases, and open-source activity."""
    query = query.strip()[:300]
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    limit = max(1, min(int(limit), 10))
    cli_result = _run(
        [
            "gh",
            "search",
            "repos",
            query,
            "--limit",
            str(limit),
            "--json",
            "nameWithOwner,description,url,updatedAt,stargazerCount",
        ],
        timeout=60,
    )
    # gh CLI requires authentication for this endpoint in some environments.
    # Retain Agent-Reach's preferred client when it works, but do not turn a
    # public-information query into a dead end when the user has not logged in.
    if '"error"' not in cli_result:
        return cli_result
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "updated", "order": "desc", "per_page": limit},
            headers={"Accept": "application/vnd.github+json", "User-Agent": "FeishuResearchAssistant/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        items = [
            {
                "name": item.get("full_name"),
                "description": item.get("description"),
                "url": item.get("html_url"),
                "updated_at": item.get("updated_at"),
                "stars": item.get("stargazers_count"),
            }
            for item in response.json().get("items", [])
        ]
        return json.dumps(items, ensure_ascii=False)
    except requests.RequestException:
        return cli_result


@tool("youtube_video_details")
def youtube_video_details(url: str) -> str:
    """Read metadata and available subtitles from a public YouTube video URL."""
    url = url.strip()[:1500]
    if not url.startswith(("https://", "http://")):
        return json.dumps({"error": "a public HTTP(S) video URL is required"}, ensure_ascii=False)
    return _run(["yt-dlp", "--dump-single-json", "--no-download", url], timeout=120)


@tool("bilibili_search")
def bilibili_search(query: str) -> str:
    """Search Bilibili for public technical videos using Agent-Reach's selected CLI."""
    query = query.strip()[:300]
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    return _run(["bili", "search", query, "--type", "video"], timeout=60)
