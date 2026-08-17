"""Feishu transport helpers and message parsing utilities."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import UTC, datetime
from typing import Any

import lark_oapi as lark
import requests

from kairos.infrastructure.settings import APP_ID, APP_SECRET, RECENT_LIMIT
from oauth_server import user_access_token

LOG = logging.getLogger("kairos.channels.feishu")

http = requests.Session()
http.headers["User-Agent"] = "Kairós/1.0"

_token_lock = threading.Lock()
_tenant_token: str | None = None
_token_expiry = 0.0


def event_to_dict(data: Any) -> dict[str, Any]:
    marshalled = lark.JSON.marshal(data)
    return json.loads(marshalled) if isinstance(marshalled, str) else marshalled


def tenant_token() -> str:
    global _tenant_token, _token_expiry
    now = datetime.now(UTC).timestamp()
    with _token_lock:
        if _tenant_token and now < _token_expiry:
            return _tenant_token
        response = http.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": APP_ID, "app_secret": APP_SECRET},
            timeout=20,
        )
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"unable to obtain tenant token: {payload.get('msg')}")
        _tenant_token = payload["tenant_access_token"]
        _token_expiry = now + int(payload.get("expire", 7200)) - 120
        return _tenant_token


def feishu_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {tenant_token()}"
    response = http.request(
        method,
        f"https://open.feishu.cn/open-apis{path}",
        headers=headers,
        timeout=30,
        **kwargs,
    )
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"Feishu API {path}: {payload.get('code')} {payload.get('msg')}")
    return payload


def user_feishu_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {user_access_token()}"
    response = http.request(method, f"https://open.feishu.cn/open-apis{path}", headers=headers, timeout=30, **kwargs)
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"Feishu API {path}: {payload.get('msg', 'unknown error')}")
    return payload


def reply(message_id: str, text: str) -> str:
    """Reply to a message; returns the new message id when available."""
    payload = feishu_request(
        "POST",
        f"/im/v1/messages/{message_id}/reply",
        json={"msg_type": "text", "content": json.dumps({"text": text}, ensure_ascii=False)},
    )
    data = payload.get("data") or {}
    return str(data.get("message_id") or "")


def recall_message(message_id: str) -> None:
    """Withdraw a message previously sent by the bot (best effort)."""
    try:
        feishu_request("DELETE", f"/im/v1/messages/{message_id}")
    except Exception as exc:
        LOG.warning("recall failed for %s: %s", message_id, type(exc).__name__)


TEXT_CHUNK_LIMIT = 1600


def chunk_text(text: str, limit: int = TEXT_CHUNK_LIMIT) -> list[str]:
    """Split plain text into Feishu-friendly chunks on paragraph boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        piece = paragraph.strip()
        if not piece:
            continue
        if current and len(current) + len(piece) + 1 > limit:
            chunks.append(current)
            current = ""
        if current:
            current = f"{current}\n{piece}"
        else:
            current = piece
            while len(current) > limit:  # a single over-long paragraph
                chunks.append(current[:limit])
                current = current[limit:]
    if current:
        chunks.append(current)
    return chunks or [text]


def message_text(raw_content: str) -> str:
    try:
        content = json.loads(raw_content)
        if content.get("text"):
            return str(content["text"])

        def flatten(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, dict):
                out = []
                for key in ("text", "title", "url", "href"):
                    if value.get(key):
                        out.append(str(value[key]))
                for item in value.values():
                    out.extend(flatten(item))
                return out
            if isinstance(value, list):
                return [part for item in value for part in flatten(item)]
            return []

        return " ".join(flatten(content))
    except (TypeError, json.JSONDecodeError):
        return str(raw_content or "")


def clean_question(text: str) -> str:
    text = re.sub(r"@_user_\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def recent_chat(chat_id: str) -> str:
    try:
        result = feishu_request(
            "GET",
            "/im/v1/messages",
            params={
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": RECENT_LIMIT,
                "sort_type": "ByCreateTimeDesc",
            },
        )
        items = result.get("data", {}).get("items", [])
        lines = []
        for item in reversed(items):
            if item.get("msg_type") != "text":
                continue
            text = clean_question(message_text(item.get("body", {}).get("content", "")))
            if text:
                lines.append(text[:1000])
        return "\n".join(lines[-RECENT_LIMIT:]) or "（当前会话没有可读的文本消息）"
    except Exception as exc:
        LOG.warning("recent chat unavailable: %s", exc)
        return "（无法读取近期群聊；请检查 im:message:readonly 权限和机器人是否已加入该群）"


def latest_file_in_chat(chat_id: str) -> tuple[str, str] | None:
    """Find the newest file in this chat, so a later @ message can refer to it."""
    try:
        result = feishu_request(
            "GET",
            "/im/v1/messages",
            params={
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": 20,
                "sort_type": "ByCreateTimeDesc",
            },
        )
        for item in result.get("data", {}).get("items", []):
            if item.get("msg_type") == "file":
                return item.get("message_id", ""), item.get("body", {}).get("content", "{}")
    except Exception as exc:
        LOG.warning("latest file lookup unavailable: %s", exc)
    return None


def urls_in_message_content(raw_content: str) -> list[str]:
    """Extract public and Feishu URLs from text/post/card payloads without losing card links."""
    try:
        decoded = json.loads(raw_content)
        serialized = json.dumps(decoded, ensure_ascii=False)
    except (TypeError, json.JSONDecodeError):
        serialized = str(raw_content or "")
    return list(dict.fromkeys(re.findall(r"https?://[^\s\\\"<>]+", serialized)))


def latest_reference_in_chat(chat_id: str) -> dict[str, str] | None:
    """Return the last shared link/card or attachment so follow-up requests have an object to act on."""
    try:
        result = feishu_request(
            "GET",
            "/im/v1/messages",
            params={
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": RECENT_LIMIT,
                "sort_type": "ByCreateTimeDesc",
            },
        )
        for item in result.get("data", {}).get("items", []):
            content = item.get("body", {}).get("content", "{}")
            urls = urls_in_message_content(content)
            if urls:
                url = urls[0]
                kind = "feishu_doc" if re.search(r"/(?:wiki|docx|docs)/", url) else "webpage"
                return {"kind": kind, "url": url, "message_id": str(item.get("message_id", ""))}
            if item.get("msg_type") == "file":
                return {"kind": "file", "message_id": str(item.get("message_id", "")), "content": content}
    except Exception as exc:
        LOG.warning("latest reference lookup unavailable: %s", exc)
    return None

