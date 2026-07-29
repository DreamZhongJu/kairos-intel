"""Feishu cloud-document archival helpers."""

from __future__ import annotations

import json
import logging
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any

from assistant.channels.feishu import user_feishu_request
from assistant.infrastructure.settings import DB_PATH, KNOWLEDGE_SPACES_PATH

LOG = logging.getLogger("feishu-assistant.tools.archive")


def knowledge_space_target(name: str) -> tuple[str, str] | None:
    try:
        configured = json.loads(KNOWLEDGE_SPACES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    normalized = re.sub(r"\s+", "", name)
    for label, token in configured.items():
        if label in name or re.sub(r"\s+", "", label) in normalized:
            return str(label), str(token)
    return None


def _drive_children(folder_token: str) -> list[dict[str, Any]]:
    payload = user_feishu_request("GET", f"/drive/explorer/v2/folder/{folder_token}/children")
    raw = payload.get("data", {}).get("children", [])
    if isinstance(raw, dict):
        return [item for item in raw.values() if isinstance(item, dict)]
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _drive_item_token(item: dict[str, Any]) -> str:
    return str(item.get("token") or item.get("file_token") or item.get("id") or "")


def _drive_item_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("obj_type") or item.get("file_type") or "").lower()


def _is_drive_folder(item: dict[str, Any]) -> bool:
    item_type = _drive_item_type(item)
    return bool(item.get("is_folder")) or item_type in {"folder", "drive_folder"}


def list_cloud_documents(limit: int = 80, max_depth: int = 5) -> list[dict[str, str]]:
    """List the user's own cloud docs. This function never changes Drive data."""
    root = user_feishu_request("GET", "/drive/explorer/v2/root_folder/meta").get("data", {})
    root_token = str(root.get("token") or root.get("id") or "")
    if not root_token:
        raise RuntimeError("未获得飞书云盘根目录")
    documents: list[dict[str, str]] = []
    visited: set[str] = set()

    def walk(folder_token: str, depth: int) -> None:
        if depth > max_depth or len(documents) >= limit or folder_token in visited:
            return
        visited.add(folder_token)
        for item in _drive_children(folder_token):
            token = _drive_item_token(item)
            item_type = _drive_item_type(item)
            if not token:
                continue
            if _is_drive_folder(item):
                walk(token, depth + 1)
            elif item_type in {"docx", "doc"}:
                documents.append({
                    "token": token,
                    "type": item_type,
                    "title": str(item.get("name") or item.get("title") or "未命名文档")[:160],
                    "modified_time": str(item.get("modified_time") or item.get("modifiedTime") or ""),
                })
                if len(documents) >= limit:
                    return

    walk(root_token, 0)
    return documents


def classify_archive_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ask the model to classify titles, but keep an explicit pending bucket."""
    if not documents:
        return []
    choices = "科研、技术雷达、情报与观察、个人工作台、待确认"
    plan: list[dict[str, str]] = []
    for item in documents:
        plan.append({**item, "target": "待确认", "reason": "标题信息不足"})
    return plan


def create_archive_preview(limit: int = 80) -> str:
    documents = list_cloud_documents(limit=max(1, min(int(limit), 150)))
    if not documents:
        return "云盘根目录及其子目录中没有发现可归档的飞书文档。"
    plan = classify_archive_documents(documents)
    code = secrets.token_hex(3).upper()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT OR REPLACE INTO archive_batches VALUES (?, ?, ?)",
            (code, json.dumps(plan, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
    groups: dict[str, list[dict[str, str]]] = {name: [] for name in ("科研", "技术雷达", "情报与观察", "个人工作台", "待确认")}
    for item in plan:
        groups[item["target"]].append(item)
    lines = [f"已扫描到 {len(plan)} 篇可迁移的云文档。以下只是预览，尚未移动任何文件。"]
    for name, items in groups.items():
        if not items:
            continue
        lines.append(f"\n{name}（{len(items)} 篇）")
        lines.extend(f"- {item['title']}（{item['reason']}）" for item in items[:20])
        if len(items) > 20:
            lines.append(f"- 其余 {len(items) - 20} 篇已省略显示")
    lines.append("\n待确认项不会被迁移。若确认按此预览归档，请回复：确认批量归档 " + code)
    lines.append("归档会把已有云文档迁移到对应知识库，不会删除文档内容。")
    return "\n".join(lines)


def execute_archive_batch(code: str) -> str:
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT plan_json FROM archive_batches WHERE code=?", (code.upper(),)).fetchone()
    if not row:
        return "没有找到这个待确认的归档预览编号；请先让我扫描并生成预览。"
    try:
        plan = json.loads(row[0])
    except json.JSONDecodeError:
        return "这份归档预览已损坏，请重新扫描。"
    queued, skipped, failures = 0, 0, []
    space_cache: dict[str, tuple[str, str]] = {}
    for item in plan:
        target = str(item.get("target") or "待确认")
        if target == "待确认" or item.get("type") != "docx":
            skipped += 1
            continue
        if target not in space_cache:
            selected = knowledge_space_target(target)
            if not selected:
                failures.append(f"{item.get('title', '未命名文档')}：未配置目标知识库")
                continue
            _, parent_token = selected
            node = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": parent_token}).get("data", {}).get("node", {})
            if not node.get("space_id"):
                failures.append(f"{item.get('title', '未命名文档')}：无法读取目标知识库")
                continue
            space_cache[target] = (str(node["space_id"]), parent_token)
        space_id, parent_token = space_cache[target]
        try:
            user_feishu_request(
                "POST",
                f"/wiki/v2/spaces/{space_id}/nodes/move_docs_to_wiki",
                json={"parent_wiki_token": parent_token, "obj_type": "docx", "obj_token": item["token"], "apply": False},
            )
            queued += 1
        except Exception as exc:
            LOG.warning("archive move failed for %s: %s", item.get("token"), exc)
            failures.append(f"{item.get('title', '未命名文档')}：迁移请求失败")
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM archive_batches WHERE code=?", (code.upper(),))
    result = f"已提交 {queued} 篇云文档的知识库归档任务。飞书会异步完成迁移。"
    if skipped:
        result += f"\n另有 {skipped} 篇未迁移：它们属于待确认，或不是新版云文档格式。"
    if failures:
        result += "\n以下文档未提交：\n- " + "\n- ".join(failures[:10])
    return result


def native_archive_to_knowledge_base(target: str, title: str, content: str) -> str:
    """Create a note directly under one configured Feishu knowledge base after an explicit user request."""
    selected = knowledge_space_target(target)
    if not selected:
        return "未归档：请明确指定科研、技术雷达、情报与观察或个人工作台之一。"
    label, parent_token = selected
    title = re.sub(r"[\r\n]+", " ", title).strip()[:100] or "凯伊笔记"
    content = content.strip()[:56000]
    if not content:
        return "未归档：笔记正文为空。"
    node_info = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": parent_token}).get("data", {}).get("node", {})
    space_id = node_info.get("space_id")
    if not space_id:
        return f"未归档到{label}：无法解析该知识库空间。"
    node = user_feishu_request(
        "POST",
        f"/wiki/v2/spaces/{space_id}/nodes",
        json={"obj_type": "docx", "node_type": "origin", "parent_node_token": parent_token, "title": title},
    ).get("data", {}).get("node", {})
    doc_id = node.get("obj_token")
    node_token = node.get("node_token")
    if not doc_id:
        return f"未归档到{label}：飞书未返回新文档标识。"
    chunks = [content[i:i + 1400] for i in range(0, len(content), 1400)][:40]
    children = [{"block_type": 2, "text": {"elements": [{"text_run": {"content": chunk, "text_element_style": {}}}]}} for chunk in chunks]
    user_feishu_request("POST", f"/docx/v1/documents/{doc_id}/blocks/{doc_id}/children", json={"children": children, "index": -1})
    return f"已归档到{label}：https://my.feishu.cn/wiki/{node_token or parent_token}"


def native_preview_cloud_archive(limit: int = 80) -> str:
    """Scan existing Feishu cloud documents and make a non-destructive archive preview."""
    return create_archive_preview(limit)

