"""Document, paper, and knowledge-base helpers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from assistant.channels.feishu import user_feishu_request
from assistant.infrastructure.settings import DEEPSEEK_KEY, KNOWLEDGE_SPACES_PATH, MODEL, SKILLS_DIR
from assistant.memory.store import _memory_terms
from assistant.tools.text import plain_text

LOG = logging.getLogger("feishu-assistant.tools.docs")

http = requests.Session()
http.headers["User-Agent"] = "FeishuResearchAssistant/1.0"
llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")


def document_summary(link: str) -> str:
    wiki_match = re.search(r"/wiki/([A-Za-z0-9]+)", link)
    if wiki_match:
        node = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": wiki_match.group(1)}).get("data", {}).get("node", {})
        if node.get("obj_type") != "docx" or not node.get("obj_token"):
            return "这个知识库节点不是可直接读取的新版本云文档。"
        link = f"https://feishu.cn/docx/{node['obj_token']}"

    match = re.search(r"/(?:docx|docs)/([A-Za-z0-9]+)", link)
    if not match:
        return "请发送完整的飞书云文档链接。"

    document_id = match.group(1)
    result = user_feishu_request("GET", f"/docx/v1/documents/{document_id}/raw_content")
    content = result.get("data", {}).get("content", "")
    if not content:
        return "已访问文档，但未获取到正文。"

    prompt = (
        "请用中文简洁介绍下面飞书文档：主题、3-6个要点、待办、风险（如有）。不要编造。\n\n"
        + content[:18000]
    )
    answer_text = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    ).choices[0].message.content
    return plain_text(answer_text or "文档内容为空。")[:12000]


def search_and_summarize_document(query: str) -> str:
    query = re.sub(r".*?(读取文档|总结文档|搜索文档|我的云文档)", "", query).strip(" ，。;:：")
    if not query:
        return "请说明文档标题、主题或关键词。"

    result = user_feishu_request("POST", "/search/v2/doc_wiki/search", json={"query": query, "page_size": 10})
    data = result.get("data", {})
    candidates = data.get("items") or data.get("docs_entities") or data.get("entities") or []
    if isinstance(candidates, dict):
        candidates = list(candidates.values())

    chosen = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and (item.get("doc_type") == "docx" or item.get("type") == "docx" or str(item.get("token", "")).startswith("dox"))
        ),
        None,
    )
    if not chosen:
        return f"没有找到与“{query}”匹配的可访问云文档。"

    doc_id = chosen.get("token") or chosen.get("doc_token") or chosen.get("document_id")
    if not doc_id:
        return "找到了结果，但没有返回可读取的文档标识。"
    return document_summary(f"https://feishu.cn/docx/{doc_id}")


def installed_skill(name: str) -> str:
    if name not in {"paper-lookup", "huggingface-papers", "markitdown"}:
        return ""
    path = SKILLS_DIR / name / "SKILL.md"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:6000] if path.is_file() else ""
    except OSError as exc:
        LOG.warning("installed skill unavailable: %s", exc)
        return ""


def huggingface_paper_lookup(query: str) -> list[dict[str, str]]:
    if not installed_skill("huggingface-papers"):
        return []
    match = re.search(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", query)
    if not match:
        return []
    paper_id = match.group(1)
    response = http.get(f"https://huggingface.co/api/papers/{paper_id}", timeout=25)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    item = response.json()
    title = str(item.get("title") or paper_id)
    summary = str(item.get("summary") or item.get("ai_summary") or "")
    url = str(item.get("url") or f"https://huggingface.co/papers/{paper_id}")
    return [{"title": title[:300], "url": url, "snippet": summary[:1800]}]


def academic_paper_lookup(query: str) -> list[dict[str, str]]:
    if not installed_skill("paper-lookup"):
        return []
    hf_results = huggingface_paper_lookup(query)
    if hf_results:
        return hf_results

    response = http.get(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query[:250], "rows": 5, "select": "title,URL,published,abstract,DOI"},
        timeout=25,
    )
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    results: list[dict[str, str]] = []
    for item in items:
        title = (item.get("title") or [""])[0]
        url = item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else "")
        abstract = re.sub(r"<[^>]+>", "", str(item.get("abstract") or ""))
        if title and url:
            results.append({"title": str(title)[:300], "url": str(url), "snippet": abstract[:1200]})
    return results


def knowledge_context(query: str) -> str:
    if len(query.strip()) < 4 or re.search(r"^(你好|在吗|谢谢|早上好)", query.strip()):
        return ""

    try:
        data = user_feishu_request("POST", "/search/v2/doc_wiki/search", json={"query": query[:200], "page_size": 3}).get("data", {})
        items = data.get("items") or data.get("docs_entities") or data.get("entities") or []
        if isinstance(items, dict):
            items = list(items.values())

        snippets: list[str] = []
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            token = item.get("token") or item.get("doc_token") or item.get("document_id")
            if not token:
                continue
            if item.get("type") == "wiki" or item.get("doc_type") == "wiki":
                node = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": token}).get("data", {}).get("node", {})
                token = node.get("obj_token") if node.get("obj_type") == "docx" else None
            if token:
                raw = user_feishu_request("GET", f"/docx/v1/documents/{token}/raw_content").get("data", {}).get("content", "")
                if raw:
                    snippets.append(raw[:5000])

        result = "\n\n".join(snippets)
        return result or configured_wiki_context(query)
    except Exception as exc:
        LOG.info("knowledge search unavailable: %s", exc)
        return ""


def configured_wiki_context(query: str, limit: int = 3) -> str:
    try:
        configured = json.loads(KNOWLEDGE_SPACES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    query_terms = _memory_terms(query)
    targeted = [label for label in configured if str(label) in query]
    candidates: list[tuple[int, str, str]] = []

    for label, parent_token in configured.items():
        if targeted and label not in targeted:
            continue
        try:
            parent = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": parent_token}).get("data", {}).get("node", {})
            space_id = str(parent.get("space_id") or "")
            if not space_id:
                continue
            nodes = user_feishu_request(
                "GET",
                f"/wiki/v2/spaces/{space_id}/nodes",
                params={"parent_node_token": parent_token, "page_size": 30},
            ).get("data", {}).get("items", [])
            for node in nodes if isinstance(nodes, list) else []:
                if not isinstance(node, dict) or str(node.get("obj_type")) != "docx":
                    continue
                title = str(node.get("title") or "")
                token = str(node.get("obj_token") or "")
                if not token:
                    continue
                score = len(query_terms & _memory_terms(title)) * 10
                if str(label) in query:
                    score += 5
                candidates.append((score, title, token))
        except Exception as exc:
            LOG.info("configured wiki lookup unavailable for %s: %s", label, type(exc).__name__)

    candidates.sort(key=lambda item: item[0], reverse=True)
    snippets: list[str] = []
    for _, title, token in candidates[:limit]:
        try:
            raw = user_feishu_request("GET", f"/docx/v1/documents/{token}/raw_content").get("data", {}).get("content", "")
            if raw:
                snippets.append(f"[知识库文档：{title}]\n{raw[:5000]}")
        except Exception as exc:
            LOG.info("configured wiki document unavailable: %s", type(exc).__name__)
    return "\n\n".join(snippets)


def native_paper_lookup(query: str) -> str:
    return json.dumps(academic_paper_lookup(query), ensure_ascii=False, default=str)[:24000]


def native_huggingface_papers(query: str) -> str:
    return json.dumps(huggingface_paper_lookup(query), ensure_ascii=False, default=str)[:24000]


def native_knowledge_search(query: str) -> str:
    return knowledge_context(query)[:18000] or "没有找到可访问的私有内容。"


def native_daily_report() -> str:
    from assistant.tools.search import latest_report

    return latest_report()


def native_knowledge_save(keywords: str, content: str) -> str:
    return "知识库写入能力仍然保留在主流程中，后续我会再单独拆出来。"


def native_save_cloud_document(title: str, content: str) -> str:
    return "云文档创建能力仍然保留在主流程中，后续我会再单独拆出来。"
