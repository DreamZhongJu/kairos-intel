"""Feishu document, knowledge-base, and scholarly-source tools."""

from __future__ import annotations

import json
import logging
import re

import requests
from langchain_core.tools import tool
from openai import OpenAI

from assistant.channels.feishu import user_feishu_request
from assistant.infrastructure.settings import DEEPSEEK_KEY, KNOWLEDGE_SPACES_PATH, MODEL, SKILLS_DIR
from assistant.memory.store import _memory_terms
from assistant.tools.text import plain_text

LOG = logging.getLogger("feishu-assistant.tools.docs")
http = requests.Session()
http.headers["User-Agent"] = "FeishuResearchAssistant/1.0"
llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")


def _document_id_from_link(link: str) -> str:
    wiki_match = re.search(r"/wiki/([A-Za-z0-9]+)", link)
    if wiki_match:
        node = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": wiki_match.group(1)}).get("data", {}).get("node", {})
        if node.get("obj_type") != "docx" or not node.get("obj_token"):
            raise ValueError("这个知识库节点不是可直接读取的新版云文档。")
        return str(node["obj_token"])
    match = re.search(r"/(?:docx|docs)/([A-Za-z0-9]+)", link)
    if not match:
        raise ValueError("请发送完整的飞书云文档或知识库链接。")
    return match.group(1)


def _summarize(content: str) -> str:
    prompt = (
        "请用中文纯文本总结下面的飞书文档：主题、3-6 个要点、待办和风险（如有）。"
        "只依据原文，不要编造，不要使用 Markdown。\n\n"
        + content[:18_000]
    )
    response = llm.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
    return plain_text(response.choices[0].message.content or "文档内容为空。")[:12_000]


def document_summary(link: str) -> str:
    """Read a Feishu cloud document by URL and return a grounded summary."""
    try:
        document_id = _document_id_from_link(link)
        content = user_feishu_request("GET", f"/docx/v1/documents/{document_id}/raw_content").get("data", {}).get("content", "")
        if not content:
            return "老师，已访问文档，但没有获取到可读正文。请确认文档权限。"
        return _summarize(content)
    except ValueError as exc:
        return f"老师，{exc}"
    except Exception as exc:
        LOG.warning("document summary unavailable: %s", type(exc).__name__)
        return "老师，暂时无法读取这篇飞书文档，请稍后重试。"


def search_and_summarize_document(query: str) -> str:
    """Search authorized Feishu documents, then summarize the best match."""
    query = re.sub(r".*?(读取文档|总结文档|搜索文档|我的云文档)", "", query).strip(" ，。:：")
    if not query:
        return "老师，请提供文档标题、主题或关键词。"
    result = user_feishu_request("POST", "/search/v2/doc_wiki/search", json={"query": query[:200], "page_size": 10})
    items = result.get("data", {}).get("items") or result.get("data", {}).get("docs_entities") or result.get("data", {}).get("entities") or []
    if isinstance(items, dict):
        items = list(items.values())
    chosen = next(
        (
            item for item in items
            if isinstance(item, dict) and (item.get("doc_type") == "docx" or item.get("type") == "docx" or str(item.get("token", "")).startswith("dox"))
        ),
        None,
    )
    token = chosen and (chosen.get("token") or chosen.get("doc_token") or chosen.get("document_id"))
    if not token:
        return f"老师，没有找到与“{query}”匹配的可访问云文档。"
    return document_summary(f"https://feishu.cn/docx/{token}")


def installed_skill(name: str) -> str:
    """Read only reviewed, explicitly supported local skill instructions."""
    if name not in {"paper-lookup", "huggingface-papers", "markitdown"}:
        return ""
    path = SKILLS_DIR / name / "SKILL.md"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:6000] if path.is_file() else ""
    except OSError as exc:
        LOG.warning("installed skill unavailable: %s", exc)
        return ""


def huggingface_paper_lookup(query: str) -> list[dict[str, str]]:
    """Look up a paper by arXiv ID using Hugging Face's public API."""
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
    return [{
        "title": str(item.get("title") or paper_id)[:300],
        "url": str(item.get("url") or f"https://huggingface.co/papers/{paper_id}"),
        "snippet": str(item.get("summary") or item.get("ai_summary") or "")[:1800],
    }]


def academic_paper_lookup(query: str) -> list[dict[str, str]]:
    """Look up primary scholarly metadata through Crossref, with HF ID support."""
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
    results: list[dict[str, str]] = []
    for item in response.json().get("message", {}).get("items", []):
        title = (item.get("title") or [""])[0]
        url = item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else "")
        if title and url:
            results.append({"title": str(title)[:300], "url": str(url), "snippet": re.sub(r"<[^>]+>", "", str(item.get("abstract") or ""))[:1200]})
    return results


def configured_wiki_context(query: str, limit: int = 3) -> str:
    """Fallback scan for configured knowledge spaces not yet indexed by search."""
    try:
        configured = json.loads(KNOWLEDGE_SPACES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    terms = _memory_terms(query)
    candidates: list[tuple[int, str, str]] = []
    for label, parent_token in configured.items():
        if any(str(other) in query for other in configured) and str(label) not in query:
            continue
        try:
            parent = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": parent_token}).get("data", {}).get("node", {})
            space_id = str(parent.get("space_id") or "")
            if not space_id:
                continue
            nodes = user_feishu_request(
                "GET", f"/wiki/v2/spaces/{space_id}/nodes", params={"parent_node_token": parent_token, "page_size": 30}
            ).get("data", {}).get("items", [])
            for node in nodes if isinstance(nodes, list) else []:
                if not isinstance(node, dict) or node.get("obj_type") != "docx" or not node.get("obj_token"):
                    continue
                title, token = str(node.get("title") or ""), str(node["obj_token"])
                candidates.append((len(terms & _memory_terms(title)), title, token))
        except Exception as exc:
            LOG.info("configured wiki lookup unavailable for %s: %s", label, type(exc).__name__)
    snippets: list[str] = []
    for _, title, token in sorted(candidates, reverse=True)[:limit]:
        try:
            content = user_feishu_request("GET", f"/docx/v1/documents/{token}/raw_content").get("data", {}).get("content", "")
            if content:
                snippets.append(f"[知识库文档：{title}]\n{content[:5000]}")
        except Exception as exc:
            LOG.info("configured wiki document unavailable: %s", type(exc).__name__)
    return "\n\n".join(snippets)


def knowledge_context(query: str) -> str:
    """Retrieve grounded snippets from accessible Feishu documents."""
    if len(query.strip()) < 4:
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
            if item.get("type") == "wiki" or item.get("doc_type") == "wiki":
                node = user_feishu_request("GET", "/wiki/v2/spaces/get_node", params={"token": token}).get("data", {}).get("node", {})
                token = node.get("obj_token") if node.get("obj_type") == "docx" else None
            if token:
                content = user_feishu_request("GET", f"/docx/v1/documents/{token}/raw_content").get("data", {}).get("content", "")
                if content:
                    snippets.append(content[:5000])
        return "\n\n".join(snippets) or configured_wiki_context(query)
    except Exception as exc:
        LOG.info("knowledge search unavailable: %s", type(exc).__name__)
        return ""


@tool("paper_lookup")
def native_paper_lookup(query: str) -> str:
    """Find scholarly papers, citations, abstracts, and DOI records."""
    return json.dumps(academic_paper_lookup(query), ensure_ascii=False, default=str)[:24_000]


@tool("huggingface_papers")
def native_huggingface_papers(query: str) -> str:
    """Look up an arXiv identifier or Hugging Face paper page."""
    return json.dumps(huggingface_paper_lookup(query), ensure_ascii=False, default=str)[:24_000]


@tool("knowledge_search")
def native_knowledge_search(query: str) -> str:
    """Search the user's authorized Feishu documents and knowledge base."""
    return knowledge_context(query)[:18_000] or "没有找到可访问的相关私人内容。"


@tool("read_feishu_document")
def native_read_feishu_document(link: str) -> str:
    """Read and summarize a Feishu cloud-document or wiki URL shared by the user."""
    return document_summary(link)


@tool("daily_report")
def native_daily_report() -> str:
    """Read the latest generated daily intelligence report."""
    from assistant.tools.search import latest_report

    return latest_report()


@tool("knowledge_save")
def native_knowledge_save(keywords: str, content: str) -> str:
    """Explain that a target knowledge base must be named before archival."""
    del keywords, content
    return "请明确指定要归档到的知识库名称；随后可使用 archive_to_knowledge_base 写入。"


@tool("save_cloud_document")
def native_save_cloud_document(title: str, content: str) -> str:
    """Create a Feishu cloud document after an explicit user request."""
    title = re.sub(r"[\r\n]+", " ", title).strip()[:100] or "助手笔记"
    content = content.strip()[:56_000]
    if not content:
        return "未写入：笔记正文为空。"
    created = user_feishu_request("POST", "/docx/v1/documents", json={"title": title}).get("data", {}).get("document", {})
    document_id = created.get("document_id")
    if not document_id:
        return "创建云文档失败：飞书未返回文档标识。"
    children = [
        {"block_type": 2, "text": {"elements": [{"text_run": {"content": content[index:index + 1400], "text_element_style": {}}}]}}
        for index in range(0, len(content), 1400)
    ][:40]
    user_feishu_request(
        "POST", f"/docx/v1/documents/{document_id}/blocks/{document_id}/children", json={"children": children, "index": -1}
    )
    return f"已创建云文档：https://my.feishu.cn/docx/{document_id}"
