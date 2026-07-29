"""Web search, webpage reading, and report lookup helpers."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any, Literal, TypedDict
from urllib.parse import quote_plus, urlsplit

import requests
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from assistant.infrastructure.settings import DEEPSEEK_KEY, MODEL, REPORT_DIR
from assistant.tools.text import plain_text

LOG = logging.getLogger("feishu-assistant.tools.search")

http = requests.Session()
http.headers["User-Agent"] = "FeishuResearchAssistant/1.0"
direct_http = requests.Session()
direct_http.trust_env = False
direct_http.headers["User-Agent"] = "FeishuResearchAssistant/1.0"
llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")


def _normalize_query(query: str) -> str:
    query = query.strip()[:300]
    query = re.sub(
        r"帮我|联网|查一下|查询|搜索|今天|今日|最新|最近|有什么值得关注的|新动态|新闻|问一下|一下|请问",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", query).strip()


def extract_search_query(question: str) -> str:
    query = _normalize_query(question)
    return query or question.strip()[:300]


def web_search(query: str) -> list[dict[str, str]]:
    query = query.strip()[:300]
    if not query:
        return []

    search_query = _normalize_query(query) or query
    research_terms = (
        "arxiv",
        "paper",
        "research",
        "nlp",
        "translation",
        "machine translation",
        "论文",
        "研究",
        "翻译",
        "机器翻译",
    )
    if any(term in search_query.lower() for term in research_terms):
        try:
            arxiv_terms = [
                token
                for token in re.findall(r"[A-Za-z]{3,}", search_query.lower())
                if token not in {"research", "trend", "trends", "latest", "recent", "paper", "papers"}
            ][:4]
            arxiv_expression = " AND ".join(f"all:{token}" for token in arxiv_terms) or "all:machine AND all:translation"
            response = http.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": arxiv_expression,
                    "start": 0,
                    "max_results": 5,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                timeout=25,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            atom = "{http://www.w3.org/2005/Atom}"
            papers: list[dict[str, str]] = []
            for entry in root.findall(f"{atom}entry")[:5]:
                title = " ".join((entry.findtext(f"{atom}title", default="")).split())
                url = entry.findtext(f"{atom}id", default="")
                summary = " ".join((entry.findtext(f"{atom}summary", default="")).split())
                published = entry.findtext(f"{atom}published", default="")[:10]
                paper_text = f"{title} {summary}".lower()
                if "translation" in search_query.lower() and not any(
                    marker in paper_text for marker in ("translation", "translator", "multilingual", "cross-lingual")
                ):
                    continue
                if title and url:
                    papers.append({"title": title[:300], "url": url, "snippet": f"{published} {summary[:700]}"})
            if papers:
                return papers
        except Exception as exc:
            LOG.info("arXiv search unavailable: %s", exc)

    for rss_url, label in (
        (f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans", "Google News RSS"),
        (f"https://www.bing.com/news/search?q={quote_plus(search_query)}&format=rss", "Bing News RSS"),
    ):
        try:
            response = http.get(rss_url, timeout=25)
            response.raise_for_status()
            channel = ET.fromstring(response.content).find("channel")
            if channel is None:
                continue
            results: list[dict[str, str]] = []
            for item in channel.findall("item")[:5]:
                title = unescape(item.findtext("title", default=""))
                link = item.findtext("link", default="")
                description = re.sub(r"<[^>]+>", "", unescape(item.findtext("description", default="")))
                if link:
                    results.append({"title": title[:300], "url": link, "snippet": description[:800]})
            if results:
                return results
        except Exception as exc:
            LOG.info("%s unavailable: %s", label, exc)

    try:
        with DDGS() as search:
            try:
                results = list(search.news(search_query, max_results=5))
            except Exception as exc:
                LOG.info("DuckDuckGo news backend unavailable: %s", exc)
                results = []
            if not results:
                results = list(search.text(search_query, region="wt-wt", safesearch="moderate", max_results=5))
        return [
            {
                "title": str(item.get("title", ""))[:300],
                "url": str(item.get("url") or item.get("href") or ""),
                "snippet": str(item.get("body") or item.get("excerpt") or "")[:800],
            }
            for item in results
            if item.get("url") or item.get("href")
        ]
    except Exception as exc:
        LOG.warning("web search failed: %s", exc)
        return []


def read_public_webpage(url: str) -> dict[str, str]:
    url = url.strip().rstrip(".,;:!?)】》”'\"]")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    blocked = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    if parsed.scheme not in {"http", "https"} or not host or host in blocked or host.startswith(("10.", "192.168.", "169.254.")):
        return {"error": "The URL is not a permitted public HTTP(S) webpage."}

    last_error: requests.RequestException | None = None
    response: requests.Response | None = None
    for session in (http, direct_http):
        try:
            response = session.get(url, timeout=25, allow_redirects=True)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
    if response is None:
        LOG.warning("webpage reader failed for %s: %s", host, last_error)
        return {"error": "The webpage could not be fetched right now.", "url": url}

    try:
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            return {"error": "The URL does not expose readable HTML/text content.", "url": response.url}
        page = response.text[:600_000]
        page = re.sub(r"<(script|style|noscript)[^>]*>.*?</\\1>", " ", page, flags=re.IGNORECASE | re.DOTALL)
        title_match = re.search(r"<title[^>]*>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", unescape(title_match.group(1))).strip() if title_match else host
        text = re.sub(r"<[^>]+>", " ", page)
        text = re.sub(r"\s+", " ", unescape(text)).strip()
        return {"title": title[:500], "url": response.url, "text": text[:18_000]}
    except (UnicodeError, ValueError) as exc:
        LOG.warning("webpage reader could not parse %s: %s", host, exc)
        return {"error": "The webpage content could not be parsed.", "url": url}


def latest_report(query: str = "") -> str:
    reports = sorted(REPORT_DIR.glob("structured-*.md"), reverse=True)
    if not reports:
        return "暂无已归档日报。"
    content = reports[0].read_text(encoding="utf-8", errors="replace")
    return f"文件：{reports[0].name}\n\n{content[:16000]}"


def needs_research(question: str) -> bool:
    return bool(
        re.search(
            r"新闻|今日|今天|最新|动态|查询|查一下|找资料|论文|技术|研究|arxiv|NLP|机器翻译|translation|research|search",
            question,
            re.IGNORECASE,
        )
    )


class AssistantState(TypedDict, total=False):
    question: str
    context: str
    route: Literal["research", "report", "chat"]
    search_query: str
    search_results: list[dict[str, str]]
    report_context: str
    answer: str


def route_request(state: AssistantState) -> dict[str, Any]:
    question = state["question"]
    if "日报" in question:
        route: Literal["research", "report", "chat"] = "report"
    elif needs_research(question):
        route = "research"
    else:
        route = "chat"
    return {"route": route, "search_query": extract_search_query(question)}


def select_route(state: AssistantState) -> str:
    return state["route"]


def research_node(state: AssistantState) -> dict[str, Any]:
    return {"search_results": web_search(state["search_query"])}


def report_node(_: AssistantState) -> dict[str, Any]:
    return {"report_context": latest_report()}


def compose_node(state: AssistantState) -> dict[str, Any]:
    system = "你是飞书里的私人研究助理。只用中文纯文本回答，不要 Markdown。"
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"最近聊天上下文：\n{state.get('context', '')}\n\n"
                f"已检索资料：\n{state.get('search_results', [])}\n\n"
                f"日报资料：\n{state.get('report_context', '')}\n\n"
                f"用户问题：{state['question']}"
            ),
        },
    ]
    final = llm.chat.completions.create(model=MODEL, messages=messages, temperature=0.25).choices[0].message.content
    return {"answer": plain_text(final or "暂时无法生成回答。")[:12000]}


def build_graph() -> Any:
    graph = StateGraph(AssistantState)
    graph.add_node("route", route_request)
    graph.add_node("research", research_node)
    graph.add_node("report", report_node)
    graph.add_node("compose", compose_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", select_route, {"research": "research", "report": "report", "chat": "compose"})
    graph.add_edge("research", "compose")
    graph.add_edge("report", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


@tool("web_search")
def native_web_search(query: str) -> str:
    """Search current public web and news sources."""
    import json

    return json.dumps(web_search(query), ensure_ascii=False, default=str)[:24000]


@tool("read_webpage")
def native_read_webpage(url: str) -> str:
    """Read visible text from a public webpage shared in the conversation."""
    import json

    return json.dumps(read_public_webpage(url), ensure_ascii=False, default=str)[:24000]
