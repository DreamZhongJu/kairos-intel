"""Structured Chinese daily intelligence report.

This script intentionally keeps the report layout separate from Horizon's
importance-ranked digest.  Edit data/report_config.json on the server to change
research topics, search queries, or the rotating company roster.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import smtplib
import sys
from datetime import date, datetime, timedelta, timezone
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import feedparser
import httpx
from kairos.infrastructure import settings
from kairos.infrastructure.llm import build_client, model_name


DATA = settings.DATA_DIR
CONFIG_PATH = DATA / "report_config.json"
STATE_PATH = DATA / "report_state.json"

COMPANY_FALLBACKS = {
    "Modal": "面向开发者的无服务器云平台，重点提供按需 GPU、批处理和 Python 应用运行环境。",
    "Baseten": "面向生产环境的机器学习推理部署平台，帮助团队部署、扩缩容和监控模型服务。",
    "Replicate": "通过 API 提供开源模型运行能力的平台，降低团队使用生成式模型的工程门槛。",
    "Fireworks AI": "提供高速、低延迟大模型推理与微调服务的 AI 基础设施团队。",
    "Together AI": "围绕开放权重大模型提供训练、微调和推理云服务的平台。",
    "Runpod": "面向开发者与研究者的 GPU 云平台，主打按需算力和容器化部署。",
    "Weights & Biases": "机器学习实验跟踪、模型评估和协作平台，常用于训练过程管理。",
    "Turso": "基于 libSQL 的边缘数据库平台，面向低延迟应用和分布式 SQLite 工作负载。",
    "Neon": "无服务器 PostgreSQL 平台，支持按需扩缩容、分支数据库与开发环境隔离。",
    "Astral": "开发 Ruff 与 uv 等高性能 Python 工具链的团队，聚焦开发者效率。",
    "MotherDuck": "将 DuckDB 扩展到云端协作与分析场景的数据平台。",
    "Temporal": "提供持久化工作流执行引擎，帮助工程团队可靠运行长任务和分布式业务流程。",
    "PostHog": "开源产品分析套件，包含事件分析、会话录制、特性开关与实验功能。",
    "Tailscale": "基于 WireGuard 的组网与零信任访问平台，简化跨设备和跨云网络连接。",
    "Nscale": "为 AI 训练与推理提供 GPU 云基础设施的欧洲团队。",
}

DEFAULT_CONFIG = {
    "research_topic": "自然语言处理（NLP）中的机器翻译（Machine Translation）",
    "social_query": "国际社会 重要事件 科学 商业 气候 灾害 -中国 -中方 -政府 -外交 -治理",
    "tech_query": "AI software developer tools semiconductors cybersecurity open source -中国 -政府 -治理",
    "opensource_query": "GitHub release open source AI developer tools framework -中国 -政府",
    "research_query": "all:\"machine translation\" OR all:\"neural machine translation\"",
    "new_tech_query": "artificial intelligence OR computer software OR developer tools OR computer systems OR semiconductors OR machine learning OR open source framework",
    "max_items_per_section": 5,
    "github_trending": True,
    "trending_since": "daily",
    "trending_language": "",
    "company_roster": [
        "Astera Labs", "Celestial AI", "Groq", "SiFive", "Wiz",
        "Scale AI", "CoreWeave", "Nscale", "PostHog", "Snyk",
        "Mistral AI", "Cohere", "Hugging Face", "Lambda", "Cerebras"
    ],
    "company_profiles": [
        {"name": "Modal", "url": "https://modal.com/"},
        {"name": "Baseten", "url": "https://www.baseten.co/"},
        {"name": "Replicate", "url": "https://replicate.com/"},
        {"name": "Fireworks AI", "url": "https://fireworks.ai/"},
        {"name": "Together AI", "url": "https://www.together.ai/"},
        {"name": "Runpod", "url": "https://www.runpod.io/"},
        {"name": "Weights & Biases", "url": "https://wandb.ai/"},
        {"name": "Turso", "url": "https://turso.tech/"},
        {"name": "Neon", "url": "https://neon.com/"},
        {"name": "Astral", "url": "https://astral.sh/"},
        {"name": "MotherDuck", "url": "https://motherduck.com/"},
        {"name": "Temporal", "url": "https://temporal.io/"},
        {"name": "PostHog", "url": "https://posthog.com/"},
        {"name": "Tailscale", "url": "https://tailscale.com/"},
        {"name": "Nscale", "url": "https://www.nscale.com/"}
    ]
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return DEFAULT_CONFIG.copy()
    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {**DEFAULT_CONFIG, **saved}


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"seen_research_ids": [], "last_company": ""}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"seen_research_ids": [], "last_company": ""}


REPORT_TIMEZONE = ZoneInfo("Asia/Shanghai")


def news_window_start(now: datetime | None = None) -> datetime:
    """Daily news begins at 09:00 China time on the previous calendar day."""
    now = now.astimezone(REPORT_TIMEZONE) if now else datetime.now(REPORT_TIMEZONE)
    yesterday = now.date() - timedelta(days=1)
    return datetime.combine(yesterday, datetime.min.time(), tzinfo=REPORT_TIMEZONE).replace(hour=9)


async def fetch_feed(
    client: httpx.AsyncClient,
    url: str,
    limit: int,
    published_after: datetime | None = None,
) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            result = []
            # Google News may return older articles before newer qualifying entries;
            # inspect a wider slice and enforce the precise reporting window locally.
            for entry in feed.entries[: max(limit * 8, 50)]:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title or not link:
                    continue
                published = None
                if entry.get("published_parsed"):
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).astimezone(REPORT_TIMEZONE)
                if published_after and (published is None or published < published_after):
                    continue
                summary = html.unescape((entry.get("summary") or entry.get("description") or "").strip())
                result.append({
                    "id": entry.get("id") or link,
                    "title": title,
                    "url": link,
                    "summary": summary[:1200],
                    "published_at": published.isoformat() if published else "",
                })
                if len(result) >= limit:
                    break
            return result
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep((attempt + 1) * 4)
    assert last_error is not None
    raise last_error


async def fetch_company_official_page(client: httpx.AsyncClient, company: dict) -> list[dict]:
    """Fetch first-party company metadata, independent of daily news coverage."""
    url = str(company["url"])
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            break
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep((attempt + 1) * 4)
    else:
        assert last_error is not None
        raise last_error
    page = response.text[:500_000]
    title_match = re.search(r"<title[^>]*>\s*(.*?)\s*</title>", page, re.IGNORECASE | re.DOTALL)
    description_match = re.search(
        r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\'](.*?)["\']',
        page,
        re.IGNORECASE | re.DOTALL,
    )
    title = re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip() if title_match else company["name"]
    description = re.sub(r"\s+", " ", html.unescape(description_match.group(1))).strip() if description_match else ""
    description = description or COMPANY_FALLBACKS.get(company["name"], "")
    return [{"id": f"official:{company['name']}", "title": title[:300], "url": url, "summary": description[:1200]}]


async def fetch_hackernews(client: httpx.AsyncClient, limit: int, published_after: datetime) -> list[dict]:
    """Use HN as an independent, non-Google source for computing developments."""
    response = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    response.raise_for_status()
    ids = response.json()[:40]
    item_responses = await asyncio.gather(
        *(client.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json") for item_id in ids),
        return_exceptions=True,
    )
    markers = ("ai", "llm", "model", "software", "open source", "github", "database", "security", "linux", "compiler", "gpu", "programming", "developer")
    result = []
    for response in item_responses:
        if isinstance(response, Exception):
            continue
        try:
            response.raise_for_status()
            item = response.json()
            published = datetime.fromtimestamp(int(item.get("time", 0)), tz=timezone.utc).astimezone(REPORT_TIMEZONE)
            title = str(item.get("title", "")).strip()
            if item.get("type") != "story" or published < published_after or not title:
                continue
            if not any(marker in title.lower() for marker in markers):
                continue
            url = str(item.get("url") or f"https://news.ycombinator.com/item?id={item.get('id')}")
            result.append({"id": f"hn:{item.get('id')}", "title": title[:300], "url": url, "summary": f"Hacker News 热度：{item.get('score', 0)} 分，{item.get('descendants', 0)} 条讨论。", "published_at": published.isoformat()})
        except (ValueError, TypeError, httpx.HTTPError):
            continue
        if len(result) >= limit:
            break
    return result


async def fetch_github_releases(client: httpx.AsyncClient, limit: int, published_after: datetime) -> list[dict]:
    repos = (
        "huggingface/transformers", "ollama/ollama", "pytorch/pytorch",
        "astral-sh/uv", "langchain-ai/langgraph", "kubernetes/kubernetes",
        "microsoft/vscode", "fastapi/fastapi",
    )
    feeds = await asyncio.gather(
        *(fetch_feed(client, f"https://github.com/{repo}/releases.atom", limit, published_after) for repo in repos),
        return_exceptions=True,
    )
    result = []
    for feed in feeds:
        if isinstance(feed, Exception):
            continue
        result.extend(feed)
    return sorted(result, key=lambda item: item.get("published_at", ""), reverse=True)[:limit]


def parse_trending_html(page: str) -> list[dict]:
    """Parse the server-rendered GitHub Trending page (no new dependency).

    The page is static HTML: each repo is an <article class="Box-row"> block
    with an h2 link to /owner/repo, an optional description <p>, the
    programming language, and a "N stars today" span.  Structure changes are
    tolerated: a block without a repo link is skipped, missing fields degrade
    to empty strings.
    """
    items: list[dict] = []
    for block in re.split(r'<article\s+class="Box-row"', page)[1:]:
        repo_match = re.search(r'href="/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"', block)
        if not repo_match or repo_match.group(1).startswith("sponsors/"):
            continue
        repo = repo_match.group(1)
        # <p\s avoids matching SVG <path ...> elements that embed <p in their data.
        description_match = re.search(r"<p\s[^>]*>(.*?)</p>", block, re.DOTALL)
        description = re.sub(r"<[^>]+>", "", description_match.group(1)) if description_match else ""
        description = re.sub(r"\s+", " ", html.unescape(description)).strip()
        stars_match = re.search(r"([\d,]+)[^0-9]{0,20}stars?\s+today", block)
        stars_today = stars_match.group(1).replace(",", "") if stars_match else ""
        language_match = re.search(r'itemprop="programmingLanguage">([^<]+)<', block)
        language = language_match.group(1).strip() if language_match else ""
        markers = []
        if stars_today:
            markers.append(f"今日新增 {stars_today} star")
        if language:
            markers.append(f"语言 {language}")
        summary = f"{description}（{'，'.join(markers)}）" if markers else description
        items.append({
            "id": f"trending:{repo}",
            "title": f"{repo.replace('/', ' / ')} (GitHub Trending)",
            "url": f"https://github.com/{repo}",
            "summary": (summary or "GitHub Trending 热门仓库")[:1200],
            "published_at": datetime.now(REPORT_TIMEZONE).isoformat(),
        })
    return items


async def fetch_github_trending(
    client: httpx.AsyncClient,
    limit: int,
    since: str = "daily",
    language: str = "",
) -> list[dict]:
    """Fetch today's GitHub Trending repositories (daily/weekly, optional language)."""
    language = (language or "").strip().lower()
    path = f"/trending/{language}" if language else "/trending"
    response = await client.get(f"https://github.com{path}", params={"since": since})
    response.raise_for_status()
    return parse_trending_html(response.text)[:limit]


async def fetch_feed_pool(
    client: httpx.AsyncClient,
    urls: tuple[str, ...],
    limit: int,
    published_after: datetime,
) -> list[dict]:
    """Collect independent RSS/Atom sources without allowing one outage to erase a section."""
    feeds = await asyncio.gather(
        *(fetch_feed(client, url, limit, published_after) for url in urls),
        return_exceptions=True,
    )
    unique: dict[str, dict] = {}
    for feed in feeds:
        if isinstance(feed, Exception):
            continue
        for item in feed:
            unique.setdefault(str(item["id"]), item)
    return sorted(unique.values(), key=lambda item: item.get("published_at", ""), reverse=True)[:limit]


def bbc_world_url() -> str:
    return "https://feeds.bbci.co.uk/news/world/rss.xml"


TECH_RSS_FEEDS = (
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.technologyreview.com/feed/",
)

OPENSOURCE_RSS_FEEDS = (
    "https://github.blog/feed/",
    "https://www.cncf.io/feed/",
    "https://about.gitlab.com/atom.xml",
)


def google_news_url(query: str) -> str:
    # A broad upstream window, followed by exact timestamp filtering in fetch_feed.
    params = urlencode({"q": f"{query} when:2d", "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"})
    return f"https://news.google.com/rss/search?{params}"


def arxiv_url(query: str) -> str:
    params = urlencode({"search_query": query, "start": 0, "max_results": 20, "sortBy": "submittedDate", "sortOrder": "descending"})
    return f"https://export.arxiv.org/api/query?{params}"


def compact(items: list[dict]) -> str:
    if not items:
        return "（本次检索未获得可用条目）"
    return "\n".join(f"- 标题：{x['title']}\n  摘要：{x['summary']}\n  链接：{x['url']}" for x in items)


def generate_report(config: dict, sections: dict, research_new: list[dict], nlp_new: list[dict], suzhou_news: list[dict], company: str, start_at: datetime, failed_sources: set[str]) -> str:
    # If the state store has already seen today's papers, keep the section
    # useful with a labelled watchlist instead of emitting an empty block.
    if research_new:
        research_note = "【机器翻译新增论文】\n" + compact(research_new)
    elif nlp_new:
        research_note = "【NLP 相关新增进展（机器翻译暂无新增时的回退）】\n" + compact(nlp_new)
    elif suzhou_news:
        research_note = "【苏州大学计算机学院近期动态（机器翻译与 NLP 暂无新增时的回退）】\n" + compact(suzhou_news)
    elif sections.get("research"):
        research_note = "【持续跟踪：近期机器翻译论文，非本次新增】\n" + compact(sections["research"][:3])
    elif sections.get("research_nlp"):
        research_note = "【持续跟踪：近期 NLP 论文，非本次新增】\n" + compact(sections["research_nlp"][:3])
    else:
        research_note = "研究候选来源本次联网异常，未能取得可核验的论文条目。"
    prompt = f"""你是严谨的中文情报编辑。基于下面联网检索到的候选资料，写一份简洁但有信息密度的日报。

本期新闻时间范围：{start_at.strftime('%Y-%m-%d %H:%M')}（中国标准时间）至生成时刻。不要把候选之外、或早于此范围的新闻写入新闻板块。

硬性格式：必须且只能使用如下 7 个二级标题，顺序不能变：
## 今日总览
## 今日社会新闻
## 今日科技圈新闻
## {config['research_topic']}最新动态
## 开源社区动态
## 每日一个新技术
## 每日一个“小而强”的大厂/团队

规则：
1. “今日总览”先用 3-5 条短句概括今天社会、科技、研究、开源及技术趋势中最重要的变化；每条不超过 45 个汉字，不重复后文细节。
2. 每个新闻板块选择 3-5 条真正重要且不重复的内容。每条包含“发生了什么｜为什么值得关注｜[原文](URL)”。不得编造。
3. 研究板块严格使用“研究候选”中的内容：优先“机器翻译新增论文”，其次“NLP 相关新增进展”，再其次“苏州大学计算机学院近期动态”。若候选中提供“持续跟踪”，必须照常介绍这些近期高相关论文，并明确标注“非本次新增”；不得在候选非空时输出“今日无值得更新的研究动态”。
4. 开源板块优先 GitHub Release、热门项目、重要安全或社区事件，不能用同一家公司宣传稿凑数。
5. “每日一个新技术”必须从计算技术候选或 GitHub Release/Hacker News 候选中挑选一个具体技术或工程能力，写：它是什么、解决什么问题、适用场景和局限、一个原始链接；不得以政策、产业口号或社会新闻充数。
6. “每日一个小而强的大厂/团队”本日介绍 {company}。避开泛泛宣传，写做什么、核心产品、行业位置、为何值得关注，并提供至少一个来源链接；即使官网资料较简短，也必须基于候选资料完成简洁介绍，不能杜撰。
7. 语言为简体中文；链接必须来自候选资料；每节不要超过 5 条。

【社会候选】
{compact(sections['social'])}

【科技候选】
{compact(sections['tech'])}

【研究候选】
{research_note}

【通用 NLP 候选（仅在机器翻译无更新时使用）】
{compact(nlp_new)}

【苏州大学计算机学院候选（仅在机器翻译和通用 NLP 都无更新时使用）】
{compact(suzhou_news)}

【开源候选】
{compact(sections['opensource'])}

【新技术检索候选】
{compact(sections['new_tech'])}

【公司检索候选】
{compact(sections['company'])}
"""
    prompt += f"""
Additional non-negotiable rules:
8. The “daily new technology” section must be strictly about computing: AI/ML, software, programming languages, developer tools, computer systems, cybersecurity, databases, networking, chips, or robotics. Exclude medicine, biology, energy, materials, consumer gadgets, and general science. If no computing candidate is available, say so rather than substituting another field.
9. The “daily company/team” section is mandatory every day and independent of the news sections. Introduce {company} even if it was not mentioned in any other news. Use the first-party company page included in the candidate material as a required source; explain what it builds, its core product, its niche/industry position, and why a technically minded reader should care. Never omit this section, never replace it with a company mentioned incidentally in another section, and never output “资料不足，今日不强行介绍”. This rule overrides any earlier contrary instruction.
10. Do not include Chinese political content anywhere in the report: no Chinese political figures, Party or government institutions, diplomatic statements, policy propaganda, political meetings, or related current affairs. This prohibition also applies to the overview and to any explanatory text.
11. 以下来源本次联网失败：{', '.join(sorted(failed_sources)) or '无'}。这是内部采集状态，不要在日报中逐个披露单一来源失败。只要该板块仍有任何候选资料，就必须正常编辑并使用候选；只有一个板块没有任何候选资料且其全部来源都失败时，才说明该板块的联网异常。不得让单一来源失败覆盖 GitHub Release、RSS、Google News 或其他正常候选。
12. 中国政治内容一律不采纳，即使它出现在候选中也跳过：包括“中方表示”、部委/政府表态、外交关系、国际治理倡议、党政宣传和政策口号。社会新闻优先国际民生、科学、商业、灾害与公共议题；科技新闻只写产品、工程、论文、开源和公司技术事实。
"""
    client = build_client()
    model = model_name()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "只输出可直接发送的 Markdown 日报，不要附加解释。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def send_feishu(markdown: str, report_date: str) -> None:
    card = {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"每日情报日报｜{report_date}"}, "template": "blue"},
        "body": {"elements": [{"tag": "markdown", "content": markdown}]},
    }
    target_chat = os.getenv("FEISHU_REPORT_CHAT_ID", "").strip()
    if target_chat:
        # Preferred transport: the assistant's own Feishu bot (tenant token),
        # unified with the interactive channel used by the brain.
        from kairos.channels.feishu import feishu_request

        feishu_request(
            "POST",
            "/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            json={
                "receive_id": target_chat,
                "msg_type": "interactive",
                "content": json.dumps(card, ensure_ascii=False),
            },
        )
        print("FEISHU_SENT via=brain_channel")
    else:
        # Backward-compatible fallback: custom-bot webhook.
        webhook = os.environ["FEISHU_WEBHOOK_URL"]
        payload = {"msg_type": "interactive", "card": card}
        response = httpx.post(webhook, json=payload, timeout=30)
        response.raise_for_status()
        print(f"FEISHU_SENT via=webhook status={response.status_code}")


def send_email(markdown: str, report_date: str) -> None:
    sender = os.environ["SMTP_USER"]
    recipient = os.environ["MAIL_TO"]
    message = MIMEMultipart("alternative")
    message["Subject"] = Header(f"每日情报日报｜{report_date}", "utf-8")
    message["From"] = sender
    message["To"] = recipient
    message.attach(MIMEText(markdown, "plain", "utf-8"))
    message.attach(MIMEText(f"<html><body><pre style='white-space:pre-wrap;font-family:sans-serif'>{html.escape(markdown)}</pre></body></html>", "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.163.com", 465, timeout=30) as server:
        server.login(sender, os.environ["SMTP_PASSWORD"])
        server.send_message(message)
    print("EMAIL_SENT recipient_configured=true")


async def collect(config: dict, state: dict) -> tuple[dict, list[dict], list[dict], list[dict], str, datetime, set[str], list[str], list[str]]:
    limit = int(config["max_items_per_section"]) + 3
    start_at = news_window_start()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=12)) as client:
        tasks = {
            "social": fetch_feed(client, google_news_url(config["social_query"]), limit, start_at),
            "social_bbc": fetch_feed(client, bbc_world_url(), limit, start_at),
            "tech": fetch_feed(client, google_news_url(config["tech_query"]), limit, start_at),
            "tech_rss": fetch_feed_pool(client, TECH_RSS_FEEDS, limit, start_at),
            "opensource": fetch_feed(client, google_news_url(config["opensource_query"]), limit, start_at),
            "opensource_rss": fetch_feed_pool(client, OPENSOURCE_RSS_FEEDS, limit, start_at),
            "research": fetch_feed(client, arxiv_url(config["research_query"]), 20),
            "research_nlp": fetch_feed(client, arxiv_url("cat:cs.CL"), 20),
            "suzhou": fetch_feed(client, google_news_url('"苏州大学计算机科学与技术学院" OR "苏州大学 计算机学院"'), limit, start_at),
            "new_tech": fetch_feed(client, google_news_url(config["new_tech_query"]), limit, start_at),
            "new_tech_hn": fetch_hackernews(client, limit, start_at),
            "opensource_releases": fetch_github_releases(client, limit, start_at),
        }
        if config.get("github_trending", True):
            tasks["opensource_trending"] = fetch_github_trending(
                client,
                limit,
                since=str(config.get("trending_since", "daily")),
                language=str(config.get("trending_language", "") or "").strip().lower(),
            )
        profiles = config["company_profiles"]
        profile = profiles[date.today().toordinal() % len(profiles)]
        company = profile["name"]
        tasks["company"] = fetch_feed(client, google_news_url(f'"{company}" 公司 技术 产品'), limit, start_at)
        tasks["company_official"] = fetch_company_official_page(client, profile)
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    sections = {}
    failed_sources: set[str] = set()
    for key, value in zip(tasks, results):
        if isinstance(value, Exception):
            print(f"SOURCE_FAILED source={key} error={type(value).__name__}", file=sys.stderr)
            sections[key] = []
            failed_sources.add(key)
        else:
            sections[key] = value
    sections["company"] = sections.get("company_official", []) + sections.get("company", [])
    sections["social"] = (sections.get("social", []) + sections.get("social_bbc", []))[:limit]
    sections["tech"] = (
        sections.get("tech_rss", []) + sections.get("tech", []) + sections.get("new_tech_hn", [])
    )[:limit]
    sections["opensource"] = (
        sections.get("opensource_releases", []) + sections.get("opensource_trending", [])
        + sections.get("opensource_rss", [])
        + sections.get("opensource", []) + sections.get("new_tech_hn", [])
    )[:limit]
    sections["new_tech"] = (
        sections.get("new_tech_hn", []) + sections.get("new_tech", [])
        + sections.get("opensource_releases", []) + sections.get("opensource_trending", [])
    )[:limit]
    if not sections.get("new_tech"):
        # Preserve the daily technology section even when its dedicated news
        # query is empty, while keeping its evidence strictly in computing.
        sections["new_tech"] = (sections.get("tech", []) + sections.get("opensource", []))[:limit]

    # The report describes a section as unavailable only when every independent
    # source for that section failed or returned no qualifying item.
    for section in ("social", "tech", "opensource", "new_tech", "company"):
        if sections.get(section):
            failed_sources.discard(section)
    core_sources = {"social", "tech", "opensource", "research", "new_tech", "company"}
    if core_sources.issubset(failed_sources):
        raise RuntimeError("all primary report sources are unreachable; retry instead of sending an empty report")
    seen = set(state.get("seen_research_ids", []))
    research_new = [item for item in sections["research"] if item["id"] not in seen]
    research_ids = [item["id"] for item in sections["research"]][:200]
    seen_nlp = set(state.get("seen_nlp_ids", []))
    nlp_new = [item for item in sections["research_nlp"] if item["id"] not in seen_nlp and item["id"] not in seen]
    nlp_ids = [item["id"] for item in sections["research_nlp"]][:200]
    return sections, research_new, nlp_new, sections["suzhou"], company, start_at, failed_sources, research_ids, nlp_ids


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    config = load_config()
    state = load_state()
    sections, research_new, nlp_new, suzhou_news, company, start_at, failed_sources, research_ids, nlp_ids = asyncio.run(collect(config, state))
    report_date = datetime.now(REPORT_TIMEZONE).strftime("%Y-%m-%d")
    report = generate_report(config, sections, research_new, nlp_new, suzhou_news, company, start_at, failed_sources)
    output = settings.REPORT_DIR / f"structured-{report_date}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"# 每日情报日报｜{report_date}\n\n{report}\n", encoding="utf-8")
    if os.getenv("REPORT_DRY_RUN") == "1":
        print("REPORT_DRY_RUN=true")
    else:
        send_feishu(report, report_date)
        # Never consume research updates when the research source itself failed,
        # and only advance state after a report was actually sent.
        if "research" not in failed_sources:
            state["seen_research_ids"] = research_ids
        if "research_nlp" not in failed_sources:
            state["seen_nlp_ids"] = nlp_ids
        state["last_company"] = company
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT_SAVED path={output}")


if __name__ == "__main__":
    main()
