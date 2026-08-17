"""Weekly and monthly intelligence report generation for Kairós.

Builds on the daily report pipeline: it summarizes the recent daily reports,
the user's request history (from the observability DB) and long-term memory
into a periodic digest, saves it under ``REPORT_DIR/periodic/`` and pushes it
through the same Feishu channel as the daily report.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from kairos.infrastructure import settings
from kairos.infrastructure.llm import build_client, model_name
from kairos.reports import generator

LOG = logging.getLogger("kairos.reports.periodic")

REPORT_TIMEZONE = ZoneInfo("Asia/Shanghai")
PERIOD_DAYS = {"weekly": 7, "monthly": 30}
PERIOD_TITLES = {"weekly": "每周情报周报", "monthly": "每月情报月报"}
_INPUT_PRICE_PER_M = float(os.getenv("DEEPSEEK_INPUT_PRICE_PER_M", "0.27"))
_OUTPUT_PRICE_PER_M = float(os.getenv("DEEPSEEK_OUTPUT_PRICE_PER_M", "1.10"))


def recent_daily_reports(kind: str) -> list[str]:
    """Return the plain-text bodies of the most recent daily reports."""
    days = PERIOD_DAYS.get(kind, 7)
    reports = sorted(settings.REPORT_DIR.glob("structured-*.md"), reverse=True)[:days]
    bodies: list[str] = []
    for path in reports:
        body = path.read_text(encoding="utf-8", errors="replace")
        bodies.append(f"[日报 {path.stem.replace('structured-', '')}]\n{body[:12000]}")
    return bodies


def request_stats(kind: str) -> dict[str, object]:
    """Aggregate request-log metrics for the period (requests, tools, tokens)."""
    days = PERIOD_DAYS.get(kind, 7)
    since = (datetime.now(REPORT_TIMEZONE) - timedelta(days=days)).isoformat()
    try:
        con = sqlite3.connect(settings.DB_PATH)
        try:
            total = con.execute(
                "SELECT COUNT(*) FROM request_logs WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            errors = con.execute(
                "SELECT COUNT(*) FROM request_logs WHERE created_at >= ? AND status='error'", (since,)
            ).fetchone()[0]
            tokens = con.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM request_logs WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            prompt_tokens = con.execute(
                "SELECT COALESCE(SUM(prompt_tokens),0) FROM request_logs WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            completion_tokens = con.execute(
                "SELECT COALESCE(SUM(completion_tokens),0) FROM request_logs WHERE created_at >= ?", (since,)
            ).fetchone()[0]
            cost = (
                prompt_tokens / 1_000_000 * _INPUT_PRICE_PER_M
                + completion_tokens / 1_000_000 * _OUTPUT_PRICE_PER_M
            )
            top_questions = [
                row[0]
                for row in con.execute(
                    "SELECT question FROM request_logs WHERE created_at >= ? AND status='ok' "
                    "GROUP BY question ORDER BY COUNT(*) DESC LIMIT 8",
                    (since,),
                ).fetchall()
            ]
            top_tools: dict[str, int] = {}
            for (seq,) in con.execute(
                "SELECT tool_sequence FROM request_logs WHERE created_at >= ?", (since,)
            ).fetchall():
                for name in json.loads(seq or "[]"):
                    top_tools[name] = top_tools.get(name, 0) + 1
            return {
                "total": int(total),
                "errors": int(errors),
                "total_tokens": int(tokens),
                "estimated_cost_usd": round(cost, 4),
                "top_questions": top_questions,
                "top_tools": sorted(top_tools.items(), key=lambda item: item[1], reverse=True)[:6],
            }
        finally:
            con.close()
    except Exception as exc:  # observability must never block a report
        LOG.warning("request stats unavailable: %s", type(exc).__name__)
        return {"total": 0, "errors": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "top_questions": [], "top_tools": []}


def memory_highlights() -> str:
    """Return core long-term memories as personalization context."""
    try:
        from kairos.memory import store as memory_store

        owners = memory_store.all_owners()
        blocks: list[str] = []
        for owner in owners[:3]:
            core = memory_store.core_memories(owner)
            if core and core != "（无核心记忆）":
                blocks.append(f"[用户 {owner[:8]}]\n{core}")
        return "\n\n".join(blocks)[:6000] or "（无长期记忆）"
    except Exception as exc:
        LOG.warning("memory highlights unavailable: %s", type(exc).__name__)
        return "（长期记忆暂不可用）"


def generate_periodic(kind: str) -> str:
    """Generate one weekly or monthly digest via the configured model."""
    now = datetime.now(REPORT_TIMEZONE)
    days = PERIOD_DAYS.get(kind, 7)
    start = (now - timedelta(days=days - 1)).strftime("%m-%d")
    end = now.strftime("%m-%d")
    reports = recent_daily_reports(kind)
    stats = request_stats(kind)
    memories = memory_highlights()

    period_label = f"{start} ~ {end}"
    prompt = f"""你是严谨的中文情报编辑。基于最近 {days} 天的每日情报日报、使用数据和长期记忆，写一份简洁但有信息密度的{PERIOD_TITLES[kind]}。

本期时间范围：{period_label}

硬性格式：必须且只能使用如下 6 个二级标题，顺序不能变：
## 本期要览
## 研究动态
## 科技与开源动态
## 你关注了什么
## 用量与健康
## 下期展望

规则：
1. “本期要览”用 3-5 条短句概括本期最重要的变化，每条不超过 45 个汉字。
2. “研究动态”与“科技与开源动态”从下方日报资料中提炼真正重要的进展，每条包含“发生了什么｜为什么值得关注｜[原文](URL)”；链接必须来自日报资料，不得编造。
3. “你关注了什么”根据使用数据里的高频问题，归纳你近期关注的主题，并提示对应动态。
4. “用量与健康”如实报告：请求数、成功率、总 token、估算成本（美元）、常用工具。
5. “下期展望”给出 2-3 条可执行的关注建议。
6. 语言为简体中文；不得编造日报资料之外的事实。

【日报资料】
{'</分隔>'.join(reports) if reports else '（本期没有可用的日报存档）'}

【使用数据】
请求数：{stats['total']}，失败：{stats['errors']}，总 token：{stats['total_tokens']}，估算成本：${stats['estimated_cost_usd']}
高频问题：{'；'.join(stats['top_questions']) if stats['top_questions'] else '无'}
常用工具：{', '.join(f'{name}×{count}' for name, count in stats['top_tools']) or '无'}

【长期记忆】
{memories}
"""
    client = build_client()
    response = client.chat.completions.create(
        model=model_name(),
        temperature=0.2,
        messages=[
            {"role": "system", "content": "只输出可直接发送的 Markdown 周报/月报，不要附加解释。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def main(kind: str = "weekly") -> None:
    """Generate and deliver one periodic report (respects REPORT_DRY_RUN)."""
    kind = kind if kind in PERIOD_DAYS else "weekly"
    settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(REPORT_TIMEZONE)
    start = (now - timedelta(days=PERIOD_DAYS[kind] - 1)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    report = generate_periodic(kind)
    output = settings.REPORT_DIR / "periodic" / f"{kind}-{now.strftime('%Y-%m-%d')}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"# {PERIOD_TITLES[kind]}｜{start} ~ {end}\n\n{report}\n", encoding="utf-8")
    if os.getenv("REPORT_DRY_RUN") == "1":
        print("REPORT_DRY_RUN=true")
    else:
        generator.send_feishu(report, f"{start} ~ {end}", title=f"{PERIOD_TITLES[kind]}｜{start} ~ {end}")
    print(f"PERIODIC_SAVED kind={kind} path={output}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weekly")
