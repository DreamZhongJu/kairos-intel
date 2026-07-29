"""Read-only Feishu calendar tooling."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from assistant.channels.feishu import user_feishu_request


@tool("today_schedule")
def today_schedule() -> str:
    """Read today's Feishu calendar schedule."""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    primary_data = user_feishu_request("POST", "/calendar/v4/calendars/primary").get("data", {})
    primary = primary_data.get("calendar") or (primary_data.get("calendars") or [{}])[0]
    calendar_id = primary.get("calendar_id")
    if not calendar_id:
        return "老师，未找到可访问的主日历。"

    result = user_feishu_request(
        "GET",
        f"/calendar/v4/calendars/{calendar_id}/events",
        params={"start_time": str(int(start.timestamp())), "end_time": str(int(end.timestamp())), "page_size": 100},
    )
    items = result.get("data", {}).get("items", [])
    if not items:
        return "老师，今天没有日程安排。"

    lines = ["老师，今天的安排："]
    for item in items[:30]:
        begin = datetime.fromtimestamp(int(item.get("start_time", {}).get("timestamp", 0)), ZoneInfo("Asia/Shanghai")).strftime("%H:%M")
        finish = datetime.fromtimestamp(int(item.get("end_time", {}).get("timestamp", 0)), ZoneInfo("Asia/Shanghai")).strftime("%H:%M")
        lines.append(f"- {begin}–{finish}：{item.get('summary', '未命名日程')}")
    return "\n".join(lines)
