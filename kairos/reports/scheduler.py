"""Daily / weekly / monthly intelligence report scheduler for Kairós."""

from __future__ import annotations

import logging
import os
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from kairos.reports import generator, periodic

LOG = logging.getLogger("kairos.reports.scheduler")

REPORT_TIMEZONE = "Asia/Shanghai"
REPORT_HOUR = max(0, min(int(os.getenv("DAILY_REPORT_HOUR", "9")), 23))
WEEKLY_REPORT = os.getenv("WEEKLY_REPORT", "1") == "1"
MONTHLY_REPORT = os.getenv("MONTHLY_REPORT", "1") == "1"
WEEKLY_DAY = os.getenv("DAILY_REPORT_WEEKLY_DAY", "mon").lower()
MONTHLY_DAY = max(1, min(int(os.getenv("DAILY_REPORT_MONTHLY_DAY", "1")), 28))


def run_report() -> None:
    """Generate one report, never letting a failure take down the app."""
    try:
        generator.main()
    except Exception:
        LOG.exception("daily report generation failed")


def run_weekly() -> None:
    try:
        periodic.main("weekly")
    except Exception:
        LOG.exception("weekly report generation failed")


def run_monthly() -> None:
    try:
        periodic.main("monthly")
    except Exception:
        LOG.exception("monthly report generation failed")


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=REPORT_TIMEZONE)
    scheduler.add_job(
        run_report,
        CronTrigger(hour=REPORT_HOUR, minute=0, timezone=REPORT_TIMEZONE),
        id="daily-intel-report",
        name="daily-intel-report",
        coalesce=True,
        misfire_grace_time=3600,
        replace_existing=True,
    )
    if WEEKLY_REPORT:
        scheduler.add_job(
            run_weekly,
            CronTrigger(day_of_week=WEEKLY_DAY, hour=REPORT_HOUR, minute=5, timezone=REPORT_TIMEZONE),
            id="weekly-intel-report",
            name="weekly-intel-report",
            coalesce=True,
            misfire_grace_time=3600,
            replace_existing=True,
        )
    if MONTHLY_REPORT:
        scheduler.add_job(
            run_monthly,
            CronTrigger(day=MONTHLY_DAY, hour=REPORT_HOUR, minute=10, timezone=REPORT_TIMEZONE),
            id="monthly-intel-report",
            name="monthly-intel-report",
            coalesce=True,
            misfire_grace_time=3600,
            replace_existing=True,
        )
    return scheduler


def start_scheduler() -> BackgroundScheduler:
    scheduler = build_scheduler()
    scheduler.start()
    LOG.info(
        "report scheduler started: daily %02d:00, weekly %s %02d:05, monthly day %d %02d:10 (%s)",
        REPORT_HOUR, WEEKLY_DAY, REPORT_HOUR, MONTHLY_DAY, REPORT_HOUR, REPORT_TIMEZONE,
    )
    return scheduler


def main() -> None:
    """Run the scheduler in the foreground for a standalone process."""
    scheduler = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
