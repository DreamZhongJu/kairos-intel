"""Daily intelligence report scheduler for Kairós."""

from __future__ import annotations

import logging
import os
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from kairos.reports import generator

LOG = logging.getLogger("kairos.reports.scheduler")

REPORT_TIMEZONE = "Asia/Shanghai"
REPORT_HOUR = max(0, min(int(os.getenv("DAILY_REPORT_HOUR", "9")), 23))


def run_report() -> None:
    """Generate one report, never letting a failure take down the app."""
    try:
        generator.main()
    except Exception:
        LOG.exception("daily report generation failed")


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
    return scheduler


def start_scheduler() -> BackgroundScheduler:
    scheduler = build_scheduler()
    scheduler.start()
    LOG.info("daily report scheduler started: every day at %02d:00 %s", REPORT_HOUR, REPORT_TIMEZONE)
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
