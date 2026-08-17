"""Tests for the GitHub Trending parser used by the daily report."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.reports.generator import parse_trending_html  # noqa: E402

TRENDING_FIXTURE = """
<html><body>
<article class="Box-row">
  <div class="d-flex width-full">
    <h2 class="h3 lh-condensed">
      <a href="/bytedance/deer-flow" data-view-component="true" class="Link--primary">
        bytedance / <em>deer-flow</em>
      </a>
    </h2>
    <p class="col-9 color-fg-muted my-1 pr-4">An open-source long-horizon SuperAgent harness that researches, codes, and creates.</p>
    <div class="f6 color-fg-muted mt-2">
      <span class="d-inline-block ml-0 mr-3"><span class="repo-language-color" style="background-color: #3572A5"></span><span itemprop="programmingLanguage">Python</span></span>
      <span class="d-inline-block float-sm-right"><a class="Link--muted" href="/bytedance/deer-flow/stargazers">1,234</a> stars today</span>
    </div>
  </div>
</article>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/astral-sh/uv" class="Link--primary">astral-sh / <em>uv</em></a>
  </h2>
  <p class="col-9 color-fg-muted my-1 pr-4">An extremely fast Python package and project manager.</p>
  <div class="f6 color-fg-muted mt-2">
    <span itemprop="programmingLanguage">Rust</span>
    <span class="d-inline-block float-sm-right">876 stars today</span>
  </div>
</article>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/some/owner-page" class="Link--primary">some / <em>missing-fields</em></a>
  </h2>
</article>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/sponsors/mukul975" class="Link--primary">sponsors / <em>mukul975</em></a>
  </h2>
  <p class="col-9 color-fg-muted my-1 pr-4">Sponsored: Anthropic cybersecurity skills.</p>
</article>
</body></html>
"""


class TrendingParseTest(unittest.TestCase):
    def test_parses_repo_links_and_metadata(self) -> None:
        items = parse_trending_html(TRENDING_FIXTURE)
        self.assertEqual(len(items), 3)
        first = items[0]
        self.assertEqual(first["id"], "trending:bytedance/deer-flow")
        self.assertEqual(first["url"], "https://github.com/bytedance/deer-flow")
        self.assertIn("bytedance / deer-flow", first["title"])
        self.assertIn("SuperAgent harness", first["summary"])
        self.assertIn("今日新增 1234 star", first["summary"])
        self.assertIn("语言 Python", first["summary"])

    def test_sponsored_items_are_skipped(self) -> None:
        items = parse_trending_html(TRENDING_FIXTURE)
        self.assertNotIn("mukul975", [item["id"] for item in items])

    def test_stars_inside_anchor_and_bare_variants(self) -> None:
        items = parse_trending_html(TRENDING_FIXTURE)
        second = items[1]
        self.assertIn("今日新增 876 star", second["summary"])
        self.assertIn("语言 Rust", second["summary"])

    def test_missing_fields_do_not_break(self) -> None:
        items = parse_trending_html(TRENDING_FIXTURE)
        third = items[2]
        self.assertEqual(third["summary"], "GitHub Trending 热门仓库")
        self.assertNotIn("今日新增", third["summary"])

    def test_non_trending_html_yields_nothing(self) -> None:
        self.assertEqual(parse_trending_html("<html><body>no articles here</body></html>"), [])

    def test_default_config_includes_trending_switch(self) -> None:
        from kairos.reports.generator import DEFAULT_CONFIG

        self.assertIs(DEFAULT_CONFIG["github_trending"], True)
        self.assertIn("trending_since", DEFAULT_CONFIG)
        self.assertIn("trending_language", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
