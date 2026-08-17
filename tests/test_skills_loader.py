"""Tests for the external skill reader client (skill_list / skill_load)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.tools import skills_loader  # noqa: E402
from kairos.agent import runtime as agent_runtime  # noqa: E402

FAKE_LIST = {"skills": [{"name": "deep-research", "description": "Systematic literature review in 6 phases."}]}
FAKE_SKILL = "# deep-research\n\nConduct systematic academic literature reviews..."


class SkillListTest(unittest.TestCase):
    def setUp(self) -> None:
        skills_loader.API_URL = "http://localhost:8777"

    def _list(self) -> str:
        return skills_loader.native_skill_list.invoke({})

    def test_skill_list_returns_formatted(self) -> None:
        with patch("kairos.tools.skills_loader.requests.get") as mock:
            mock.return_value.json.return_value = FAKE_LIST
            result = self._list()
        self.assertIn("deep-research", result)
        self.assertIn("Systematic literature review", result)

    def test_skill_list_no_api_url(self) -> None:
        skills_loader.API_URL = ""
        result = self._list()
        self.assertIn("未配置", result)

    def test_skill_list_network_error(self) -> None:
        with patch("kairos.tools.skills_loader.requests.get") as mock:
            mock.side_effect = Exception("timeout")
            result = self._list()
        self.assertIn("暂不可用", result)


class SkillLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        skills_loader.API_URL = "http://localhost:8777"

    def _load(self, name: str) -> str:
        return skills_loader.native_skill_load.invoke({"name": name})

    def test_skill_load_returns_content(self) -> None:
        with patch("kairos.tools.skills_loader.requests.get") as mock:
            mock.return_value.status_code = 200
            mock.return_value.text = FAKE_SKILL
            result = self._load("deep-research")
        self.assertIn("deep-research", result)

    def test_skill_load_404(self) -> None:
        with patch("kairos.tools.skills_loader.requests.get") as mock:
            mock.return_value.status_code = 404
            result = self._load("nonexistent")
        self.assertIn("没有找到", result)

    def test_skill_load_empty_name(self) -> None:
        result = self._load("")
        self.assertIn("请提供技能名称", result)

    def test_skill_load_no_api_url(self) -> None:
        skills_loader.API_URL = ""
        result = self._load("any")
        self.assertIn("未配置", result)

    def test_registered_in_native_tools(self) -> None:
        names = {t.name for t in agent_runtime.NATIVE_TOOLS}
        self.assertIn("skill_list", names)
        self.assertIn("skill_load", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)