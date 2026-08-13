"""Offline smoke tests for importability and LangGraph wiring."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import app
from kairos.agent import runtime
from kairos.infrastructure import settings
from kairos.tools import docs


class AssistantSmokeTests(unittest.TestCase):
    def test_all_registered_tools_are_langchain_tools(self) -> None:
        self.assertEqual(len(runtime.NATIVE_TOOLS), 20)
        self.assertTrue(all(getattr(tool, "name", "") for tool in runtime.NATIVE_TOOLS))
        self.assertTrue(all(hasattr(tool, "invoke") for tool in runtime.NATIVE_TOOLS))

    def test_graph_builds_without_production_credentials(self) -> None:
        self.assertIsNotNone(runtime.build_graph())

    def test_dsml_fallback_is_parsed_as_a_real_tool_call(self) -> None:
        content = '<|DSML|> <|tool_calls|> <|invoke name="web_search"> <|parameter name="query" string="true">NLP news</|parameter> </|invoke>'
        calls = runtime._dsml_tool_calls(content)
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["args"]["query"], "NLP news")

    def test_paper_lookup_does_not_require_an_optional_skill_file(self) -> None:
        response = MagicMock()
        response.json.return_value = {"message": {"items": [{"title": ["Example paper"], "URL": "https://doi.org/example"}]}}
        with patch.object(docs.http, "get", return_value=response):
            self.assertEqual(docs.academic_paper_lookup("example")[0]["title"], "Example paper")

    def test_runtime_validation_reports_missing_configuration(self) -> None:
        with (
            patch.multiple(settings, APP_ID="", APP_SECRET="", DEEPSEEK_KEY=""),
            patch.dict("os.environ", {"TOKEN_ENCRYPTION_KEY": ""}, clear=False),
            self.assertRaisesRegex(RuntimeError, "LARK_APP_ID"),
        ):
            settings.validate_runtime_settings()

    def test_main_wires_services_before_starting_websocket(self) -> None:
        handler_builder = MagicMock()
        handler_builder.register_p2_im_message_receive_v1.return_value.build.return_value = object()
        websocket = MagicMock()
        with (
            patch.object(app, "validate_runtime_settings"),
            patch.object(app, "init_db"),
            patch.object(app, "init_memory_runtime"),
            patch.object(app, "init_oauth"),
            patch.object(app, "start_oauth_server"),
            patch.object(app, "start_scheduler"),
            patch.object(app.agent_runtime, "build_graph", return_value=object()),
            patch.object(app.lark.EventDispatcherHandler, "builder", return_value=handler_builder),
            patch.object(app.lark.ws, "Client", return_value=websocket),
        ):
            app.main()
        websocket.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
