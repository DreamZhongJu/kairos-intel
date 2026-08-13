from contextlib import ExitStack
import unittest
from unittest import mock

from kairos.infrastructure import llm


class ResolveModelConfigTest(unittest.TestCase):
    def _resolve(self, **overrides):
        base = {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_BASE_URL": "",
            "MODEL_API_KEY": "",
            "MODEL_NAME": "",
            "DEEPSEEK_KEY": "",
            "MODEL": "deepseek-v4-flash",
        }
        base.update(overrides)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(llm.os, "getenv", return_value=""))
            for name, value in base.items():
                stack.enter_context(mock.patch.object(llm.settings, name, value))
            return llm.resolve_model_config()

    def test_default_deepseek_backward_compatible(self):
        cfg = self._resolve(DEEPSEEK_KEY="sk-legacy")
        self.assertEqual(cfg.provider, "deepseek")
        self.assertEqual(cfg.base_url, "https://api.deepseek.com")
        self.assertEqual(cfg.api_key, "sk-legacy")
        self.assertEqual(cfg.model, "deepseek-v4-flash")

    def test_switch_to_openai(self):
        cfg = self._resolve(
            MODEL_PROVIDER="openai",
            MODEL_NAME="gpt-4.1-mini",
            MODEL_API_KEY="sk-openai",
        )
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
        self.assertEqual(cfg.model, "gpt-4.1-mini")

    def test_custom_base_url_and_model(self):
        cfg = self._resolve(
            MODEL_BASE_URL="http://localhost:11434/v1",
            MODEL_API_KEY="sk-custom",
            MODEL_NAME="llama3.1",
        )
        self.assertEqual(cfg.base_url, "http://localhost:11434/v1")
        self.assertEqual(cfg.api_key, "sk-custom")
        self.assertEqual(cfg.model, "llama3.1")


if __name__ == "__main__":
    unittest.main()
