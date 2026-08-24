"""FailoverClient unit tests with fake SDK clients."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kairos.infrastructure import llm as kllm  # noqa: E402
from kairos.infrastructure.llm import FailoverClient, ModelConfig, resolve_chain  # noqa: E402


def _cfg(name: str) -> ModelConfig:
    return ModelConfig(base_url=f"https://{name}.test/v1", api_key="k-" + name, model=name, provider=name)


class _FakeSDK:
    def __init__(self, tag: str, behavior):
        self.tag = tag
        self.behavior = behavior  # Exception or return value
        self.calls = 0

    @property
    def chat(self):
        sdk = self

        class _Comp:
            @staticmethod
            def create(**kwargs):
                sdk.calls += 1
                if isinstance(sdk.behavior, Exception):
                    raise sdk.behavior

                class _R:
                    pass

                return _R()

        class _Chat:
            completions = _Comp()

        return _Chat()


class FailoverTest(unittest.TestCase):
    def test_fails_over_to_second_provider(self) -> None:
        fc = FailoverClient([_cfg("primary"), _cfg("backup")], cooldown=60)
        bad, good = _FakeSDK("p", RuntimeError("down")), _FakeSDK("b", "ok")
        with patch.object(FailoverClient, "_sdk", side_effect=[bad, good]):
            resp = fc.create(messages=[{"role": "user", "content": "hi"}])
        self.assertIsNotNone(resp)
        self.assertEqual(fc.model_name(), "backup")
        self.assertEqual(bad.calls, 1)
        self.assertEqual(good.calls, 1)

    def test_sticky_and_cooldown(self) -> None:
        fc = FailoverClient([_cfg("primary"), _cfg("backup")], cooldown=300)
        primary, backup = _FakeSDK("p", RuntimeError("boom")), _FakeSDK("b", None)
        with patch.object(FailoverClient, "_sdk", side_effect=[primary, backup]):
            fc.create()
        self.assertEqual((primary.calls, backup.calls), (1, 1))  # failed over once
        # sticky=backup now; primary in cooldown → next call goes straight there
        with patch.object(FailoverClient, "_sdk", return_value=backup):
            fc.create()
        self.assertEqual((primary.calls, backup.calls), (1, 2))

    def test_resolve_chain_from_env(self) -> None:
        env = {
            "MODEL_FALLBACK_1_PROVIDER": "zen",
            "MODEL_FALLBACK_1_BASE_URL": "https://opencode.ai/zen/v1",
            "MODEL_FALLBACK_1_API_KEY": "sk-test",
            "MODEL_FALLBACK_1_MODEL": "x-preview-f-free",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(kllm.settings, "MODEL_FALLBACKS", [
            {"provider": "zen", "base_url": "https://opencode.ai/zen/v1", "api_key": "sk-test",
             "model": "x-preview-f-free", "proxy": "__inherit__"},
        ]):
            chain = resolve_chain()
        self.assertGreaterEqual(len(chain), 2)
        self.assertEqual(chain[-1].model, "x-preview-f-free")
        self.assertEqual(chain[-1].base_url, "https://opencode.ai/zen/v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
