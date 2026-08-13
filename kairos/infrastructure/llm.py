"""OpenAI-compatible model provider resolution and client factory.

Centralizes the previously duplicated ``OpenAI(api_key=DEEPSEEK_KEY,
base_url="https://api.deepseek.com")`` construction so the assistant can target
any OpenAI-compatible provider through environment configuration while keeping
the legacy ``DEEPSEEK_API_KEY`` / ``DEEPSEEK_MODEL`` variables working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from kairos.infrastructure import settings

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "env_key": "DEEPSEEK_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "env_key": "OPENAI_API_KEY"},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "env_key": "DASHSCOPE_API_KEY"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "env_key": "MOONSHOT_API_KEY"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "env_key": "ZHIPUAI_API_KEY"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "env_key": "SILICONFLOW_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "env_key": "OLLAMA_API_KEY"},
}


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    provider: str


def resolve_model_config() -> ModelConfig:
    provider = settings.MODEL_PROVIDER
    preset = PROVIDER_PRESETS.get(provider, {})
    base_url = settings.MODEL_BASE_URL or preset.get("base_url", "")
    api_key = settings.MODEL_API_KEY
    if not api_key and preset:
        api_key = os.getenv(preset.get("env_key", ""), "").strip()
    if not api_key:
        api_key = settings.DEEPSEEK_KEY
    model = settings.MODEL_NAME or settings.MODEL
    return ModelConfig(base_url=base_url, api_key=api_key, model=model, provider=provider)


def build_client(config: ModelConfig | None = None) -> OpenAI:
    """Return an OpenAI-compatible client (constructed even without a key)."""
    config = config or resolve_model_config()
    kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def build_client_optional(config: ModelConfig | None = None) -> OpenAI | None:
    """Return a client, or None when no API key is configured (lazy-safe)."""
    config = config or resolve_model_config()
    if not config.api_key:
        return None
    return build_client(config)


def model_name() -> str:
    return resolve_model_config().model
