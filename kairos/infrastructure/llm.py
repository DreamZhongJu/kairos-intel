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
    proxy: str = ""


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
    return ModelConfig(base_url=base_url, api_key=api_key, model=model, provider=provider, proxy=settings.MODEL_PROXY)


def _http_client_for(proxy: str) -> Any | None:
    """Build an explicit httpx client for the given proxy policy.

    ``""``/``direct`` → force a direct connection ignoring env proxies;
    other non-empty values → route through that proxy URL.
    """
    import httpx

    if proxy in ("", "direct"):
        return httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0))
    return httpx.Client(proxy=proxy, timeout=httpx.Timeout(120.0))


def build_client(config: ModelConfig | None = None) -> OpenAI:
    """Return an OpenAI-compatible client (constructed even without a key)."""
    config = config or resolve_model_config()
    kwargs: dict[str, Any] = {"api_key": config.api_key, "max_retries": settings.MODEL_MAX_RETRIES}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.proxy:
        kwargs["http_client"] = _http_client_for(config.proxy)
    return OpenAI(**kwargs)


# --- Failover chain ---------------------------------------------------------

class _ChatNamespace:
    """Duck-typed ``client.chat.completions`` surface for :class:`FailoverClient`."""

    def __init__(self, create_fn):
        self._create_fn = create_fn

    @property
    def completions(self) -> "_ChatNamespace":
        return self

    def create(self, **kwargs: Any):
        return self._create_fn(**kwargs)


class FailoverClient:
    """Duck-typed chat client that walks a provider chain on failure.

    The first config is the primary; the rest are tried in order when a call
    fails. A provider that failed enters a cooldown and is skipped until it
    expires; the last successful provider is sticky so healthy paths are not
    re-paid for on every call.
    """

    def __init__(self, configs: list[ModelConfig], cooldown: int | None = None):
        from time import time as _now

        self._configs = [c for c in configs if c.api_key]
        if not self._configs:
            raise ValueError("failover chain needs at least one configured provider")
        self._clients: dict[int, OpenAI] = {}
        self._sticky = 0
        self._cooldown_until = [0.0] * len(self._configs)
        self._cooldown = cooldown if cooldown is not None else settings.MODEL_FAILOVER_COOLDOWN
        self._now = _now
        self.max_retries = 1  # informational; real retries live per-SDK-client

    def _sdk(self, idx: int) -> OpenAI:
        client = self._clients.get(idx)
        if client is None:
            cfg = self._configs[idx]
            kwargs: dict[str, Any] = {"api_key": cfg.api_key, "max_retries": 1}
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            if cfg.proxy:
                kwargs["http_client"] = _http_client_for(cfg.proxy)
            client = OpenAI(**kwargs)
            self._clients[idx] = client
        return client

    @staticmethod
    def _usable(resp: Any) -> bool:
        """A response is usable when it carries visible content or tool calls.

        Reasoning providers can burn the whole token budget on hidden thinking
        and return an empty ``content`` with a normal HTTP status — treat that
        as a soft failure and fall through to the next provider.
        """
        try:
            msg = resp.choices[0].message
        except Exception:  # noqa: BLE001
            return True  # non-standard shape: don't second-guess
        if getattr(msg, "tool_calls", None):
            return True
        content = (getattr(msg, "content", "") or "")
        return bool(str(content).strip())

    def create(self, **kwargs: Any):
        now = self._now()
        order = [self._sticky] + [i for i in range(len(self._configs)) if i != self._sticky]
        last_exc: Exception | None = None
        last_resp: Any = None
        for pos, idx in enumerate(order):
            if self._cooldown_until[idx] > now:
                continue
            try:
                call_kwargs = dict(kwargs)
                call_kwargs["model"] = self._configs[idx].model
                response = self._sdk(idx).chat.completions.create(**call_kwargs)
                self._sticky = idx
                if self._usable(response) or pos == len(order) - 1:
                    return response
                last_resp = response  # empty output: try the next provider
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._cooldown_until[idx] = self._now() + self._cooldown
        if last_resp is not None:
            return last_resp
        if last_exc is not None and self._cooldown_until[self._sticky] <= now:
            pass  # a non-cooled provider already raised; nothing more to try
        # everything cooling down (or all-empty): force-try sticky, never die
        try:
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = self._configs[self._sticky].model
            return self._sdk(self._sticky).chat.completions.create(**call_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise last_exc or exc or RuntimeError("no model provider available")

    @property
    def chat(self) -> _ChatNamespace:
        return _ChatNamespace(self.create)

    def model_name(self) -> str:
        return self._configs[self._sticky].model


def resolve_chain() -> list[ModelConfig]:
    """Primary model config plus any MODEL_FALLBACK_<n>_* providers."""
    chain = [resolve_model_config()]
    seen = {(c.base_url, c.model) for c in chain}
    for spec in settings.MODEL_FALLBACKS:
        preset = PROVIDER_PRESETS.get(spec["provider"], {})
        base_url = spec["base_url"] or preset.get("base_url", "")
        api_key = spec["api_key"] or os.getenv(preset.get("env_key", ""), "").strip()
        model = spec["model"]
        proxy = spec["proxy"]
        if proxy == "__inherit__":
            proxy = settings.MODEL_PROXY
        elif proxy in ("none", "direct"):
            proxy = "direct"
        if not api_key or not model:
            continue
        cfg = ModelConfig(base_url=base_url, api_key=api_key, model=model,
                          provider=spec["provider"] or "custom", proxy=proxy)
        if (cfg.base_url, cfg.model) not in seen:
            chain.append(cfg)
            seen.add((cfg.base_url, cfg.model))
    return chain


def build_client_optional(config: ModelConfig | None = None) -> OpenAI | FailoverClient | None:
    """Return a client, or None when no API key is configured (lazy-safe).

    When MODEL_FALLBACK_<n>_* providers are configured this returns a
    :class:`FailoverClient` covering the whole chain; callers keep using the
    plain ``client.chat.completions.create(...)`` interface unchanged.
    """
    if config is None and settings.MODEL_FALLBACKS:
        chain = resolve_chain()
        if len(chain) > 1:
            return FailoverClient(chain)
        config = chain[0]
    else:
        config = config or resolve_model_config()
    if not config.api_key:
        return None
    return build_client(config)


def model_name() -> str:
    return resolve_model_config().model


# --- Multi-model routing ---------------------------------------------------

STRONG_TERMS = (
    "调研", "深度", "复杂", "分析报告", "综述", "论文", "研究", "报告",
    "代码", "review", "research", "survey", "litreview", "grant", "立项",
    "文献", "写作", "润色", "评审", "审稿", "rebat", "latex", "slides",
)


def route_question(question: str) -> str:
    """Heuristic role classifier: 'strong' for complex work, 'default' otherwise."""
    q = (question or "").lower()
    if any(term in q for term in STRONG_TERMS):
        return "strong"
    return "default"


def role_client(role: str) -> tuple[OpenAI | None, str]:
    """Return (client, model) for a named role, or (None, '') if unconfigured.

    Config via env: ROUTER_<ROLE>_PROVIDER, ROUTER_<ROLE>_MODEL,
    ROUTER_<ROLE>_BASE_URL, ROUTER_<ROLE>_API_KEY.
    When unset, the caller should fall back to the global :func:`build_client_optional` / :func:`model_name`.
    """
    role = role.strip().lower()
    if not role:
        return None, ""
    provider = os.getenv(f"ROUTER_{role.upper()}_PROVIDER", "").strip()
    model = os.getenv(f"ROUTER_{role.upper()}_MODEL", "").strip()
    base_url = os.getenv(f"ROUTER_{role.upper()}_BASE_URL", "").strip()
    api_key = os.getenv(f"ROUTER_{role.upper()}_API_KEY", "").strip()
    if not (provider or model or base_url or api_key):
        return None, ""
    preset = PROVIDER_PRESETS.get(provider or settings.MODEL_PROVIDER, {})
    actual_base = base_url or preset.get("base_url", "")
    actual_key = api_key or settings.MODEL_API_KEY or os.getenv(
        preset.get("env_key", ""), ""
    ).strip() or settings.DEEPSEEK_KEY
    if not actual_key and not api_key:
        return None, ""
    cfg = ModelConfig(
        base_url=actual_base,
        api_key=actual_key,
        model=model or settings.MODEL_NAME or settings.MODEL,
        provider=provider or "default",
        proxy=settings.MODEL_PROXY,
    )
    if settings.MODEL_FALLBACKS:
        chain = [cfg] + [c for c in resolve_chain() if (c.base_url, c.model) != (cfg.base_url, cfg.model)]
        return FailoverClient(chain), cfg.model
    return build_client_optional(cfg), cfg.model
