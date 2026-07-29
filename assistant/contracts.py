"""Shared data contracts for assistant requests and responses."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AssistantRequest(TypedDict, total=False):
    """Normalized request passed from a channel adapter into the app layer."""

    source: Literal["feishu", "web", "cli"]
    chat_id: str
    message_id: str
    owner_id: str
    text: str
    sender: dict[str, Any]
    attachments: list[dict[str, Any]]
    metadata: dict[str, Any]


class ToolResult(TypedDict, total=False):
    """Uniform result returned by a tool node."""

    ok: bool
    name: str
    summary: str
    payload: dict[str, Any]
    error: str


class AssistantResponse(TypedDict, total=False):
    """User-facing response emitted by the agent pipeline."""

    text: str
    summary: str
    sources: list[str]
    metadata: dict[str, Any]

