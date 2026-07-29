"""Shared text helpers used by assistant tools."""

from __future__ import annotations

import re


def plain_text(text: str) -> str:
    """Strip Markdown control syntax for Feishu-friendly plain text."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"!?(?:\[([^\]]+)\]\([^\)]+\))", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()
