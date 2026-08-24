"""PII desensitization for group-chat ingestion.

Channels listed in ``PRIVACY_EXEMPT_GROUPS`` (default: 830070676 only) are
ingested verbatim. Every other channel is sanitized before anything touches
the database or the LLM:

1. Nicknames become stable pseudonyms ("群成员" + last 4 digits of the QQ id),
   so person nodes stay linkable across windows without exposing real names.
2. Nickname occurrences inside message bodies (@mentions, quoted names) are
   replaced with the same pseudonyms.
3. Regex masking of high-risk literals: 身份证号、手机号、邮箱、银行卡类长数字串.
"""

from __future__ import annotations

import re
from typing import Any

ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Remaining long digit runs after ID/phone removal: bank cards, order ids…
LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

MASK = "[已脱敏]"


def mask_text(text: str) -> str:
    """Mask identity literals; never leaves raw digits of those classes behind."""
    text = ID_CARD_RE.sub(MASK, text)
    text = EMAIL_RE.sub(MASK, text)
    text = PHONE_RE.sub(MASK, text)
    text = LONG_DIGITS_RE.sub(MASK, text)
    return text


def _pseudonym(uid: str) -> str:
    return f"群成员{(uid or '0')[-4:]}"


def pseudonymize_messages(
    channel_id: str,
    messages: list[dict[str, Any]],
    exempt_groups: set[str],
) -> list[dict[str, Any]]:
    """Return a sanitized copy of the window for non-exempt channels."""
    channel_id = str(channel_id or "").strip()
    if channel_id in exempt_groups:
        return messages

    uids = sorted({str(m.get("user_id") or "").strip() for m in messages} - {""})
    uid_map = {uid: _pseudonym(uid) for uid in uids}
    # Original nickname -> pseudonym, longest first so overlapping names
    # replace without leaving partial fragments behind.
    nick_pairs = sorted(
        {
            (str(m.get("nickname") or "").strip(), uid_map[str(m.get("user_id")).strip()])
            for m in messages
            if m.get("user_id") and str(m.get("nickname") or "").strip()
            and not str(m.get("nickname")).strip().isdigit()
        },
        key=lambda kv: len(kv[0]),
        reverse=True,
    )

    out: list[dict[str, Any]] = []
    for item in messages:
        cleaned = dict(item)
        uid = str(item.get("user_id") or "").strip()
        cleaned["nickname"] = uid_map.get(uid, "群成员")
        text = str(item.get("text") or item.get("content") or "")
        for original, pseudo in nick_pairs:
            text = text.replace(original, pseudo)
        cleaned["text"] = mask_text(text)
        cleaned.pop("content", None)
        out.append(cleaned)
    return out
