"""LLM-based entity/relation extraction for the knowledge graph."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from kairos.infrastructure.llm import build_client_optional, model_name

LOG = logging.getLogger("kairos.knowledge.extract")

llm = build_client_optional()

TYPE_WHITELIST = ("机构", "人名", "论文", "技术", "产品", "会议", "地点", "领域", "事件", "项目")

EXTRACT_SYSTEM = (
    "你是知识图谱实体抽取器。只输出 JSON，不要输出任何其他文字。"
)
EXTRACT_PROMPT = """从下面的文本中提取知识图谱三元组。
规则：
1. 只提取文本中【明确出现】的实体，不要臆造；实体类型限定为：机构 / 人名 / 论文 / 技术 / 产品 / 会议 / 地点 / 领域 / 事件 / 项目。
2. 关系 predicate 用具体中文动词，如：属于、位于、发表、研究、由…提出、任职于、获得、包含 等。
3. relation 的 subject 与 object 必须来自该文本中出现的实体（或直接可从文本辨认的实体名）。
4. 只输出一个 JSON 对象，结构为：
{{"entities":[{{"name":"实体名","type":"机构"}}],"relations":[{{"subject":"A","predicate":"关系","object":"B"}}]}}
最多输出 30 个实体、40 条关系；没有则输出 {"entities":[],"relations":[]}。

文本：
{text}"""


def _parse_json(content: str) -> dict[str, Any]:
    text = content or ""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}


def extract(text: str, max_text: int = 12000) -> dict[str, Any]:
    """Return {"entities": [...], "relations": [...]} for a text block."""
    if llm is None:
        return {"entities": [], "relations": []}
    snippet = (text or "").strip()[:max_text]
    if not snippet:
        return {"entities": [], "relations": []}
    try:
        response = llm.chat.completions.create(
            model=model_name(),
            temperature=0.1,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": f"文本：\n{snippet}"},
            ],
        )
        data = _parse_json(response.choices[0].message.content)
    except Exception as exc:
        LOG.warning("kg extraction failed: %s", type(exc).__name__)
        return {"entities": [], "relations": []}

    entities = data.get("entities") or []
    relations = data.get("relations") or []
    cleaned_entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in entities:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        etype = str(item.get("type") or "").strip() or "实体"
        if etype not in TYPE_WHITELIST:
            etype = "实体"
        if name and (name, etype) not in seen:
            seen.add((name, etype))
            cleaned_entities.append({"name": name[:80], "type": etype})

    cleaned_relations: list[dict[str, str]] = []
    seen_r: set[tuple[str, str, str]] = set()
    known = {e["name"] for e in cleaned_entities}
    for item in relations:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        if subject not in known:
            known.add(subject)
            cleaned_entities.append({"name": subject[:80], "type": "实体"})
        if obj not in known:
            known.add(obj)
            cleaned_entities.append({"name": obj[:80], "type": "实体"})
        key = (subject, predicate, obj)
        if key not in seen_r:
            seen_r.add(key)
            cleaned_relations.append({"subject": subject, "predicate": predicate, "object": obj})

    return {"entities": cleaned_entities[:30], "relations": cleaned_relations[:30]}