"""LLM-based entity/relation extraction for the knowledge graph.

Extraction strategy follows the pipeline proven by high-star graph-RAG
projects — Microsoft GraphRAG (microsoft/graphrag) and HKUDS LightRAG — instead
of one single-shot call on the whole document:

1. Split the document into paragraph-aligned ~1200-char chunks.
2. Extract entities + relations per chunk with a strict, type-whitelisted
   prompt and a single JSON schema.
3. When a chunk's output hits the cap, run one "continue" pass to pick up what
   the model left out (GraphRAG calls this a "gleanings" pass).
4. Merge across chunks: normalize entity names, dedupe by canonical name, drop
   code-identifier noise, and keep only relations whose endpoints are real
   extracted entities.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kairos.infrastructure.llm import build_client_optional, model_name, role_client
from kairos.knowledge.engine import _normalize, chunk_text

LOG = logging.getLogger("kairos.knowledge.extract")

# Extraction can target a dedicated (typically faster/cheaper) model through the
# ROUTER_EXTRACT_* env vars, falling back to the global model when unset.
_extract_llm, _extract_model = role_client("extract")
llm = _extract_llm or build_client_optional()
model = _extract_model or model_name()

# Each chunk is one independent LLM call, so a bounded thread pool cuts wall
# clock time roughly linearly (measured ~2.6x at 3 workers vs api.deepseek.com).
_EXTRACT_WORKERS = max(1, min(int(os.getenv("EXTRACT_WORKERS", "3")), 8))

TYPE_WHITELIST = ("机构", "人名", "论文", "技术", "产品", "会议", "地点", "领域", "事件", "项目", "群组")

MAX_ENTITIES_PER_CHUNK = 25
MAX_RELATIONS_PER_CHUNK = 35
MAX_ENTITIES_TOTAL = 300
MAX_RELATIONS_TOTAL = 400

EXTRACT_SYSTEM = "你是知识图谱抽取器。只输出一个 JSON 对象，不要输出任何解释、注释或 Markdown。"

EXTRACT_PROMPT = """从下面的文本中抽取知识图谱的实体与关系。

【实体类型词汇表】实体 type 只能取：机构、人名、论文、技术、产品、会议、地点、领域、事件、项目。

【抽取规则】
1. 只抽取文本中【明确出现】的命名实体，绝不臆造、不推断；实体名要具体、规范（如"项目A"、"FastAPI"、"肉鸽玩法"），不要用"这个系统"、"那个项目"等指代，也不要把事件描述、Bug 症状、字段名或代码片段当实体。
2. 每个实体输出：name（规范实体名）、type（来自词汇表）、description（一句话说明，30 字以内）。
3. 关系两端必须都是【本段已列出的实体】：subject 与 object 用实体的 name 精确对应。
4. 每条关系输出：subject、predicate（具体中文动词，如"使用/实现/依赖/涉及/采用/位于/属于/触发"）、object、confidence（1-10 的整数，越确定越高）。
5. 若文本中有明确的时间线索（如"2023年""去年九月""高二时""上学期"），为对应关系补充 time_start 与 time_end 字段，格式 YYYY-MM-DD 或 YYYY；无法确定具体时间的字段省略。仍在持续的关系省略 time_end。绝不臆造时间。
6. 【语气判断】群聊里充满玩笑与梗。对每条关系判断说话人是否当真：明显是开玩笑、调侃、反讽、自嘲、吹牛、玩梗、夸张整活的关系（如"我是你爹""他考上了清华（阴阳怪气）""这游戏好玩爆了"），输出 "playful": true 并把 confidence 压到 1-3；认真陈述省略 playful 字段。拿不准就当作认真的。
7. 最多输出 __MAX_ENTITIES__ 个实体、__MAX_RELATIONS__ 条关系。只输出一个 JSON 对象，格式：
{"entities":[{"name":"项目A","type":"项目","description":"Java Spring Boot 实现的本地服务端"}],"relations":[{"subject":"项目A","predicate":"使用","object":"Java","confidence":9,"time_start":"2024-03","time_end":"2024-08"},{"subject":"康哥","predicate":"毕业于","object":"清华","confidence":2,"playful":true}]}
若没有可抽取的内容，输出 {"entities":[],"relations":[]}。

文本：
__TEXT__"""

EXTRACT_CONTINUE_PROMPT = """上一次从该文本抽取实体时可能遗漏了一部分。请基于同一文本，补充【之前没有列出】的实体与关系，使用相同 JSON 格式输出，不要重复已经列出的内容。

文本：
__TEXT__"""

# A code-ish identifier is not a knowledge entity: dotted paths (gacha.normal),
# method calls (getIntValue()), lowerCamelCase fields (dropType) and anything
# wrapped in brackets.
_CODE_JUNK_RE = re.compile(r"[()\[\]{}]|[A-Za-z0-9_]+\.[A-Za-z0-9_]")


def _render(prompt: str, chunk: str) -> str:
    return (
        prompt.replace("__MAX_ENTITIES__", str(MAX_ENTITIES_PER_CHUNK))
        .replace("__MAX_RELATIONS__", str(MAX_RELATIONS_PER_CHUNK))
        .replace("__TEXT__", chunk)
    )


def _is_junk_name(name: str) -> bool:
    """Return True for names that look like code identifiers, not entities."""
    name = (name or "").strip()
    if not name or len(name) > 40:
        return True
    if _CODE_JUNK_RE.search(name):
        return True
    # lowerCamelCase single word: dropType / occPercent / poolId
    if re.fullmatch(r"[a-z]+[A-Z][a-zA-Z0-9]*", name):
        return True
    return False


def _parse_json(content: str) -> dict[str, Any]:
    """Parse the model's JSON output, tolerating fences and stray text."""
    text = (content or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except (TypeError, json.JSONDecodeError):
            pass
    return {}


def _repair_json(raw: str) -> dict[str, Any]:
    """One repair pass: ask the model to fix malformed JSON."""
    if llm is None or not raw.strip():
        return {}
    try:
        response = llm.chat.completions.create(
            model=model,
            temperature=0,
            timeout=90,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": f"下面这段输出不是合法 JSON，请修正为合法 JSON，只输出 JSON：\n{raw[:6000]}"},
            ],
        )
        return _parse_json(response.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("json repair failed: %s", type(exc).__name__)
        return {}


def _call_extract(chunk: str, continue_pass: bool = False) -> dict[str, Any]:
    if llm is None:
        return {}
    prompt = _render(EXTRACT_CONTINUE_PROMPT if continue_pass else EXTRACT_PROMPT, chunk)
    try:
        response = llm.chat.completions.create(
            model=model,
            temperature=0,
            timeout=90,
            messages=[{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        LOG.warning("kg extraction call failed: %s", type(exc).__name__)
        return {}
    data = _parse_json(raw)
    if not data and raw.strip():
        data = _repair_json(raw)
    return data


def _clean_entities(raw: list[Any]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        etype = str(item.get("type") or "").strip() or "实体"
        if etype not in TYPE_WHITELIST:
            etype = "实体"
        if _is_junk_name(name):
            continue
        norm = _normalize(name)
        if norm in seen:
            continue
        seen.add(norm)
        entities.append({"name": name[:80], "type": etype, "description": str(item.get("description") or "").strip()[:80]})
    return entities


def _clean_time(value: Any) -> str | None:
    """Keep only plausible date fragments (YYYY or YYYY-MM or YYYY-MM-DD)."""
    text = str(value or "").strip()
    m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
    if not m:
        return None
    year = int(m.group(1))
    if not 1990 <= year <= 2035:  # group-chat era guard against hallucinated years
        return None
    out = str(year)
    if m.group(2):
        month = int(m.group(2))
        if not 1 <= month <= 12:
            return out
        out += f"-{month:02d}"
        if m.group(3):
            day = int(m.group(3))
            if 1 <= day <= 31:
                out += f"-{day:02d}"
    return out


def _merge(all_entities: list[Any], all_relations: list[Any]) -> dict[str, Any]:
    """Normalize, dedupe, and cross-check endpoints across every chunk."""
    entities = _clean_entities(all_entities)
    canonical: dict[str, str] = {_normalize(e["name"]): e["name"] for e in entities}

    relations: list[dict[str, Any]] = []
    seen_r: set[tuple[str, str, str]] = set()
    for item in all_relations:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        if _is_junk_name(subject) or _is_junk_name(obj):
            continue
        subject_norm, object_norm = _normalize(subject), _normalize(obj)
        # GraphRAG/LightRAG only admit relations whose endpoints are extracted
        # entities; map back to the canonical spelling so storage dedupes too.
        if subject_norm not in canonical or object_norm not in canonical:
            continue
        key = (subject_norm, predicate, object_norm)
        if key in seen_r:
            continue
        seen_r.add(key)
        try:
            confidence = int(item.get("confidence") or 1)
        except (TypeError, ValueError):
            confidence = 1
        t_start = _clean_time(item.get("time_start"))
        t_end = _clean_time(item.get("time_end"))
        playful = bool(item.get("playful"))
        if playful:
            confidence = min(confidence, 3)  # jokes never rank as solid facts
        relations.append(
            {
                "subject": canonical[subject_norm],
                "predicate": predicate[:40],
                "object": canonical[object_norm],
                "confidence": max(1, min(10, confidence)),
                "time_start": t_start,
                "time_end": t_end,
                "playful": playful,
            }
        )

    return {
        "entities": entities[:MAX_ENTITIES_TOTAL],
        "relations": relations[:MAX_RELATIONS_TOTAL],
    }


def _extract_chunk(chunk: str) -> tuple[list[Any], list[Any]]:
    """Extract (entities, relations) from one chunk, including the gleanings pass."""
    data = _call_extract(chunk)
    entities = list(data.get("entities") or [])
    relations = list(data.get("relations") or [])
    # Gleanings: if a chunk hit the cap, ask once more for what was missed.
    if len(entities) >= MAX_ENTITIES_PER_CHUNK or len(relations) >= MAX_RELATIONS_PER_CHUNK:
        more = _call_extract(chunk, continue_pass=True)
        entities += more.get("entities") or []
        relations += more.get("relations") or []
    return entities, relations


def extract(text: str, max_text: int = 12000, chunk_limit: int = 1200, workers: int | None = None) -> dict[str, Any]:
    """Return {"entities": [...], "relations": [...]} for a text block.

    Runs chunked extraction with a gleanings pass, then merges the results.
    Chunks are extracted concurrently (bounded thread pool) since each chunk is
    an independent LLM call; set ``workers`` or ``EXTRACT_WORKERS`` to tune it.
    """
    if llm is None:
        return {"entities": [], "relations": []}
    snippet = (text or "").strip()[:max_text]
    chunks = chunk_text(snippet, limit=chunk_limit)
    if not chunks:
        return {"entities": [], "relations": []}

    n_workers = int(workers) if workers is not None else _EXTRACT_WORKERS
    if n_workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(chunks))) as pool:
            chunk_results = list(pool.map(_extract_chunk, chunks))
    else:
        chunk_results = [_extract_chunk(chunk) for chunk in chunks]

    all_entities: list[Any] = []
    all_relations: list[Any] = []
    for entities, relations in chunk_results:
        all_entities.extend(entities)
        all_relations.extend(relations)

    return _merge(all_entities, all_relations)
