"""Knowledge-graph agent tools: ingest, search, and graph-query."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from kairos.knowledge import engine
from kairos.knowledge import extract as kg_extract

LOG = logging.getLogger("kairos.knowledge.tools")


def _format_graph(result: dict) -> str:
    if not result.get("found"):
        return f"知识图谱中没有找到「{result.get('entity')}」关联；你可以先让我用 knowledge_ingest 把资料入库再查询。"
    nodes = {n["id"]: n["name"] for n in result.get("nodes", [])}
    lines = [f"图谱中「{result.get('entity')}」的关联（2 跳内）："]
    for edge in result.get("edges", []):
        subject = nodes.get(edge.get("subject"), str(edge.get("subject")))
        obj = nodes.get(edge.get("object"), str(edge.get("object")))
        predicate = edge.get("predicate", "相关")
        lines.append(f"- {subject} --[{predicate}]--> {obj}")
    return "\n".join(lines) if len(lines) > 1 else "图谱中有该实体，但没有更多关联关系。"


@tool("knowledge_ingest")
def native_knowledge_ingest(text: str, title: str = "") -> str:
    """把一段文本/笔记/网页内容摄入本地知识库，自动抽取实体关系并建图。"""
    text = (text or "").strip()
    if len(text) < 20:
        return "请提供至少 20 字的文本内容用于入库。"
    title = (title or text[:40]).strip()
    engine.init()
    doc_id = engine.add_document(title, text)
    extracted = kg_extract.extract(text)
    entities = extracted.get("entities", [])
    relations = extracted.get("relations", [])
    for ent in entities:
        try:
            engine.upsert_entity(ent["name"], ent["type"])
        except Exception:
            pass
    for rel in relations:
        try:
            engine.add_relation(
                rel["subject"], rel["predicate"], rel["object"], confidence=int(rel.get("confidence", 1))
            )
        except Exception:
            pass
    return (
        f"已入库《{title}》（文档 #{doc_id}）：新增 {len(entities)} 个实体、"
        f"{len(relations)} 条关系。当前知识库：{engine.stats()}"
    )


@tool("knowledge_graph_query")
def native_knowledge_graph_query(entity: str) -> str:
    """查询知识图谱中某个实体的关联关系（1-2 跳）。"""
    engine.init()
    result = engine.graph_query(entity)
    if not result.get("found"):
        # Exact (alias-aware) lookup missed: offer fuzzy candidates so the
        # model can re-query with a name that actually exists in the graph.
        candidates = engine.search_entities(entity)
        if len(candidates) == 1:
            result = engine.graph_query(candidates[0]["name"])
        elif candidates:
            names = "、".join(c["name"] for c in candidates[:8])
            return f"图谱中没有叫「{entity}」的实体，最接近的候选：{names}。请用确切名称再查一次。"
    return _format_graph(result)


@tool("local_knowledge_search")
def native_local_knowledge_search(query: str) -> str:
    """检索本地已入库的知识片段（关键词匹配，区别于飞书云文档的 knowledge_search）。"""
    query = (query or "").strip()
    if not query:
        return "请输入检索关键词。"
    engine.init()
    hits = engine.keyword_search(query)
    if not hits:
        return f"本地知识库中没有匹配「{query}」的内容。"
    lines = []
    for hit in hits[:6]:
        title = hit.get("title") or "未命名文档"
        text = (hit.get("text") or "")[:400]
        lines.append(f"[{title}]\n{text}")
    return "\n\n".join(lines)