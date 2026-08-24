"""OpenAPI REST service that exposes Kai Yi to scriptable/HTTP clients.

Provides a chat endpoint backed by the same LangGraph agent, plus read-only
endpoints for tools, reports and knowledge-graph stats. The OpenAPI document
is served at /openapi.json.
"""

from __future__ import annotations

import json
import secrets

from flask import Flask, jsonify, request

from kairos.agent import runtime as agent_runtime
from kairos.infrastructure import settings
from kairos.knowledge import engine as kg_engine
from kairos.knowledge import ingest as kg_ingest

GRAPH: object | None = None


def _graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = agent_runtime.build_graph()
    return GRAPH


def _authorized() -> bool:
    """Machine-to-machine endpoints accept an optional shared token."""
    token = settings.KAIROS_API_TOKEN
    if not token:
        return True
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {token}" or request.headers.get("X-Token", "") == token


def build_openapi() -> dict:
    return {
        "openapi": "3.0.3",
        "info": {"title": "Kairos", "version": "1.0.0", "description": "自托管个人情报助手（凯伊）的 HTTP 接口"},
        "paths": {
            "/health": {
                "get": {
                    "summary": "存活检查",
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/api/tools": {
                "get": {
                    "summary": "列出可用工具",
                    "responses": {"200": {"description": "工具名+描述列表"}},
                }
            },
            "/api/chat": {
                "post": {
                    "summary": "向凯伊提问",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "owner_id": {"type": "string", "default": "api-user"},
                                        "chat_id": {"type": "string", "default": ""},
                                    },
                                    "required": ["question"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "回答文本 + request_id"}},
                }
            },
            "/api/reports": {
                "get": {
                    "summary": "列出已生成的日报/周报/月报文件",
                    "responses": {"200": {"description": "文件路径列表"}},
                }
            },
            "/api/knowledge/stats": {
                "get": {
                    "summary": "知识图谱规模统计",
                    "responses": {"200": {"description": "docs/entities/relations 计数"}},
                }
            },
            "/api/knowledge/ingest": {
                "post": {
                    "summary": "批量入库聊天窗口并更新知识图谱（供 Koishi 采集端调用）",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "channel_id": {"type": "string", "description": "群号"},
                                        "source": {"type": "string", "default": "qq"},
                                        "title": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "user_id": {"type": "string"},
                                                    "nickname": {"type": "string"},
                                                    "time": {"type": "string"},
                                                    "text": {"type": "string"},
                                                },
                                                "required": ["user_id", "text"],
                                            },
                                        },
                                    },
                                    "required": ["channel_id", "messages"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "实体/关系/成员统计"}},
                }
            },
            "/api/knowledge/query": {
                "get": {
                    "summary": "轻量知识检索（关键词片段 + 图谱子图，毫秒级）",
                    "parameters": [
                        {"name": "q", "in": "query", "schema": {"type": "string"}, "description": "关键词"},
                        {"name": "entity", "in": "query", "schema": {"type": "string"}, "description": "实体名/QQ号"},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 6}},
                    ],
                    "responses": {"200": {"description": "chunks + graph + answer 文本"}},
                }
            },
        },
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return jsonify({"service": "kairos-api", "openapi": "/openapi.json"})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/openapi.json")
    def openapi():
        return jsonify(build_openapi())

    @app.get("/api/tools")
    def tools():
        return jsonify(
            [
                {"name": t.name, "description": t.description}
                for t in agent_runtime.ACTIVE_TOOLS
            ]
        )

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question required"}), 400
        owner_id = str(data.get("owner_id") or "api-user")
        chat_id = str(data.get("chat_id") or "")
        request_id = secrets.token_hex(8)
        answer = agent_runtime.answer(_graph(), question, context="", owner_id=owner_id, chat_id=chat_id)
        return jsonify({"request_id": request_id, "answer": answer})

    @app.get("/api/reports")
    def reports():
        files = sorted((p.name for p in settings.REPORT_DIR.glob("structured-*.md")))
        periodic = sorted(p.name for p in (settings.REPORT_DIR / "periodic").glob("*.md")) if (settings.REPORT_DIR / "periodic").is_dir() else []
        return jsonify({"daily": files, "periodic": periodic})

    @app.get("/api/knowledge/stats")
    def knowledge_stats():
        try:
            kg_engine.init()
            return jsonify(kg_engine.stats())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/knowledge/neo4j")
    def knowledge_neo4j_status():
        try:
            from kairos.knowledge import graph_store

            if not graph_store.available():
                return jsonify({"neo4j": "unavailable"})
            with graph_store._driver().session() as s:
                counts = s.run(
                    "MATCH (e:Entity) WITH count(e) AS entities "
                    "OPTIONAL MATCH ()-[r:REL]->() WITH entities, count(r) AS relations "
                    "OPTIONAL MATCH (c:Chunk) RETURN entities, relations, count(c) AS chunks"
                ).single()
            return jsonify({"neo4j": "ok", **dict(counts)})
        except Exception as exc:
            return jsonify({"neo4j": "error", "error": str(exc)}), 200

    @app.post("/api/knowledge/ingest")
    def knowledge_ingest():
        if not _authorized():
            return jsonify({"error": "unauthorized"}), 401
        data = request.get_json(silent=True) or {}
        channel_id = str(data.get("channel_id") or data.get("group_id") or "").strip()
        messages = data.get("messages")
        if not channel_id:
            return jsonify({"error": "channel_id required"}), 400
        if not isinstance(messages, list) or len(messages) < 2:
            return jsonify({"error": "messages must be a list with at least 2 items"}), 400
        try:
            result = kg_ingest.ingest_chat_window(
                channel_id,
                messages,
                source=str(data.get("source") or "qq"),
                title=str(data.get("title") or ""),
            )
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/knowledge/query")
    def knowledge_query():
        if not _authorized():
            return jsonify({"error": "unauthorized"}), 401
        try:
            limit = int(request.args.get("limit", "6"))
        except ValueError:
            limit = 6
        result = kg_ingest.query_knowledge(
            q=request.args.get("q", ""),
            entity=request.args.get("entity", ""),
            limit=limit,
        )
        return jsonify(result)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8095, threaded=True)