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

GRAPH: object | None = None


def _graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = agent_runtime.build_graph()
    return GRAPH


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

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8095, threaded=True)