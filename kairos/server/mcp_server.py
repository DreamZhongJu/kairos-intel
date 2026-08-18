"""Expose Kai Yi's core capabilities as an MCP server.

Lets any MCP client (Claude Desktop, Cherry Studio, LangBot, other agents)
call Kai Yi's tools over stdio. Run with `python -m kai.server.mcp`.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from kairos.agent import runtime as agent_runtime
from kairos.knowledge import tools as kg_tools
from kairos.memory import store as memory_store
from kairos.tools.docs import academic_paper_lookup
from kairos.tools.search import read_public_webpage, web_search
from kairos.tools.skills_loader import native_skill_list, native_skill_load

mcp = MCPServer(
    "kairos",
    description="Kai Yi: a self-hosted personal intelligence assistant (LangGraph + SQLite + Feishu).",
)

_agent_graph: Any | None = None


def _ask(question: str, owner_id: str) -> str:
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = agent_runtime.build_graph()
    return agent_runtime.answer(_agent_graph, question, context="", owner_id=owner_id, chat_id="mcp")


def _memory_search(query: str, owner_id: str) -> str:
    if not owner_id:
        owners = memory_store.all_owners()
        owner_id = owners[0] if owners else "default"
    return memory_store.combined_memory_context(owner_id, query) or "（暂无相关记忆）"


def _latest_report() -> str:
    from kairos.tools.search import latest_report

    return latest_report()


@mcp.tool()
def web_search(query: str) -> str:
    """Search current web and news for a query; returns JSON hits with URLs."""
    import json

    return json.dumps(web_search(query), ensure_ascii=False, default=str)[:12000]


@mcp.tool()
def read_webpage(url: str) -> str:
    """Fetch visible text from a public webpage."""
    import json

    return json.dumps(read_public_webpage(url), ensure_ascii=False, default=str)[:12000]


@mcp.tool()
def paper_lookup(query: str) -> str:
    """Look up a scholarly paper (Crossref/HF); returns hits with links."""
    import json

    return json.dumps(academic_paper_lookup(query), ensure_ascii=False, default=str)[:12000]


@mcp.tool()
def memory_search(query: str, owner_id: str = "") -> str:
    """Search the user's private long-term memory."""
    return _memory_search(query, owner_id)


@mcp.tool()
def report_today() -> str:
    """Return the latest generated daily intelligence report."""
    return _latest_report()


@mcp.tool()
def skill_list() -> str:
    """List external skills the assistant can load."""
    return native_skill_list.invoke({})


@mcp.tool()
def skill_load(name: str) -> str:
    """Load one external skill's full instructions by name."""
    return native_skill_load.invoke({"name": name})


@mcp.tool()
def knowledge_ingest(text: str, title: str = "") -> str:
    """Store text into the local knowledge graph (entities + relations)."""
    return kg_tools.native_knowledge_ingest(text, title)


@mcp.tool()
def knowledge_graph_query(entity: str) -> str:
    """Query the knowledge graph for an entity's 1-2 hop relations."""
    return kg_tools.native_knowledge_graph_query(entity)


@mcp.tool()
def knowledge_search(query: str) -> str:
    """Keyword-search local ingested knowledge chunks."""
    return kg_tools.native_knowledge_search(query)


@mcp.tool()
def ask_agent(question: str, owner_id: str = "mcp-user") -> str:
    """Ask Kai Yi to answer using her LangGraph agent and all her tools."""
    return _ask(question, owner_id)


def main() -> None:
    """Start the MCP server over stdio."""
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()