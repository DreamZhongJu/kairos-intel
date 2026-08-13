"""Feishu event entrypoint for the research assistant.

All domain logic lives under :mod:`assistant`; this module only wires the
transport, explicit confirmation flows, and the LangGraph runtime together.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

import lark_oapi as lark

from kairos.agent import runtime as agent_runtime
from kairos.channels.feishu import (
    clean_question,
    event_to_dict,
    latest_file_in_chat,
    latest_reference_in_chat,
    message_text,
    recent_chat,
    reply,
    urls_in_message_content,
)
from kairos.infrastructure.settings import APP_ID, APP_SECRET, validate_runtime_settings
from kairos.memory.store import (
    claim_message,
    forget_memories,
    init_db,
    init_memory_runtime,
    memory_owner_id,
    persist_memory_async,
)
from kairos.observability import metrics as obs
from kairos.tools.archive import execute_archive_batch
from kairos.tools.attachments import create_note, prepare_note
from kairos.tools.docs import document_summary
from kairos.reports.scheduler import start_scheduler
from oauth_server import authorization_link, init_oauth, start_oauth_server

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("kairos")

TOOL_GRAPH: Any | None = agent_runtime.build_graph()

_FORGET_MEMORY_TERMS = ("忘记我的记忆", "清除我的记忆", "删除我的记忆")
_FILE_REFERENCE_TERMS = ("这个文件", "刚才的文件", "上面的文件", "最新文件", "附件")
_NOTE_ACTION_TERMS = ("整理", "解读", "总结", "做笔记", "写笔记")


def answer(question: str, context: str, owner_id: str, chat_id: str = "") -> str:
    """Delegate ordinary conversation to the independent LangGraph runtime."""
    return agent_runtime.answer(TOOL_GRAPH, question, context, owner_id, chat_id)


def _is_file_request(question: str) -> bool:
    return any(term in question for term in _FILE_REFERENCE_TERMS)


def _is_forget_memory_request(question: str) -> bool:
    return any(term in question for term in _FORGET_MEMORY_TERMS)


def process_event(data: Any) -> None:
    """Handle one Feishu event in a background worker."""
    payload = event_to_dict(data)
    event = payload.get("event", payload)
    message = event.get("message", {})
    sender = event.get("sender", {})
    if sender.get("sender_type") == "bot" or message.get("message_type") not in {"text", "post", "file"}:
        return

    message_id = str(message.get("message_id", ""))
    chat_id = str(message.get("chat_id", ""))
    owner_id = memory_owner_id(sender, chat_id)
    question = clean_question(message_text(message.get("content", "")))
    LOG.info("received message type=%s id=%s text=%r", message.get("message_type"), message_id, question[:160])

    # A file is deliberately silent: the user decides when it should be read.
    if message.get("message_type") == "file":
        LOG.info("file received in chat=%s; waiting for explicit request", chat_id)
        return
    if not message_id or not chat_id or not question or not claim_message(message_id):
        return

    try:
        if _is_forget_memory_request(question):
            result = f"老师，已清除 {forget_memories(owner_id)} 条长期记忆。"
        elif re.fullmatch(r"确认笔记\s+[A-Za-z0-9]+", question):
            result = create_note(question.split()[-1])
        elif re.fullmatch(r"确认批量归档\s+[A-Za-z0-9]+", question):
            result = execute_archive_batch(question.split()[-1])
        elif _is_file_request(question):
            latest = latest_file_in_chat(chat_id)
            result = prepare_note(*latest) if latest and latest[0] else "老师，我在本群最近 20 条消息中没有找到可读取的附件。"
        elif "授权飞书" in question or "连接飞书" in question:
            link = authorization_link()
            result = f"老师，请点击此链接完成一次个人飞书授权：\n{link}" if link else "授权通道正在启动，请稍后再发送“授权飞书”。"
        else:
            shared_url = re.search(r"https?://\S+", question)
            if shared_url and re.search(r"/(?:wiki|docx|docs)/", shared_url.group(0)):
                # A Feishu document URL is a typed resource, not a heuristic
                # topic match; resolve it directly through the authorized API.
                result = document_summary(shared_url.group(0))
            else:
                if message.get("message_type") == "post":
                    card_urls = urls_in_message_content(message.get("content", ""))
                    public_url = next((url for url in card_urls if not re.search(r"/(?:wiki|docx|docs)/", url)), "")
                    if public_url:
                        question = f"{question}\n\nShared card URL: {public_url}"
                else:
                    reference = latest_reference_in_chat(chat_id)
                    if reference and reference.get("kind") == "webpage" and any(term in question for term in _NOTE_ACTION_TERMS):
                        question = f"{question}\n\nThe user is referring to this shared webpage: {reference['url']}"
                result = answer(question, recent_chat(chat_id), owner_id, chat_id)

        reply(message_id, result)
        if not _is_forget_memory_request(question):
            threading.Thread(
                target=persist_memory_async,
                args=(owner_id, message_id, question, result),
                daemon=True,
            ).start()
    except Exception as exc:
        LOG.exception("request failed")
        try:
            reply(message_id, f"处理失败：{str(exc)[:300]}")
        except Exception:
            LOG.exception("failure reply also failed")


def on_message(data: Any) -> None:
    """Return quickly so Feishu does not retry a long-running LLM request."""
    threading.Thread(target=process_event, args=(data,), daemon=True).start()


def main() -> None:
    """Validate configuration, initialize durable state, and start Feishu WS."""
    global TOOL_GRAPH
    validate_runtime_settings()
    init_db()
    init_memory_runtime()
    obs.init_metrics_table()
    TOOL_GRAPH = agent_runtime.build_graph()
    init_oauth()
    start_oauth_server()
    start_scheduler()
    handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(on_message).build()
    client = lark.ws.Client(APP_ID, APP_SECRET, event_handler=handler, log_level=lark.LogLevel.INFO)
    LOG.info("starting Feishu long connection")
    client.start()


if __name__ == "__main__":
    main()
