"""Offline, reproducible evaluation for the Kairós.

Two parts:
  1. Knowledge Q&A evaluation: retrieve from the project's own docs, generate
     answers with DeepSeek, then use DeepSeek as a judge to score faithfulness
     and answer relevancy. Also computes context hit@3 against keyword anchors.
  2. Tool routing evaluation: given the project's registered tools, ask the LLM
     to pick a tool for each user intent and measure top-1 / top-3 accuracy.

Requires the project's .env (DEEPSEEK_API_KEY) and network access to DeepSeek.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", r"D:\Document\MyCodeProject\kairos"))
MODEL = os.environ.get("EVAL_MODEL", "")
sys.path.insert(0, str(PROJECT_ROOT))
from kairos.infrastructure.llm import build_client, model_name  # noqa: E402
MAX_CHUNK = 700


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def get_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    return build_client()


def get_model(client: OpenAI) -> str:
    if MODEL:
        return MODEL
    return model_name()


# --------------------------------------------------------------------------
# Document loading and lightweight retrieval (char bigram TF-IDF cosine)
# --------------------------------------------------------------------------

def load_documents(tool_blocks: list[str] | None = None) -> list[str]:
    raw: list[str] = []
    for rel in ("架构设计说明.md", "README.md"):
        p = PROJECT_ROOT / rel
        if p.exists():
            raw.append(p.read_text(encoding="utf-8", errors="replace"))
    paragraphs: list[str] = []
    for doc in raw:
        for para in re.split(r"\n\s*\n", doc):
            para = para.strip()
            if len(para) >= 30:
                paragraphs.append(para)
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) <= MAX_CHUNK:
            buf = (buf + "\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    if tool_blocks:
        chunks.extend(tool_blocks)
    # Distractor chunks: content not covered by the project docs, so retrieval
    # and faithfulness have real discriminative power.
    chunks.extend(
        [
            "MySQL 的 InnoDB 引擎使用 B+ 树作为聚簇索引结构，主键索引的叶子节点直接存储整行数据，"
            "二级索引的叶子节点存储主键值，因此回表查询需要额外一次索引查找。"
            "合理设计联合索引可以减少回表，覆盖索引则可以完全避免回表。",
            "卷积神经网络通过局部感受野与权值共享降低参数量，池化层提供平移不变性。"
            "ResNet 引入残差连接缓解深层网络的梯度消失问题，使数十层甚至上百层的网络可以稳定训练。",
        ]
    )
    return chunks


def char_ngrams(text: str, n: int = 2) -> list[str]:
    norm = re.sub(r"\s+", "", text.lower())
    return [norm[i : i + n] for i in range(len(norm) - n + 1)] or ([norm] if norm else [])


def _tf_vector(grams: list[str]) -> dict[str, int]:
    v: dict[str, int] = {}
    for g in grams:
        v[g] = v.get(g, 0) + 1
    return v


def retrieve(query: str, chunks: list[str], top_k: int = 3) -> list[tuple[float, int]]:
    qv = _tf_vector(char_ngrams(query))
    qnorm = math.sqrt(sum(v * v for v in qv.values()))
    scored: list[tuple[float, int]] = []
    for idx, chunk in enumerate(chunks):
        cv = _tf_vector(char_ngrams(chunk))
        common = set(qv) & set(cv)
        dot = sum(qv[g] * cv[g] for g in common)
        cnorm = math.sqrt(sum(v * v for v in cv.values()))
        if qnorm == 0 or cnorm == 0:
            continue
        scored.append((dot / (qnorm * cnorm), idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# --------------------------------------------------------------------------
# DeepSeek helpers
# --------------------------------------------------------------------------

def chat(client: OpenAI, model: str, messages: list[dict], json_mode: bool = False) -> str:
    kwargs: dict = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 700}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


SYSTEM = "你是严谨的评估助手。只输出 JSON，不要输出其他内容。"


def generate_answer(client: OpenAI, model: str, question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "你是 Kairós，基于给定的参考资料用中文回答用户问题。只依据资料内容，不要编造；资料不足时明确说明。回答控制在 120 字以内。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}\n\n请回答："},
    ]
    for _ in range(2):
        out = chat(client, model, messages).strip()
        if out:
            return out
    return "（生成失败：模型返回空回答。）"


def judge_answer(client: OpenAI, model: str, question: str, context: str, answer: str) -> dict:
    prompt = (
        "请评估一次知识问答。\n"
        f"问题：{question}\n"
        f"参考资料（上下文）：\n{context[:2000]}\n"
        f"模型回答：{answer}\n\n"
        "请输出 JSON：\n"
        '{"faithfulness": 0到1的小数（回答是否完全基于参考资料、无编造）,\n'
        '"relevancy": 0到1的小数（回答是否直接命中问题核心）}'
    )
    out = chat(client, model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], json_mode=True)
    try:
        data = json.loads(out)
        return {
            "faithfulness": round(float(data.get("faithfulness", 0)), 3),
            "relevancy": round(float(data.get("relevancy", 0)), 3),
        }
    except Exception:
        return {"faithfulness": 0.0, "relevancy": 0.0, "raw": out[:200]}


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

QA_CASES = [
    {
        "question": "Kairós 可以访问哪些信息来源？",
        "anchors": ["公开网络", "论文", "云文档", "知识库", "日程"],
    },
    {
        "question": "助手在什么条件下才会创建或修改飞书云文档？",
        "anchors": ["明确指令", "显式"],
    },
    {
        "question": "长期记忆存储在什么地方，如何隔离？",
        "anchors": ["SQLite", "Claude-Mem", "范围隔离"],
    },
    {
        "question": "Agent 如何决定调用哪些工具？",
        "anchors": ["LLM", "tool calling", "自主"],
    },
    {
        "question": "在群聊中如何避免机器人误响应普通聊天？",
        "anchors": ["@", "触发"],
    },
    {
        "question": "单次请求的工具调用有什么约束，防止循环调用？",
        "anchors": ["最大工具轮数", "循环"],
    },
    {
        "question": "当所有联网搜索来源都失败时，助手应该怎样回答？",
        "anchors": ["未能验证实时信息", "标注"],
    },
    {
        "question": "写入工具（如归档）失败时的处理策略是什么？",
        "anchors": ["不自动无限重试", "失败原因"],
    },
    {
        "question": "批量归档云文档前需要做什么？",
        "anchors": ["预览", "确认"],
    },
    {
        "question": "项目的依赖方向是如何约束的，低层模块可以导入高层模块吗？",
        "anchors": ["依赖方向", "不能", "低层"],
    },
    {
        "question": "Kairós 的日均 token 消耗是多少？",
        "anchors": [],
        "negative": True,
    },
    {
        "question": "助手上线以来有多少注册用户？",
        "anchors": [],
        "negative": True,
    },
    {
        "question": "DeepSeek API 的计费标准是什么？",
        "anchors": [],
        "negative": True,
    },
]

TOOL_ROUTE_CASES = [
    ("最近 GitHub 上有什么开源 Agent 框架值得关注？", ["github_research"]),
    ("我今天有什么日程安排？", ["today_schedule"]),
    ("把这份总结归档到科研知识库", ["archive_to_knowledge_base"]),
    ("帮我读一下这篇 arXiv 论文的摘要", ["paper_lookup", "huggingface_papers", "read_webpage"]),
    ("我之前的 RAG 笔记结论是什么？", ["knowledge_search", "memory_search"]),
    ("搜一下 B 站上关于 LangGraph 的视频", ["bilibili_search"]),
    ("X 上最近关于 GRPO 的讨论", ["x_search"]),
    ("最新机器翻译研究进展", ["web_search", "semantic_web_search", "paper_lookup"]),
    ("把这篇文章内容保存为云文档", ["save_cloud_document"]),
    ("我收藏的 YouTube 视频讲了什么", ["youtube_video_details"]),
]


def main() -> int:
    client = get_client()
    model = get_model(client)
    print(f"model={model} project={PROJECT_ROOT}", flush=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    from kairos.agent.runtime import NATIVE_OPENAI_TOOLS

    tool_blocks = [
        f"工具 {t['function']['name']}：{t['function']['description']}" for t in NATIVE_OPENAI_TOOLS
    ]
    chunks = load_documents(tool_blocks=tool_blocks)
    print(f"doc chunks={len(chunks)}", flush=True)

    # ---- Part 1: RAG Q&A ----
    qa_rows = []
    for case in QA_CASES:
        q = case["question"]
        hits = retrieve(q, chunks, top_k=3)
        ctx = "\n---\n".join(chunks[i] for _, i in hits)
        anchors = case.get("anchors", [])
        hit3 = (not anchors) or any(any(a in chunks[i] for a in anchors) for _, i in hits)
        answer = generate_answer(client, model, q, ctx)
        judge = judge_answer(client, model, q, ctx, answer)
        row = {
            "question": q,
            "top3_hit": hit3,
            "negative": case.get("negative", False),
            "retrieved_indices": [i for _, i in hits],
            "answer": answer[:180],
            **judge,
        }
        qa_rows.append(row)
        print(
            f"[QA {len(qa_rows):02d}] neg={row['negative']} hit3={hit3} faith={judge.get('faithfulness')} rel={judge.get('relevancy')}",
            flush=True,
        )

    # ---- Part 2: tool routing ----
    tool_names = [t["function"]["name"] for t in NATIVE_OPENAI_TOOLS]
    tool_desc = "\n".join(f"- {t['function']['name']}: {t['function']['description'][:120]}" for t in NATIVE_OPENAI_TOOLS)
    route_rows = []
    for question, expected in TOOL_ROUTE_CASES:
        prompt = (
            "你是工具路由评估。根据用户意图，从工具列表中选择最合适的 1-3 个工具。\n"
            f"工具列表：\n{tool_desc}\n\n"
            f"用户意图：{question}\n\n"
            '只输出 JSON：{"tools": ["工具名", ...]}'
        )
        out = chat(client, model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], json_mode=True)
        try:
            picked = json.loads(out).get("tools", [])
        except Exception:
            picked = []
        top1 = bool(picked and picked[0] in expected)
        top3 = bool(set(picked) & set(expected))
        route_rows.append(
            {
                "question": question,
                "expected": expected,
                "picked": picked,
                "top1": top1,
                "top3": top3,
                "raw": out[:300],
            }
        )
        print(f"[ROUTE {len(route_rows):02d}] top1={top1} top3={top3} picked={picked[:3]} expected={expected}", flush=True)

    # ---- Report ----
    n = len(qa_rows)
    pos = [r for r in qa_rows if not r["negative"]]
    neg = [r for r in qa_rows if r["negative"]]
    avg_faith = sum(r.get("faithfulness", 0) for r in pos) / max(len(pos), 1)
    avg_rel = sum(r.get("relevancy", 0) for r in pos) / max(len(pos), 1)
    hit3 = sum(1 for r in pos if r["top3_hit"]) / max(len(pos), 1)
    no_hallucination = sum(1 for r in neg if r.get("faithfulness", 0) >= 0.9) / max(len(neg), 1)
    rn = len(route_rows)
    route_top1 = sum(1 for r in route_rows if r["top1"]) / max(rn, 1)
    route_top3 = sum(1 for r in route_rows if r["top3"]) / max(rn, 1)

    report = {
        "model": model,
        "doc_chunks": len(chunks),
        "qa": {
            "cases": len(pos),
            "negative_cases": len(neg),
            "avg_faithfulness": round(avg_faith, 3),
            "avg_answer_relevancy": round(avg_rel, 3),
            "context_hit_at_3": round(hit3, 3),
            "no_hallucination_on_negatives": round(no_hallucination, 3),
            "rows": qa_rows,
        },
        "tool_routing": {
            "cases": rn,
            "top1_accuracy": round(route_top1, 3),
            "top3_accuracy": round(route_top3, 3),
            "rows": route_rows,
        },
    }
    out_dir = Path(os.environ.get("EVAL_OUT_DIR", Path(__file__).resolve().parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Kairós · 离线评测报告",
        "",
        f"- 评测模型：`{model}`",
        f"- 知识库语料：项目文档分块，共 {len(chunks)} 块",
        "",
        "## 一、知识问答链路",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 正向测试用例 | {len(pos)} |",
        f"| 负例（资料无答案） | {len(neg)} |",
        f"| 平均忠实度（faithfulness） | {avg_faith:.3f} |",
        f"| 平均回答相关性（answer relevancy） | {avg_rel:.3f} |",
        f"| 上下文命中率 hit@3 | {hit3:.3f} |",
        f"| 负例不编造率 | {no_hallucination:.3f} |",
        "",
        "## 二、工具路由",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 测试用例 | {rn} |",
        f"| Top-1 准确率 | {route_top1:.3f} |",
        f"| Top-3 准确率 | {route_top3:.3f} |",
        "",
        "## 三、说明",
        "",
        "- 知识问答评测在项目自身文档构成的本地语料上进行，不依赖飞书/外部服务，可复现。",
        "- 评测脚本：`evaluation/run_eval.py`；结果：`evaluation/eval_result.json`。",
        "- 检索器为轻量字符 bigram TF-IDF，用于验证链路而非生产检索方案。",
    ]
    (out_dir / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
