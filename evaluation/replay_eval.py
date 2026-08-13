"""Replay evaluation: score real requests recorded by the observability layer.

Reads successful requests from the request log (question + answer + tool
sequence) and uses DeepSeek as a judge to score answer relevancy and
completeness. This closes the loop between production usage and evaluation:
every real conversation can be turned into a regression signal.

Usage:
    .\\.venv\\Scripts\\python.exe evaluation\\replay_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI  # noqa: E402

from kairos.infrastructure.llm import build_client, model_name  # noqa: E402
from kairos.observability import metrics  # noqa: E402


def load_replay_cases(limit: int = 50) -> list[dict]:
    rows, _ = metrics.list_logs(page=1, page_size=limit, status="ok")
    return [row for row in rows if row.get("question") and row.get("answer")]


def judge(client: OpenAI, model: str, question: str, answer: str) -> dict:
    prompt = (
        "请评估一次真实助手问答。\n"
        f"用户问题：{question}\n"
        f"助手回答：{answer[:2500]}\n\n"
        "输出 JSON：\n"
        '{"relevancy": 0到1的小数（回答是否直接命中问题核心）,\n'
        '"completeness": 0到1的小数（回答是否完整覆盖问题所需信息）}'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严谨的评估助手。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "relevancy": round(float(data.get("relevancy", 0)), 3),
            "completeness": round(float(data.get("completeness", 0)), 3),
        }
    except Exception:
        return {"relevancy": 0.0, "completeness": 0.0}


def main() -> int:
    client = build_client()
    cases = load_replay_cases()
    print(f"replay cases: {len(cases)}", flush=True)
    if not cases:
        print("暂无真实请求可回放（先让助手处理一些对话）")
        return 0

    results = []
    for case in cases:
        score = judge(client, model_name(), case["question"], case["answer"])
        results.append(
            {
                "id": case["id"],
                "created_at": case["created_at"],
                "question": case["question"][:120],
                "tools": case["tool_sequence"],
                "answer": case["answer"][:300],
                **score,
            }
        )
        print(
            f"[{case['id']}] rel={score['relevancy']} comp={score['completeness']} tools={len(case['tool_sequence'])}",
            flush=True,
        )

    n = len(results)
    avg_rel = sum(r["relevancy"] for r in results) / n
    avg_comp = sum(r["completeness"] for r in results) / n
    report = {
        "model": model_name(),
        "cases": n,
        "avg_relevancy": round(avg_rel, 3),
        "avg_completeness": round(avg_comp, 3),
        "rows": results,
    }
    out_dir = Path(__file__).resolve().parent
    (out_dir / "replay_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# 真实请求回放评测",
        "",
        f"- 评测模型：`{model_name()}`",
        f"- 回放样本：{n} 条真实请求",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 平均相关性（relevancy） | {avg_rel:.3f} |",
        f"| 平均完整性（completeness） | {avg_comp:.3f} |",
        "",
        "说明：样本来自请求日志（`request_logs` 表 status=ok 的记录），使用 DeepSeek 作为裁判模型。",
    ]
    (out_dir / "replay_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
