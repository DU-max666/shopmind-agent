"""离线评测：无密钥、无外部服务，一条命令复现全部指标。

两组数据集：
1. eval/routing_cases.jsonl —— 三层路由（规则→LLM→澄清）的意图准确率与澄清率；
2. eval/rag_cases.jsonl —— 混合检索命中@3，以及"检索→生成→校验"全链路的
   幻觉率（答案数字不可溯源）与引用完整率（事实句均带可回查引用）。

运行：
    python eval/run_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_HAS_DIGIT_RE = re.compile(r"\d")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.graphs.state import make_default_context
from backend.app.graphs.supervisor import route_intent, run_turn
from backend.app.llm import MockLLM
from backend.app.safety import citation, grounded_check
from scripts.init_kb import build_retriever

EVAL_DIR = Path(__file__).resolve().parent


def load_jsonl(name: str) -> list[dict[str, Any]]:
    rows = []
    with (EVAL_DIR / name).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eval_routing() -> dict[str, Any]:
    """路由层：意图准确率 + 澄清触发率（clarify 用例应全部触发追问）。"""
    cases = load_jsonl("routing_cases.jsonl")
    llm = MockLLM()
    per_intent: dict[str, list[int]] = {}
    correct = 0
    clarify_total = 0
    misses: list[str] = []
    for c in cases:
        got = route_intent(c["text"], llm).intent
        exp = c["expected"]
        stat = per_intent.setdefault(exp, [0, 0])
        stat[1] += 1
        if got == exp:
            correct += 1
            stat[0] += 1
        else:
            misses.append(f"  [miss] expect={exp} got={got} text={c['text']}")
        if got == "clarify":
            clarify_total += 1
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases) if cases else 0.0,
        "clarify_rate": clarify_total / len(cases) if cases else 0.0,
        "per_intent": per_intent,
        "misses": misses,
    }


def eval_chain() -> dict[str, Any]:
    """RAG 命中@3 + 全链路幻觉率/引用完整率（clarify 轮次单独计数不参与）。"""
    cases = load_jsonl("rag_cases.jsonl")
    retriever = build_retriever()
    ctx = make_default_context(llm=MockLLM(), retriever=retriever)

    hit3 = 0
    chain_n = 0
    clarify_n = 0
    halluc = 0
    cite_ok = 0
    rows: list[str] = []
    for c in cases:
        hits = retriever.search(c["question"], top_k=3)
        docs = {h.doc_id for h in hits}
        if set(c["expected_doc_ids"]) & docs:
            hit3 += 1
        else:
            rows.append(f"  [miss@3] {c['question']} -> {sorted(docs)}")

        st = run_turn(c["question"], ctx=ctx)
        if st.get("route_source") == "clarify":
            clarify_n += 1
            continue
        chain_n += 1
        answer = st.get("final_answer", "")
        evidence = st.get("evidence", [])
        ground = grounded_check.check(answer, evidence)
        if ground.unsupported:
            halluc += 1
            rows.append(f"  [halluc] {c['question']} -> {ground.unsupported}")
        cite = citation.verify(answer, evidence)
        has_fact = bool(_HAS_DIGIT_RE.search(answer))
        if (not has_fact) or (cite.used and not cite.unknown):
            cite_ok += 1
        else:
            rows.append(f"  [cite] {c['question']} -> used={cite.used} unknown={cite.unknown}")
    return {
        "total": len(cases),
        "hit3": hit3,
        "hit_rate_at_3": hit3 / len(cases) if cases else 0.0,
        "chain_n": chain_n,
        "clarify_n": clarify_n,
        "halluc": halluc,
        "hallucination_rate": halluc / chain_n if chain_n else 0.0,
        "cite_ok": cite_ok,
        "citation_completeness": cite_ok / chain_n if chain_n else 0.0,
        "rows": rows,
    }


def main() -> None:
    routing = eval_routing()
    chain = eval_chain()

    print("=" * 64)
    print("智铺 ShopMind 离线评测报告（MockLLM + 内存检索，可复现）")
    print("=" * 64)
    print("\n[1] 意图路由（三层：规则前置 → LLM 兜底 → 低置信澄清）")
    print(f"  样本数        : {routing['total']}")
    print(f"  路由准确率    : {routing['accuracy'] * 100:.1f}%")
    print(f"  澄清触发率    : {routing['clarify_rate'] * 100:.1f}%")
    print("  分意图准确率  :")
    for intent, (ok, total) in sorted(routing["per_intent"].items()):
        print(f"    - {intent:<10} {ok}/{total}")
    for line in routing["misses"]:
        print(line)

    print("\n[2] RAG 检索 + 生成链路")
    print(f"  样本数          : {chain['total']}")
    print(f"  检索命中@3      : {chain['hit_rate_at_3'] * 100:.1f}%")
    print(f"  链路评测轮次    : {chain['chain_n']}（另 {chain['clarify_n']} 轮走澄清追问）")
    print(f"  幻觉率          : {chain['hallucination_rate'] * 100:.1f}%（答案数字不可溯源的轮次占比）")
    print(f"  引用完整率      : {chain['citation_completeness'] * 100:.1f}%（事实句均带可回查【doc_id】）")
    for line in chain["rows"]:
        print(line)
    print("=" * 64)


if __name__ == "__main__":
    main()
