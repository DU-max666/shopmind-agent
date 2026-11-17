"""导购子图：需求澄清 → 约束抽取 → 商品检索 → 对比推荐话术（引用商品参数）。

链路要点：
- 约束抽取用确定性正则（预算/品类/风格/材质/尺寸），不依赖 LLM 也能跑；
- 推荐依据 = 知识库说明书（父子块回填后的完整参数）+ 库存工具实时价/库存；
- 所有价格与库存数字必须来自工具结果并进入证据集，供收尾校验回查。
"""
from __future__ import annotations

import re
from typing import Any

from backend.app.graphs.state import AgentContext, ShopMindState
from backend.app.llm import ChatMessage
from backend.prompts.loader import load_prompt

_CATEGORY_WORDS: dict[str, list[str]] = {
    "沙发": ["沙发"],
    "床垫": ["床垫", "弹簧垫", "椰棕垫"],
    "书桌": ["书桌", "办公桌", "升降桌"],
    "餐桌": ["餐桌", "岩板桌"],
    "台灯": ["台灯", "护眼灯"],
    "落地灯": ["落地灯", "阅读灯"],
    "收纳": ["收纳柜", "置物架", "储物"],
    "枕头": ["枕头", "乳胶枕"],
}
_STYLE_WORDS = ["北欧", "原木", "简约", "轻奢", "日式", "奶油风"]
_MATERIAL_WORDS = ["实木", "布艺", "乳胶", "弹簧", "椰棕", "岩板", "白橡木", "金属", "羽绒"]
_BUDGET_RE = re.compile(r"(?:预算|控制在|不超过|以内|以下|左右)[^\d]*(\d{2,6})|(\d{2,6})\s*(?:元|块)")
_SIZE_RE = re.compile(r"(\d{2,3}(?:\.\d)?)\s*(cm|厘米|米|m)")

CLARIFY_QUESTION = "请问您想选购哪类家居商品？例如沙发、床垫、书桌、台灯等；也可以告诉我预算范围或偏好的风格。"


def extract_constraints(text: str) -> dict[str, Any]:
    """从用户话语中抽取结构化选购约束。"""
    categories = [c for c, words in _CATEGORY_WORDS.items() if any(w in text for w in words)]
    budget = None
    m = _BUDGET_RE.search(text)
    if m:
        budget = float(m.group(1) or m.group(2))
    style = next((s for s in _STYLE_WORDS if s in text), None)
    materials = [w for w in _MATERIAL_WORDS if w in text]
    size = None
    ms = _SIZE_RE.search(text)
    if ms:
        size = float(ms.group(1))
    return {"category": categories[0] if categories else None,
            "categories": categories, "budget": budget, "style": style,
            "materials": materials, "size_cm": size}


def _skus_from_hits(hits: list[Any]) -> list[str]:
    """说明书文档 id 约定 M-<SKU>，解析出可回查库存的 SKU 列表。"""
    skus = []
    for hit in hits:
        if hit.doc_type == "manual" and hit.doc_id.startswith("M-"):
            skus.append(hit.doc_id[2:])
    return skus


def handle(state: ShopMindState, ctx: AgentContext) -> dict[str, Any]:
    text = state.get("user_text", "")
    constraints = extract_constraints(text)
    if not constraints["categories"]:
        return {"slots": constraints, "draft_answer": CLARIFY_QUESTION, "evidence": []}

    hits = ctx.retriever.search(text, top_k=ctx.settings.retrieval_top_k) \
        if ctx.retriever is not None else []
    evidence: list[dict[str, Any]] = [
        {"doc_id": h.doc_id, "text": h.meta.get("snippet", h.text)} for h in hits
    ]
    tool_results: list[dict[str, Any]] = []

    skus = _skus_from_hits(hits)
    if skus and "inventory" in ctx.tools:
        inv = ctx.tools["inventory"].run(skus=skus)
        tool_results.append(inv.to_dict())
        if inv.ok:
            for item in inv.data["items"]:
                line = (f"{item['name']}（型号 {item['sku']}）实时价 {item['price']} 元，"
                        f"库存 {item['stock']} 件，{item['shop']}发货")
                evidence.append({"doc_id": f"INV-{item['sku']}", "text": line})

    if not hits:
        # 知识库未命中：宁可澄清也不编造
        return {"slots": constraints, "evidence": evidence, "tool_results": tool_results,
                "draft_answer": "抱歉，暂时没有查到与该商品匹配的可靠资料，已记录您的需求，"
                                 "您可以换个商品名称再问，或转人工客服帮您核实。"}

    evidence_block = "\n".join(f"[{e['doc_id']}] {e['text']}" for e in evidence)
    prompt = (
        f"{load_prompt('guide')}\n\n"
        f"用户诉求：{text}\n选购约束：{constraints}\n\n"
        f"EVIDENCE_BEGIN\n{evidence_block}\nEVIDENCE_END\n"
        "请基于证据给出对比推荐话术，价格与库存只能引用证据中的数字，并在句尾标注来源编号。"
    )
    answer = ctx.llm.complete([ChatMessage("system", "你是智铺家居导购助手"), ChatMessage("user", prompt)])
    return {"slots": constraints, "evidence": evidence, "tool_results": tool_results,
            "draft_answer": answer}
