"""Streamlit 演示端：可视化"路由 → 子图 → 安全校验 → 答案"全链路。

与后端共用同一条执行链（supervisor.iter_turn_events），离线 MockLLM + 内存检索，
无需任何密钥/外部服务即可演示。

运行：
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (标签, 话术)：覆盖四个子图 + 澄清 + 高风险转人工
SCENARIOS: list[tuple[str, str]] = [
    ("导购推荐", "乳胶枕多少钱？有没有 12cm 枕高的"),
    ("售后退款", "ORD20261105001 沙发有破损，我要退货"),
    ("物流异常", "ORD20261114005 物流一直没更新，帮我查下"),
    ("价保申请", "台灯降价了，申请价保，订单 ORD20261020003"),
    ("模糊提问", "在吗"),
]


def main() -> None:
    import streamlit as st

    from backend.app.graphs.state import make_default_context
    from backend.app.graphs.supervisor import iter_turn_events
    from backend.app.llm import MockLLM
    from scripts.init_kb import build_retriever

    st.set_page_config(page_title="智铺 ShopMind 演示", layout="wide")
    st.title("智铺 ShopMind · 电商多智能体客服与导购")
    st.caption("离线 MockLLM + 内存混合检索：路由 → 子图 → 幻觉校验 → 引用溯源，全链路可复现")

    if "retriever" not in st.session_state:
        st.session_state.retriever = build_retriever()
    if "ctx" not in st.session_state:
        st.session_state.ctx = make_default_context(
            llm=MockLLM(), retriever=st.session_state.retriever)
    if "state" not in st.session_state:
        st.session_state.state = {
            "session_id": "demo-session", "user_id": "U1001", "messages": [],
        }

    with st.sidebar:
        st.header("演示场景")
        for label, text in SCENARIOS:
            if st.button(f"{label}：{text}", key=f"btn_{label}", use_container_width=True):
                st.session_state.pending = text
        st.header("工单账本（实时）")
        store = st.session_state.ctx.tools["ticket"].store
        tickets = list(store.tickets.values())
        if not tickets:
            st.caption("暂无工单：试试“售后退款”或“物流异常”场景")
        for rec in tickets[-6:]:
            st.text(f"{rec['ticket_id']} | {rec['kind']} | {rec['status']}")
        st.header("退款预演快照")
        for snap in list(store.rehearsals.values())[-6:]:
            st.text(f"{snap['rehearsal_id']} | {snap['amount']}元 | {snap['status']}")

    history = st.session_state.state.get("messages", [])
    for msg in history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    user_text = st.chat_input("请输入您的问题…")
    if user_text is None and "pending" in st.session_state:
        user_text = st.session_state.pop("pending")
    if not user_text:
        return

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        trace = st.container()
        final_answer = ""
        for event in iter_turn_events(user_text, st.session_state.state, st.session_state.ctx):
            data = event["data"]
            if event["event"] == "route":
                trace.caption(f"🧭 意图路由：{data['intent']}"
                              f"（来源 {data['source']}，置信 {data['confidence']:.2f}）")
            elif event["event"] == "subgraph":
                tools = data.get("tool_results") or []
                trace.caption(f"🛠 子图 {data['intent']}：调用工具 {len(tools)} 次"
                              + (f"，工单 {data['ticket_id']}（{data['ticket_status']}）"
                                 if data.get("ticket_id") else ""))
            elif event["event"] == "safety":
                trace.caption(f"🛡 证据一致性 {data['grounded_score']:.2f}"
                              + ("，已转人工复核" if data["needs_human"] else "，校验通过"))
            elif event["event"] == "answer":
                final_answer = data["text"]
        st.markdown(final_answer or "（无答案）")
        evidence = st.session_state.state.get("evidence", [])
        if evidence:
            with st.expander("本轮引用证据（doc_id 可回查）"):
                for item in evidence:
                    st.markdown(f"- **{item['doc_id']}**：{item['text']}")


if __name__ == "__main__":
    main()
