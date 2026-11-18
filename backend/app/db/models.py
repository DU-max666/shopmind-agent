"""SQLAlchemy ORM 模型：商品 / 订单 / 工单 / 退款预演 / 会话 / 审计日志。

设计说明：
- 本模块仅在配置了 MYSQL_DSN 的生产环境加载；离线单测与评测不导入本模块，
  因此顶层 import sqlalchemy 不影响无依赖环境运行；
- 审计表 audit_log 记录每一轮 Agent 决策（路由、工具、校验、转人工），
  是"幻觉可回溯"的最后一环，字段与 state.ShopMindState 对齐。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Product(Base):
    """商品主档：与库存工具 DEMO_INVENTORY 字段对齐，生产由商品中心同步。"""

    __tablename__ = "products"

    sku = Column(String(32), primary_key=True)
    name = Column(String(128), nullable=False)
    category = Column(String(32), index=True)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    shop_type = Column(String(16), default="自营")  # 自营 / 入驻
    doc_id = Column(String(64))  # 关联知识库说明书 doc_id，如 M-ZP-SF-001
    created_at = Column(DateTime, default=datetime.utcnow)


class Order(Base):
    """订单表：items 以 JSON 存储 SKU 明细，避免演示库过度范式化。"""

    __tablename__ = "orders"

    order_id = Column(String(32), primary_key=True)
    user_id = Column(String(32), index=True, nullable=False)
    channel = Column(String(16))
    items = Column(JSON, nullable=False)
    paid_amount = Column(Float, nullable=False)
    status = Column(String(16), index=True)
    order_date = Column(String(10))
    sign_date = Column(String(10))
    coupon_code = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)

    tickets = relationship("Ticket", back_populates="order")


class Ticket(Base):
    """售后/价保工单：状态字段由 after_sale 子图状态机驱动，禁止越权迁移。"""

    __tablename__ = "tickets"

    ticket_id = Column(String(16), primary_key=True)
    order_id = Column(String(32), ForeignKey("orders.order_id"), index=True)
    user_id = Column(String(32), index=True)
    kind = Column(String(16))  # return / exchange / quality / price_guard / other
    summary = Column(Text)
    status = Column(String(24), index=True)  # 与 TicketStatus 枚举一致
    needs_human = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    order = relationship("Order", back_populates="tickets")
    rehearsals = relationship("RefundRehearsal", back_populates="ticket")


class RefundRehearsal(Base):
    """退款沙箱预演快照：committed=True 必须携带人工确认令牌（HITL-xxxx）。"""

    __tablename__ = "refund_rehearsals"

    rehearsal_id = Column(String(16), primary_key=True)
    ticket_id = Column(String(16), ForeignKey("tickets.ticket_id"), index=True)
    order_id = Column(String(32))
    amount = Column(Float, nullable=False)
    status = Column(String(16))  # rehearsed / committed / rolled_back
    committed = Column(Boolean, default=False)
    confirm_token = Column(String(64))  # 人工确认令牌，Agent 无法伪造来源
    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="rehearsals")


class ChatSession(Base):
    """会话主表：一条会话对应前端窗口，turn_count 用于滑窗记忆截断参考。"""

    __tablename__ = "chat_sessions"

    session_id = Column(String(48), primary_key=True)
    user_id = Column(String(32), index=True)
    channel = Column(String(16), default="web")
    turn_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """决策审计：每轮一条，完整记录路由/证据/校验/转人工，支撑幻觉回溯。"""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(48), index=True)
    user_id = Column(String(32))
    user_text = Column(Text)
    intent = Column(String(16))
    route_source = Column(String(8))  # rule / llm / clarify
    intent_confidence = Column(Float)
    evidence_doc_ids = Column(JSON)  # 本轮引用的 doc_id 列表
    grounded_score = Column(Float)
    needs_human = Column(Boolean, default=False)
    human_reason = Column(Text)
    ticket_id = Column(String(16))
    final_answer = Column(Text)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_engine(dsn: str):
    """按 DSN 创建引擎并建表（生产入口，需安装 sqlalchemy + pymysql）。"""
    from sqlalchemy import create_engine

    engine = create_engine(dsn, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine
