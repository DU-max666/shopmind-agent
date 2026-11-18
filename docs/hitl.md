# 人在回路（HITL）与高风险资金防线

客服 Agent 最大的生产风险不是"答错"，而是**做错**：越权退款、擅自承诺、不可回滚。
本项目把"放权"设计成结构问题而非提示词问题——**Agent 在代码层面就办不到的事，
提示词被绕过也不会出事**。

## 1. 退款三级分档（safety/refund_guard.py）

`decide_refund(amount)` 按金额落档，阈值可配置（默认 50 / 200 元）：

| 档位 | 条件 | Agent 权限 | 人工角色 |
| --- | --- | --- | --- |
| `auto` | ≤ 50 元且政策允许 | 直接提交退款 | 事后抽检 |
| `sandbox` | 50 ~ 200 元 | **只能沙箱预演**，等待人工令牌才能提交 | 审批并签发 `HITL-` 令牌 |
| `manual` | > 200 元 | 只能建单升级，`commit_refund` 即使带令牌也返回 `manual_only` | 全程经办 |
| `reject` | 订单不存在 / 超售后时限 / 政策不允许 | 登记工单转人工特批，不得承诺结果 | 决定是否特批 |

前置校验先于金额分档：`order_found / within_window / policy_allowed` 任一为假直接 reject，
杜绝"没核对订单就报退款方案"。

## 2. 沙箱预演：先彩排，再打款

`sandbox` 档必须走完整闭环（tools/ticket.py）：

```
rehearse_refund(order_id, amount)      → 快照 RH00001（status=rehearsed）
   ↓ Agent 告知用户"已预演，待人工确认"
人工签发令牌（HITL- 前缀）
   ↓
commit_refund(rehearsal_id, confirm_token)
   ├─ 无令牌 / 令牌非 HITL- 前缀 → pending_confirm / failed（绝不提交）
   └─ 有效令牌 → committed
rollback_refund(rehearsal_id)          → rolled_back，可重新预演再提交
```

设计要点：
- **令牌来源即防线**：`HITL-` 前缀校验让模型自造的"确认码"天然无效；
- **快照可回滚**：预演记录金额/订单/状态全量留痕，打款错了能一键回退并审计；
- **重复提交拦截**：已 committed 的快照再次 commit 直接报错。

## 3. 工单状态机：非法迁移即异常

售后工单走 `after_sale.TRANSITIONS` 白名单：

```
INTAKE → POLICY_CHECK → SUGGESTION → CREATED ─→ RESOLVED
                          │             ├─→ AWAITING_CONFIRM → RESOLVED / ESCALATED
                          │             └─→ ESCALATED → RESOLVED
                          └─(政策不允许)→ REJECTED
```

`next_status()` 只允许白名单内的边，越权迁移（如 CREATED→REJECTED 跳步、
RESOLVED 复活）抛异常并记审计。状态字符串落进工单账本与答案文本，用户看到的
"已升级/待确认/已解决"与账本严格一致。

## 4. 各子图的升级触发点

| 子图 | 自动转人工场景 | 话术纪律 |
| --- | --- | --- |
| after_sale | 金额 >200、政策不允许的特批、沙箱待确认 | 只报工单号与时限（48h 响应/72h 审核），不承诺结果 |
| logistics | 异常滞留/丢件自动建单 `escalated` | 告知工单号与 48 小时核查时限，引用【POLICY-SLA】 |
| promo | 价保符合条件**也只登记工单** | "人工复核后 72 小时内原路返还"，打款动作永远归人工 |
| 安全收尾 | 规则×LLM 校验不过、证据一致性 <0.8 | 丢弃草稿，统一兜底话术 + 转人工 |

## 5. 可观测与审计

每轮 `iter_turn_events` 产出 `route/subgraph/safety/answer` 事件流，SSE 原样推给前端；
工单账本（`InMemoryTicketStore` / 生产 `db.models.Ticket+RefundRehearsal`）记录
kind/status/金额/令牌，演示端侧栏实时可见。转人工率、预演-提交间隔、回滚次数
都是可直接开出的运营指标。
