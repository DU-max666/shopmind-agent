# Agent 编排与检索设计

## 1. 多智能体拓扑

Supervisor 模式：一个总调度 + 四个业务子图（guide / after_sale / logistics / promo）。
在线环境用 LangGraph `StateGraph` 组网（`graphs/supervisor.py::build_graph`）；
离线环境退化为 `run_turn` 直连调度，**两条路径共享同一状态契约 `ShopMindState`**，
保证行为一致、评测可复现。

```
用户消息 → 记忆注入 → 意图路由 → 子图执行(检索+工具) → 安全收尾 → 记忆写回
```

## 2. 三层意图路由（supervisor.py）

| 层 | 机制 | 目的 |
| --- | --- | --- |
| 规则前置 | `_RULE_PATTERNS` 正则 + 置信度（售后/物流 0.90 > 活动 0.85 > 导购 0.70） | 高频意图毫秒级、完全可解释 |
| LLM 兜底 | 提示词输出 JSON（intent/confidence/reason），解析失败降级 clarify | 覆盖长尾表述 |
| 澄清闸门 | 最优候选置信 < 0.55 → clarify 追问 | "宁可多问一句，不可答错一句" |

规则命中置信 ≥ `route_answer_threshold`(0.75) 直接返回，省一次 LLM 调用；
导购规则置信刻意压到 0.70——售前表述太多样，强制走 LLM 复核。

## 3. 四个子图的职责边界

- **guide**：确定性约束抽取（品类/预算/风格/尺寸，正则零幻觉）→ 混合检索 → 库存实时价
  并入证据区 → 生成。无品类时返回澄清问题而非硬推荐；
- **after_sale**：显式状态机 `INTAKE→POLICY_CHECK→SUGGESTION→CREATED→
  {AWAITING_CONFIRM|ESCALATED|RESOLVED}`（非法迁移抛异常），
  诉求识别 → 政策检索 → 订单时限核对 → `refund_guard` 分档 → 工单落账；
- **logistics**：缺订单号先追问槽位；轨迹来自物流工具；异常件自动建工单升级人工；
- **promo**：价保差价**必须由计算工具产出**（下单单价 vs 现价/券后价取低、30 天窗口），
  提示词明令禁止口算；符合条件也只"登记工单"，打款永远归人工。

## 4. 工具层契约（tools/base.py）

每个工具 = JSON Schema 入参校验 + Provider 抽象（Mock/真实实现同契约）+ 统一结果
`ToolResult{ok, tool, data, error, latency_ms, meta}`。异常/校验失败/超时全部归一化为
`ok=False + error`，子图据此决定"如实告知"还是"转人工"，杜绝工具失败后模型瞎编。

## 5. 混合检索链（retrieval/）

1. **切分（chunking.py）**：按文档类型路由三种策略——说明书父子块（章节父块 + 段落子块，
   命中子块回填父块上下文）、政策按"第X条"、FAQ 按问答对；
2. **双路召回**：向量（生产 BGE，离线 `HashEmbedder` 字符 n-gram 哈希投影，确定性）+
   BM25（纯 Python 实现，中文二元分词 + ASCII 小写）；
3. **RRF 融合**：`score = Σ 1/(k + rank)`，对分数分布不敏感、天然稳健；
4. **精排**：生产 cross-encoder（懒加载），离线 `LexicalOverlapReranker` 词面重合度；
5. **证据区**：`evidence_block` 每行 `[doc_id] 关键句`，关键句优先取含数字的行——
   与幻觉校验直接咬合。

`scripts/init_kb.py` 的 24 篇虚构文档（10 说明书 + 4 政策 + 10 FAQ）同时服务评测与演示，
语料写法遵循"可回查约定"：每段自带型号、FAQ 尾注编号（详见 hallucination.md）。

## 6. 记忆与画像（memory/）

`SessionMemory` 滑窗（默认 8 轮）写回缓存；`UserProfileStore` 沉淀品类/风格偏好计数、
最近预算与历史工单，可压缩成一行注入导购提示词。全部落在 `cache.py` 抽象上：
离线 `InMemoryCache`（TTL），生产 `RedisCache`。

## 7. API 与事件流（main.py）

`POST /api/chat/stream` 以 SSE 推送 `route → subgraph → safety → answer` 四类事件，
前端可实时渲染决策轨迹；另有同步 `/api/chat` 与 `/api/ticket/{id}`。
鉴权（`X-API-Key`/Bearer）+ 令牌桶限流（429 带 Retry-After）。
fastapi/pydantic 懒加载：未安装时模块可导入、单测不受影响。

## 8. 评测设计（eval/）

- `routing_cases.jsonl` 40 条：五意图分桶，含"投诉快递把书桌摔坏了"这类多意图竞争样本、
  "在吗"类澄清样本 → 路由准确率 + 澄清率；
- `rag_cases.jsonl` 30 条：跨三种文档类型 → 命中@3；再经 `run_turn` 全链路 →
  幻觉率（数字不可溯源轮次）+ 引用完整率（事实句带可回查引用）；
- clarify 轮次单独计数不参与生成指标（追问文案本就不该有事实）。
