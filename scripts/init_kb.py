"""知识库初始化脚本：把虚构的家居电商语料灌入检索链。

语料构成（共 24 篇文档，全部虚构）：
- 商品说明书 x10（doc_id = M-<SKU>，父子块切分，每段自带型号便于引用回查）；
- 售后/价保/配送/安装政策 x4（doc_id = POLICY-*，按"第X条"切分）；
- FAQ x10（doc_id = FAQ-001..010，问答对切分，答案尾部带条目编号）。

用法：
    python -m scripts.init_kb            # 离线构建内存索引并打印统计
    MILVUS_URI=... python -m scripts.init_kb   # 同时灌入 Milvus 集合

评测（eval/run_eval.py）与演示端（frontend/app.py）直接复用 build_retriever()。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings
from backend.app.retrieval.chunking import Chunk, chunk_document
from backend.app.retrieval.embedding import HashEmbedder
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.milvus_store import get_vector_store
from backend.app.retrieval.rerank import LexicalOverlapReranker

# ---------------------------------------------------------------------------
# 商品说明书（价格/规格与 tools/inventory.py 的 DEMO_INVENTORY 严格对齐）
# ---------------------------------------------------------------------------
MANUALS: dict[str, str] = {
    "M-ZP-SF-001": """\
# 产品概述
型号 ZP-SF-001，北欧布艺三人沙发，落叶松实木框架，坐深 58cm，整宽 210cm。

ZP-SF-001 日常售价 1299 元，支持三人位与四人位，四人位加价 300 元。

# 面料与保养
ZP-SF-001 采用 320g 高密科技布沙发套，可整体拆洗，40℃ 以下水洗。

# 尺寸与安装
ZP-SF-001 到货包装 3 件，2 人协作约 30 分钟完成安装，入户门宽度需大于 80cm。
""",
    "M-ZP-SF-002": """\
# 产品概述
型号 ZP-SF-002，原木风实木框架沙发，白蜡木全实木，整宽 220cm，坐高 45cm。

ZP-SF-002 日常售价 2599 元，自营仓发货。

# 涂装与环保
ZP-SF-002 采用水性木器漆，甲醛释放量执行 E0 级标准，建议每 12 个月打蜡保养一次。
""",
    "M-ZP-CP-001": """\
# 产品概述
型号 ZP-CP-001，独立袋装弹簧乳胶床垫，适配 1.8m 床架，总厚 22cm。

ZP-CP-001 日常售价 1899 元，1.8m 规格内置 980 个独立袋装弹簧。

# 舒适层
ZP-CP-001 舒适层为 3cm 天然乳胶加 2cm 高密度海绵，睡感软硬适中偏支撑。

# 保养与翻转
ZP-CP-001 建议每 6 个月按头脚方向翻转一次，乳胶层不可暴晒。
""",
    "M-ZP-CP-002": """\
# 产品概述
型号 ZP-CP-002，椰棕硬支撑床垫，适配 1.5m 床架，总厚 15cm。

ZP-CP-002 日常售价 1099 元，入驻商家发货，棕芯经 180℃ 高温脱糖处理。

# 适用人群
ZP-CP-002 硬度 8 级，适合偏好硬床的成年人及 12 岁以上青少年。
""",
    "M-ZP-ZT-001": """\
# 产品概述
型号 ZP-ZT-001，白橡木电动升降书桌，台面 120cm x 60cm。

ZP-ZT-001 日常售价 899 元，升降范围 72cm 至 120cm，双电机驱动。

# 承重与安全
ZP-ZT-001 动态承重 80kg，静态承重 120kg，遇障回弹灵敏度 15kg。

# 记忆档位
ZP-ZT-001 面板支持 4 组高度记忆，童锁开启后升降需长按 3 秒确认。
""",
    "M-ZP-ZT-002": """\
# 产品概述
型号 ZP-ZT-002，岩板餐桌，台面 140cm x 80cm，岩板厚 12mm。

ZP-ZT-002 日常售价 1699 元，入驻商家发货，整桌承重 200kg。

# 耐热与清洁
ZP-ZT-002 岩板耐 300℃ 高温，可直接放置热锅，污渍 5 分钟内擦拭不留痕。
""",
    "M-ZP-DG-001": """\
# 产品概述
型号 ZP-DG-001，护眼折叠台灯，国AA级照度，中心照度 500lx。

ZP-DG-001 日常售价 159 元，双11活动到手价 129 元。

# 光源参数
ZP-DG-001 色温 3000K 至 5000K 无级调节，显色指数 Ra95，频闪豁免级。

# 电池与续航
ZP-DG-001 内置 4000mAh 电池，40% 亮度续航 18 小时，Type-C 充电约 3 小时充满。
""",
    "M-ZP-DG-002": """\
# 产品概述
型号 ZP-DG-002，落地阅读灯，灯头高度 160cm，底座直径 30cm。

ZP-DG-002 日常售价 349 元，自营仓发货。

# 调光方式
ZP-DG-002 支持触控 5 档调光，另有 30 秒与 2 小时两档定时关灯。
""",
    "M-ZP-SN-001": """\
# 产品概述
型号 ZP-SN-001，多层收纳柜，共 5 层抽屉，单层承重 10kg。

ZP-SN-001 日常售价 499 元，入驻商家发货，宽 60cm 深 40cm 高 105cm。

# 材质
ZP-SN-001 柜体为 PP 聚丙烯材质，滑轨通过 5000 次开合疲劳测试。
""",
    "M-ZP-ZC-001": """\
# 产品概述
型号 ZP-ZC-001，天然乳胶枕，乳胶含量 93%，尺寸 40cm x 60cm。

ZP-ZC-001 日常售价 249 元，提供 10cm 与 12cm 两种枕高。

# 保养
ZP-ZC-001 枕芯不可水洗不可暴晒，建议 3 年更换一次，外套可机洗。
""",
}

# ---------------------------------------------------------------------------
# 政策条款（doc_id 与 after_sale/logistics 子图引用的 POLICY-* 对齐）
# ---------------------------------------------------------------------------
POLICIES: dict[str, str] = {
    "POLICY-RMA": """\
智铺家居退换货政策（2026 版）

第一条 自签收之日起 7 天内，商品完好不影响二次销售的，可无理由退货，退货运费由买家承担。
第二条 自签收之日起 15 天内，因尺寸不合适可申请换货 1 次，同型号换色免运费。
第三条 签收后 30 天内出现非人为质量问题的，免费维修或换新，运费由平台承担。
第四条 定制类与已安装的大件商品，除质量问题外不支持无理由退货。
第五条 退款在审核通过后 72 小时内原路退回。
""",
    "POLICY-PRICE": """\
智铺家居价格保护政策

第一条 下单后 30 天内，同款商品到手价低于下单单价的，可申请补差价 1 次。
第二条 差价按下单单价与当前活动价、券后价中的较低者计算。
第三条 限时秒杀与百亿补贴渠道商品不参与价保。
第四条 价保申请审核通过后，差价 72 小时内按原支付渠道返还。
""",
    "POLICY-SHIP": """\
智铺家居配送服务政策

第一条 自营仓商品 48 小时内发货，入驻商家商品 72 小时内发货。
第二条 全国 300 个城市支持配送，大件商品送装一体。
第三条 单笔订单实付满 499 元免运费，未满收取基础运费 30 元。
第四条 物流异常滞留超过 48 小时的，平台先行补发或退款。
""",
    "POLICY-INSTALL": """\
智铺家居安装服务政策

第一条 沙发、床垫、升降桌等大件商品提供免费上门安装。
第二条 安装师傅上门前电话预约，市区范围内 24 小时内可约。
第三条 三人沙发安装约需 30 分钟，升降桌通电调试约 10 分钟。
第四条 因客户原因二次上门安装的，收取上门费 50 元。
""",
}

# ---------------------------------------------------------------------------
# FAQ（答案尾部带条目编号，保证引用回查时数字可核对）
# ---------------------------------------------------------------------------
FAQS: dict[str, str] = {
    "FAQ-001": "问：如何查询订单物流？\n答：在“我的订单”点击“查看物流”，或把订单号（ORD 开头）发给我自动查询（FAQ-001）。",
    "FAQ-002": "问：优惠券在哪里领取？\n答：首页“领券中心”可领满 999 减 100 等自营券，有效期至 2026-12-31（FAQ-002）。",
    "FAQ-003": "问：支持货到付款吗？\n答：自营仓 200 元以下订单支持货到付款，大件送装商品暂不支持（FAQ-003）。",
    "FAQ-004": "问：发票怎么开具？\n答：确认收货后可在订单详情申请电子发票，1 个工作日内开出（FAQ-004）。",
    "FAQ-005": "问：床垫保修几年？\n答：弹簧结构保修 10 年，乳胶舒适层保修 3 年（FAQ-005）。",
    "FAQ-006": "问：沙发套可以单独购买吗？\n答：可以，三人位沙发套 299 元，下单备注沙发型号即可（FAQ-006）。",
    "FAQ-007": "问：下单后可以修改收货地址吗？\n答：未发货前可修改 1 次，可在订单页自助修改或联系客服（FAQ-007）。",
    "FAQ-008": "问：拒收后退款多久到账？\n答：仓库签收退回商品后 72 小时内原路退回（FAQ-008）。",
    "FAQ-009": "问：企业采购可以开增值税专票吗？\n答：单笔满 5000 元可申请增值税专用发票并支持对公打款（FAQ-009）。",
    "FAQ-010": "问：有以旧换新活动吗？\n答：床垫与沙发以旧换新最高抵扣 200 元，旧件由师傅免费拖走（FAQ-010）。",
}

SAMPLE_DOCUMENTS: list[tuple[str, str, str]] = (
    [(doc_id, "manual", text) for doc_id, text in MANUALS.items()]
    + [(doc_id, "policy", text) for doc_id, text in POLICIES.items()]
    + [(doc_id, "faq", text) for doc_id, text in FAQS.items()]
)


def build_chunks(documents: list[tuple[str, str, str]] | None = None) -> list[Chunk]:
    """按文档类型路由到对应切分器，产出检索块列表。"""
    docs = documents if documents is not None else SAMPLE_DOCUMENTS
    chunks: list[Chunk] = []
    for doc_id, doc_type, content in docs:
        chunks.extend(chunk_document(doc_id, doc_type, content))
    return chunks


def build_retriever(chunks: list[Chunk] | None = None) -> HybridRetriever:
    """离线内存版混合检索器：Hash 嵌入 + BM25 + RRF + 词面精排。"""
    s = get_settings()
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=s.embedding_dim),
        store=get_vector_store(s),
        reranker=LexicalOverlapReranker(),
        rrf_k=s.rrf_k,
    )
    retriever.add_documents(chunks if chunks is not None else build_chunks())
    return retriever


def main() -> None:
    """灌库入口：打印统计；配置了 MILVUS_URI 时同步写入 Milvus 集合。"""
    s = get_settings()
    chunks = build_chunks()
    retriever = build_retriever(chunks)
    leaves = [c for c in chunks if not c.meta.get("is_parent")]
    print(f"[init_kb] 文档 {len(SAMPLE_DOCUMENTS)} 篇，切块 {len(chunks)}（叶子 {len(leaves)}）")
    print(f"[init_kb] 向量后端：{'Milvus ' + s.milvus_collection if s.milvus_uri else 'InMemory'}")
    probe = retriever.search("乳胶枕 价格 枕高", top_k=3)
    for hit in probe:
        print(f"  - {hit.doc_id} score={hit.score:.4f}")
    print("[init_kb] 冒烟检索完成。")


if __name__ == "__main__":
    main()
