"""面向业务动作的两种视图。

- 运营视图：解释“哪条新信息把一个组合从可销售变成待澄清”，给出触发证据链、
  分场景占比与处置动作，而不是脱离动作的分类分数。
- 供应商视图：只给该商家相关 SKU 的当前适用标签、可供货版本（文案版次）与
  需要补证的事项，不暴露其他商家与采购承诺细节。
"""
from __future__ import annotations

from .classifier import classify_with_human_override
from .commitment import CommitmentReview
from .evidence import Packet
from .taxonomy import Taxonomy


def operator_commitment_lines(reviews: list[CommitmentReview]) -> list[str]:
    lines = []
    for review in reviews:
        lines.append(
            f"[{review.disposition_text}] 买家 {review.buyer} × 组合 {review.package}"
            f"（{review.taxonomy} 承诺“{review.scope}”，{review.accepted} 接受，"
            f"状态 {review.status}）"
            + ("　【另需运营通知采购方】" if review.notify_required and review.disposition != "notify_buyer" else "")
        )
        lines.append("  触发信息：")
        for item in review.trigger_evidence or ["无新增触发信息，维持原判断"]:
            lines.append(f"   - {item}")
        lines.append("  场景枚举：")
        for scenario in review.scenarios:
            mark = "满足" if scenario.satisfies else "不满足"
            inferred = "—" if scenario.inferred_share is None else f"{scenario.inferred_share:.3f}"
            confirmed = "—" if scenario.confirmed_share is None else f"{scenario.confirmed_share:.3f}"
            lines.append(
                f"   - 选法 {list(scenario.choice)}：推断鲜果占比 {inferred}"
                f"（人工确认口径 {confirmed}），{mark}；{scenario.note}"
            )
        if review.salvage_choices:
            lines.append(f"  去掉未确认替代品后可履约的选法：{review.salvage_choices}")
        if review.open_optional_group:
            lines.append("  可选替换组尚未与买家锁定，暂停新增预订直到选法确认")
    return lines


def supplier_view(
    packet: Packet,
    merchant: str,
    taxonomies: dict[str, Taxonomy],
) -> list[dict]:
    """供应商只看到自己的 SKU：适用标签、可供货文案版次、缺证提示。"""
    out = []
    v2 = taxonomies["v2"]
    for sku, product in sorted(packet.products.items()):
        if product.merchant != merchant:
            continue
        result = classify_with_human_override(packet, sku, v2)
        latest = product.latest_revision
        missing = product.ingredients is None
        out.append(
            {
                "sku": sku,
                "suppliable_revision": latest,
                "title": product.title_at(latest),
                "label_v2": result.label,
                "confidence": result.confidence,
                "basis": result.basis,
                "abstain_reason": result.abstain_reason,
                "needs_evidence": (
                    "请补充配料事实/完整文案" if missing or result.label is None else ""
                ),
            }
        )
    return out


def supplier_lines(rows: list[dict], merchant: str) -> list[str]:
    lines = [f"供应商 {merchant} 的可供货版本与适用标签："]
    for row in rows:
        label = row["label_v2"] or "待定（不可硬归叶类）"
        lines.append(
            f"- {row['sku']} 文案 r{row['suppliable_revision']}《{row['title']}》"
            f" -> {label}（{row['basis']}/{row['confidence']}）"
            + (f"；{row['needs_evidence']}" if row["needs_evidence"] else "")
        )
    return lines


def classification_lines(rows: list[dict]) -> list[str]:
    lines = []
    for row in rows:
        label = row["label"] or "弃权"
        extra = f"，弃权理由：{row['abstain_reason']}" if row["label"] is None else ""
        lines.append(
            f"- {row['sku']} @ {row['taxonomy']}/r{row['revision']} -> {label}"
            f"（{row['basis']}/{row['confidence']}{extra}）"
        )
        for reason in row["reasons"]:
            lines.append(f"    · {reason}")
    return lines
