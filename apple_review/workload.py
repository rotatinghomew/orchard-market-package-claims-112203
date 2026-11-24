"""人工抽查的范围锁定、排序、容量安排与回退纪律。

规则（见 资料格式.md 与需求）：

- 先锁定核查范围：受影响承诺的阻断性未知件、互斥冲突件、标签发生变化的组件，
  加上 review_work 中已在册的件；范围之外不派工。
- 优先级看“受影响买家覆盖”（同一共享件一次澄清多份风险承诺排前）与误分类
  代价（policy.error_cost）；高代价小品类不得因销量/覆盖靠后而无限延后，
  ``reserve_high`` 工时为其保底。
- 已经派出去的范围（assigned 非空）不能并发再派给另一组；新裁决到达时，
  尚未派出的同证据工作回退（撤销）而不是重复派发，已派出的只登记待对账。
- 去重以“事实签名”为准：SKU + 配料 + 当前文案版次 + 已纳入裁决号。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .commitment import CommitmentReview, PAUSE_NEW, NOTIFY_BUYER
from .evidence import Packet
from .taxonomy import Taxonomy


@dataclass
class ReviewItem:
    sku: str
    evidence_signature: str
    hours: float
    cost_class: str
    error_cost: int
    affected_buyers: list[str]
    affected_commitments: list[str]
    blocking: bool
    conflict: bool
    label_changed: bool
    status: str  # assigned | queued | rolled_back | reconcile
    reason: str
    assigned: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "evidence_signature": self.evidence_signature,
            "hours": self.hours,
            "cost_class": self.cost_class,
            "error_cost": self.error_cost,
            "affected_buyers": self.affected_buyers,
            "affected_commitments": self.affected_commitments,
            "blocking": self.blocking,
            "conflict": self.conflict,
            "label_changed": self.label_changed,
            "status": self.status,
            "reason": self.reason,
            "assigned": self.assigned,
        }


@dataclass
class WorkPlan:
    locked_scope: list[str]
    items: list[ReviewItem]
    capacity_hours: float
    reserve_high: float
    scheduled: list[str]
    deferred: list[str]
    rollbacks: list[dict]
    reconciliations: list[dict]
    guardrail_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "locked_scope": self.locked_scope,
            "items": [item.to_dict() for item in self.items],
            "capacity_hours": self.capacity_hours,
            "reserve_high": self.reserve_high,
            "scheduled": self.scheduled,
            "deferred": self.deferred,
            "rollbacks": self.rollbacks,
            "reconciliations": self.reconciliations,
            "guardrail_notes": self.guardrail_notes,
        }


def evidence_signature(packet: Packet, sku: str) -> str:
    product = packet.products[sku]
    latest = product.latest_revision
    ruling_ids = ",".join(
        f"{row['taxonomy']}@{row['known']}={row['label']}"
        for row in sorted(product.rulings, key=lambda row: row["known"])
    )
    raw = "|".join(
        [
            sku,
            str(product.ingredients),
            product.texts[latest]["title"],
            product.texts[latest]["body"],
            ruling_ids,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _gather_candidates(
    packet: Packet,
    reviews: list[CommitmentReview],
    taxonomies: dict[str, Taxonomy],
) -> dict[str, dict]:
    candidates: dict[str, dict] = {}

    def ensure(sku: str) -> dict:
        return candidates.setdefault(
            sku,
            {
                "buyers": set(),
                "commitments": set(),
                "blocking": False,
                "conflict": bool(
                    packet.products[sku].evidence_conflicts()
                ),
                "label_changed": False,
            },
        )

    for review in reviews:
        ref = f"{review.buyer}/{review.package}"
        for sku, state in review.components.items():
            entry = ensure(sku)
            entry["buyers"].add(review.buyer)
            entry["commitments"].add(ref)
            # 阻断以“没有任何可推断叶类”为准；仅持有旧父类标签但证据可推断
            # 叶类的件不算阻断，进常规复核队列。
            if (
                state.inferred_label is None
                and review.disposition in (PAUSE_NEW, NOTIFY_BUYER)
            ):
                entry["blocking"] = True
            if state.changed_since_acceptance:
                entry["label_changed"] = True
        for scenario in review.scenarios:
            for sku in scenario.blocking_unknown:
                ensure(sku)["blocking"] = True
    # 在册复核工作本身也属于锁定范围
    for row in packet.data["review_work"]:
        ensure(row["sku"])
    return candidates


def build_work_plan(
    packet: Packet,
    reviews: list[CommitmentReview],
    taxonomies: dict[str, Taxonomy],
) -> WorkPlan:
    policy = packet.data["policy"]
    capacity = float(policy["hours"])
    reserve = float(policy["reserve_high"])
    error_cost = policy["error_cost"]
    work_rows = {row["sku"]: row for row in packet.data["review_work"]}

    candidates = _gather_candidates(packet, reviews, taxonomies)
    items: list[ReviewItem] = []
    rollbacks: list[dict] = []
    reconciliations: list[dict] = []

    for sku, info in sorted(candidates.items()):
        product = packet.products[sku]
        row = work_rows.get(sku)
        hours = float(row["hours"]) if row else 1.0
        cost_class = row["cost_class"] if row else "standard"
        # 误分类代价：
        # - 仍有叶类争议的件取候选叶类最大代价；
        # - 已被新裁决完整覆盖（回退）的件取裁决叶类代价，仅用于记录；
        # - 纯未知件按 unknown 代价。
        labels = {
            state.label
            for review in reviews
            for state in review.components.values()
            if state.sku == sku and state.label is not None
        }
        rolled_back = (
            row is not None
            and not row["assigned"]
            and latest_ruling is not None
        )
        if rolled_back and latest_ruling["label"] is not None:
            labels = {latest_ruling["label"]}
        cost = max(
            (error_cost.get(label, error_cost["unknown"]) for label in labels),
            default=error_cost["unknown"],
        )
        signature = evidence_signature(packet, sku)
        latest_ruling = product.ruling_on("v2")

        status, reason, assigned = "queued", "新锁定范围", None
        if row is not None and row["assigned"]:
            assigned = row["assigned"]
            if latest_ruling is not None and latest_ruling["known"] > row["assigned"]:
                # 已派出但同证据已被新裁决改写：不能并发重派，登记对账
                status = "reconcile"
                reason = (
                    f"已于 {assigned} 派出，{latest_ruling['known']} 新裁决到达，"
                    "待对账，不重复派发"
                )
                reconciliations.append(
                    {"sku": sku, "assigned": assigned, "new_ruling": latest_ruling}
                )
            else:
                status = "assigned"
                reason = f"已于 {assigned} 派出，锁定占用工时"
        elif latest_ruling is not None and row is not None:
            # 未派出但新裁决已覆盖原工作意图：回退，不重复派
            status = "rolled_back"
            reason = (
                f"未派出工作被 {latest_ruling['known']} 裁决覆盖，"
                "同证据回退而非重复派发"
            )
            rollbacks.append({"sku": sku, "superseded_by": latest_ruling})
        elif info["blocking"]:
            reason = "阻断性未知/冲突件，暂停预订的直接原因"
        elif info["label_changed"]:
            reason = "组件标签相对承诺接受时点发生变化"
        elif info["conflict"]:
            reason = "配料事实与文案互斥冲突"

        items.append(
            ReviewItem(
                sku=sku,
                evidence_signature=signature,
                hours=hours,
                cost_class=cost_class,
                error_cost=cost,
                affected_buyers=sorted(info["buyers"]),
                affected_commitments=sorted(info["commitments"]),
                blocking=info["blocking"],
                conflict=info["conflict"],
                label_changed=info["label_changed"],
                status=status,
                reason=reason,
                assigned=assigned,
            )
        )

    # 排序：误分类代价优先（高代价小品类不得因销量/覆盖靠后而延后），
    # 其次阻断属性，再看买家覆盖（共享事实一次澄清多份承诺）。
    def sort_key(item: ReviewItem) -> tuple:
        return (
            -item.error_cost,
            0 if item.blocking else 1,
            -len(item.affected_buyers),
            item.sku,
        )

    queued = sorted(
        (item for item in items if item.status == "queued"), key=sort_key
    )
    used = sum(item.hours for item in items if item.status in ("assigned", "reconcile"))
    scheduled: list[str] = []
    deferred: list[str] = []
    remaining = capacity - used

    # 第一步：高代价（cost_class=high）排队件优先占用 reserve_high 保底工时，
    # 同档内仍按误分代价/覆盖排序，确保高代价小品类不会被普通件挤出本轮。
    high_queued = [item for item in queued if item.cost_class == "high"]
    standard_queued = [item for item in queued if item.cost_class != "high"]
    high_used = 0.0
    for item in high_queued:
        if high_used < reserve and item.hours <= remaining + 1e-9:
            scheduled.append(item.sku)
            high_used += item.hours
            remaining -= item.hours
        else:
            deferred.append(item.sku)
    # 第二步：保底之外的高代价件与标准件统一按优先级填满剩余容量
    rest = sorted(
        [item for item in high_queued if item.sku in deferred] + standard_queued,
        key=sort_key,
    )
    for item in rest:
        if item.hours <= remaining + 1e-9:
            scheduled.append(item.sku)
            remaining -= item.hours
            if item.sku in deferred:
                deferred.remove(item.sku)
        elif item.sku not in deferred:
            deferred.append(item.sku)

    notes = []
    high_deferred = [
        item.sku for item in queued
        if item.cost_class == "high" and item.sku in deferred
    ]
    if high_deferred:
        notes.append(
            "高代价小品类未排入本轮（容量不足），下轮必须优先，"
            f"不得因销量延后：{high_deferred}"
        )
    if any(item.blocking for item in items):
        notes.append("高误分代价件优先于阻断件排序；同代价下共享事实（覆盖多买家）优先")

    return WorkPlan(
        locked_scope=[item.sku for item in items],
        items=items,
        capacity_hours=capacity,
        reserve_high=reserve,
        scheduled=scheduled,
        deferred=deferred,
        rollbacks=rollbacks,
        reconciliations=reconciliations,
        guardrail_notes=notes,
    )
