"""把组件级分类不确定性传播到组合级采购承诺。

业务规则（见 资料格式.md 与需求）：

- 平台授予的是**组合级**承诺，复查以承诺（买家×组合）为单位；同一共享组件在
  多个组合里出现不算同一客单，按承诺分别建单、按事实共享复核。
- v1“含鲜果”只承诺确有鲜果，不追用新版占比；v2 fresh 要求适用的新类与
  ``min_fresh`` 重量占比。
- 含替换件的混装按可选组枚举场景；某组件标签变动后，仍满足原承诺的场景可
  按旧约继续，只有去掉尚未确认的替代品才满足的场景要显式标出。
- 处置三档：
  * ``pause_new``   应暂停新增预订（存在无法排除的阻断性未知/互斥冲突）；
  * ``continue_old`` 可按旧约继续交付（承诺仍被满足，变化不削弱原承诺）；
  * ``notify_buyer`` 需运营明确通知采购方（承诺不再满足或需买家确认替换）。
  同一承诺取最严档，分场景证据全部保留。
- 重算只生成复查结论，不改写承诺原文（scope/taxonomy/actual_choices 原样保留）。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iter_product
from typing import Optional

from .classifier import classify_with_human_override
from .evidence import Packet
from .taxonomy import Taxonomy

PAUSE_NEW = "pause_new"
CONTINUE_OLD = "continue_old"
NOTIFY_BUYER = "notify_buyer"

DISPOSITION_RANK = {CONTINUE_OLD: 0, NOTIFY_BUYER: 1, PAUSE_NEW: 2}
DISPOSITION_TEXT = {
    PAUSE_NEW: "暂停新增预订",
    CONTINUE_OLD: "按旧约继续交付",
    NOTIFY_BUYER: "运营明确通知采购方",
}


@dataclass
class ComponentState:
    sku: str
    label: Optional[str]  # 可用于承诺判定的 v2 叶类（人工或经护栏的证据推断）或 None
    confidence: str
    basis: str  # human | model | rule | model_fallback
    reasons: tuple[str, ...]
    abstain_reason: Optional[str]
    human_label: Optional[str]  # 同版次人工叶类；None 表示无同版人工确认
    inferred_label: Optional[str]  # 证据/模型推断叶类，供占比参考但单独标注
    changed_since_acceptance: bool
    prior_label: Optional[str]
    change_detail: Optional[str]

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "label": self.label,
            "confidence": self.confidence,
            "basis": self.basis,
            "reasons": list(self.reasons),
            "abstain_reason": self.abstain_reason,
            "human_label": self.human_label,
            "inferred_label": self.inferred_label,
            "changed_since_acceptance": self.changed_since_acceptance,
            "prior_label": self.prior_label,
            "change_detail": self.change_detail,
        }


@dataclass
class Scenario:
    choice: tuple[str, ...]  # 每个可选组选中的 sku
    fixed: tuple[str, ...]  # base 件
    confirmed_fresh_weight: int  # 仅人工 v2 确认 fresh 的重量
    inferred_fresh_weight: int  # 证据推断 fresh（含人工确认）的重量
    total_weight: int
    confirmed_share: Optional[float]
    inferred_share: Optional[float]
    fresh_members: list[str]  # 推断口径下的 fresh 件
    confirmed_fresh_members: list[str]  # 人工确认口径
    blocking_unknown: list[str]  # 无法排除的未知/冲突件（无任何叶类依据）
    v1_has_fresh: Optional[bool]
    satisfies: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "choice": list(self.choice),
            "fixed": list(self.fixed),
            "confirmed_fresh_weight": self.confirmed_fresh_weight,
            "inferred_fresh_weight": self.inferred_fresh_weight,
            "total_weight": self.total_weight,
            "confirmed_share": None if self.confirmed_share is None else round(self.confirmed_share, 4),
            "inferred_share": None if self.inferred_share is None else round(self.inferred_share, 4),
            "fresh_members": self.fresh_members,
            "confirmed_fresh_members": self.confirmed_fresh_members,
            "blocking_unknown": self.blocking_unknown,
            "v1_has_fresh": self.v1_has_fresh,
            "satisfies": self.satisfies,
            "note": self.note,
        }


@dataclass
class CommitmentReview:
    buyer: str
    package: str
    taxonomy: str
    scope: str
    accepted: str
    original_choices: list[str]
    status: str
    disposition: str
    disposition_text: str
    trigger_evidence: list[str]
    scenarios: list[Scenario]
    components: dict[str, ComponentState]
    salvage_choices: list[list[str]]  # 去掉未确认替代品后才满足的选法
    open_optional_group: bool
    notify_required: bool  # 即使主动作是暂停，仍需运营通知采购方的事项

    def to_dict(self) -> dict:
        return {
            "buyer": self.buyer,
            "package": self.package,
            "taxonomy": self.taxonomy,
            "scope": self.scope,
            "accepted": self.accepted,
            "original_choices": list(self.original_choices),
            "status": self.status,
            "disposition": self.disposition,
            "disposition_text": self.disposition_text,
            "notify_required": self.notify_required,
            "trigger_evidence": self.trigger_evidence,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "salvage_choices": self.salvage_choices,
            "open_optional_group": self.open_optional_group,
        }


def component_states(
    packet: Packet,
    v2: Taxonomy,
    as_of: Optional[str] = None,
) -> dict[str, ComponentState]:
    """全部在册组件在 v2 下的当前状态，区分人工确认与证据推断两层。"""
    accept_dates = sorted(
        {row["accepted"] for row in packet.data["commitments"]}
    )
    states = {}
    for sku, product in packet.products.items():
        current = classify_with_human_override(
            packet, sku, v2, on_or_before=as_of
        )
        ruling_v2 = product.ruling_on(v2.name, on_or_before=as_of)
        human_label = ruling_v2["label"] if ruling_v2 is not None else None
        inferred_label = current.label  # 可能是人工、模型或规则结果
        prior_label, change_detail, changed = None, None, False
        for accepted in accept_dates:
            baseline = classify_with_human_override(
                packet, sku, v2, on_or_before=accepted
            )
            after = product.ruling_on(v2.name)
            if (
                after is not None
                and after["known"] > accepted
                and baseline.label != current.label
            ):
                changed = True
                prior_label = baseline.label
                change_detail = (
                    f"{accepted} 时点标签 {baseline.label!r} -> "
                    f"{after['known']} 裁决标签 {current.label!r}"
                )
        states[sku] = ComponentState(
            sku=sku,
            label=current.label,
            confidence=current.confidence,
            basis=current.basis,
            reasons=current.reasons,
            abstain_reason=current.abstain_reason,
            human_label=human_label,
            inferred_label=inferred_label,
            changed_since_acceptance=changed,
            prior_label=prior_label,
            change_detail=change_detail,
        )
    return states


def _enumerate_scenarios(package_row: dict) -> list[tuple[str, ...]]:
    optional = package_row["optional"] or [[]]
    return [tuple(combo) for combo in iter_product(*optional)]


def review_commitment(
    packet: Packet,
    commitment: dict,
    v2: Taxonomy,
    states: dict[str, ComponentState],
    as_of: Optional[str] = None,
) -> CommitmentReview:
    package_row = packet.package(commitment["package"])
    base = list(package_row["base"])
    optional_groups = package_row["optional"]
    min_fresh = package_row["min_fresh"]
    open_group = any(
        not commitment["actual_choices"] for _ in optional_groups
    ) and bool(optional_groups)

    all_member_skus = base + sum(optional_groups, [])
    components = {sku: states[sku] for sku in all_member_skus}

    scenarios: list[Scenario] = []
    for choice in _enumerate_scenarios(package_row):
        members = base + list(choice)
        weights = {sku: packet.products[sku].net_g for sku in members}
        total = sum(weights.values())

        confirmed_fresh = [
            sku for sku in members if states[sku].human_label == "fresh"
        ]
        inferred_fresh = [
            sku for sku in members if states[sku].inferred_label == "fresh"
        ]
        # 阻断件：连证据推断叶类都没有（人工未知、纯缺证、互斥冲突弃权）
        blocking = [sku for sku in members if states[sku].inferred_label is None]
        confirmed_weight = sum(weights[sku] for sku in confirmed_fresh)
        inferred_weight = sum(weights[sku] for sku in inferred_fresh)
        confirmed_share = confirmed_weight / total if total else None
        inferred_share = inferred_weight / total if total else None

        if commitment["taxonomy"] == "v1":
            # v1 只承诺“确有鲜果”，不追用新版占比：任一 fresh 证据即满足，
            # 人工确认与证据推断都算事实依据，未知件不阻断该弱承诺。
            has_fresh = bool(inferred_fresh)
            scenarios.append(
                Scenario(
                    choice, tuple(base), confirmed_weight, inferred_weight, total,
                    confirmed_share, inferred_share, inferred_fresh, confirmed_fresh,
                    blocking, has_fresh, has_fresh,
                    "v1 仅承诺含鲜果，不追用 v2 占比",
                )
            )
        else:
            # v2 fresh：以证据推断口径判断占比能否达标；阻断件使占比无法证实。
            ratio_ok = (
                inferred_share is not None and inferred_share + 1e-12 >= min_fresh
            )
            confirmed_ok = (
                confirmed_share is not None and confirmed_share + 1e-12 >= min_fresh
            )
            satisfies = ratio_ok and not blocking
            if blocking:
                note = (
                    f"未确认件 {blocking} 使占比无法最终证实；"
                    f"推断占比 {inferred_share:.3f}（人工确认口径 {confirmed_share:.3f}）"
                )
            elif not ratio_ok:
                note = (
                    f"推断鲜果占比 {inferred_share:.3f} 低于 min_fresh {min_fresh}"
                )
            else:
                note = (
                    f"推断鲜果占比 {inferred_share:.3f} 达标（门槛 {min_fresh}）；"
                    f"人工确认口径 {confirmed_share:.3f}"
                    + ("，已完全证实" if confirmed_ok else "，部分依赖证据推断待人工确认")
                )
            scenarios.append(
                Scenario(
                    choice, tuple(base), confirmed_weight, inferred_weight, total,
                    confirmed_share, inferred_share, inferred_fresh, confirmed_fresh,
                    blocking, None, satisfies, note,
                )
            )

    # 三档处置
    trigger: list[str] = []
    for state in components.values():
        if state.changed_since_acceptance:
            trigger.append(f"{state.sku}：{state.change_detail}")
    salvage: list[list[str]] = []
    notify_required = False
    notify_actions: list[str] = []
    if commitment["taxonomy"] == "v2":
        good = {scenario.choice for scenario in scenarios if scenario.satisfies}
        bad = {scenario.choice for scenario in scenarios if not scenario.satisfies}
        if good and bad:
            # 只有去掉某个尚未确认的替代品（改选另一件）才满足 -> 救场选法
            salvage = [list(choice) for choice in sorted(good)]
        chosen = (
            tuple(commitment["actual_choices"])
            if commitment["actual_choices"]
            else None
        )
        chosen_scenario = (
            next((item for item in scenarios if item.choice == chosen), None)
            if chosen is not None
            else None
        )
        if not good:
            disposition = PAUSE_NEW
            trigger.append("所有可选场景均无法满足 v2 fresh 承诺")
        elif chosen_scenario is not None and not chosen_scenario.satisfies:
            # 已与买家锁定的选法被打坏：必须明确通知采购方
            disposition = NOTIFY_BUYER
            notify_required = True
            notify_actions.append(
                f"已确认选法 {list(chosen)} 不再满足承诺：{chosen_scenario.note}"
            )
            if salvage:
                notify_actions.append(f"去掉未确认替代品后的可行选法：{salvage}")
        elif open_group:
            # 替换组尚未锁定且存在不达标选法：暂停新增，待买家确认后再恢复
            disposition = PAUSE_NEW
            trigger.append("可选替换组尚未与买家确定，存在不达标选法，暂停新增预订")
        elif any(
            states[sku].changed_since_acceptance
            for sku in (base + list(commitment["actual_choices"]))
        ):
            disposition = NOTIFY_BUYER
        else:
            disposition = CONTINUE_OLD
        # 已确认选法被打坏且全局无可行场景：暂停新增之外仍必须通知采购方
        if chosen_scenario is not None and not chosen_scenario.satisfies:
            notify_required = True
            if disposition != NOTIFY_BUYER:
                notify_actions.append(
                    f"已确认选法 {list(chosen)} 同时失效：{chosen_scenario.note}"
                )
                if salvage:
                    notify_actions.append(f"去掉未确认替代品后的可行选法：{salvage}")
        trigger.extend(notify_actions)
    else:
        # v1 承诺：只有标签变化削弱“含鲜果”事实才升级；否则按旧约继续
        any_satisfy = any(item.satisfies for item in scenarios)
        if not any_satisfy:
            disposition = PAUSE_NEW
            trigger.append("没有任何场景能证实 v1“含鲜果”承诺")
        elif any(
            states[sku].changed_since_acceptance
            and states[sku].prior_label == "fresh"
            and states[sku].label != "fresh"
            for sku in components
        ):
            disposition = NOTIFY_BUYER
            notify_required = True
        else:
            disposition = CONTINUE_OLD
            changed = [
                sku for sku, state in components.items() if state.changed_since_acceptance
            ]
            if changed:
                trigger.append(
                    f"变化件 {changed} 不影响 v1“含鲜果”承诺，按旧约继续"
                )

    return CommitmentReview(
        buyer=commitment["buyer"],
        package=commitment["package"],
        taxonomy=commitment["taxonomy"],
        scope=commitment["scope"],
        accepted=commitment["accepted"],
        original_choices=list(commitment["actual_choices"]),
        status=commitment["status"],
        disposition=disposition,
        disposition_text=DISPOSITION_TEXT[disposition],
        trigger_evidence=trigger,
        scenarios=scenarios,
        components=components,
        salvage_choices=salvage,
        open_optional_group=open_group,
        notify_required=notify_required,
    )


def review_all(
    packet: Packet,
    taxonomies: dict[str, Taxonomy],
    as_of: Optional[str] = None,
) -> list[CommitmentReview]:
    v2 = taxonomies["v2"]
    states = component_states(packet, v2, as_of=as_of)
    return [
        review_commitment(packet, row, v2, states, as_of=as_of)
        for row in packet.data["commitments"]
    ]
