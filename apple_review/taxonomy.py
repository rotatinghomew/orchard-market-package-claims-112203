"""类目版次与标签版本语义。

关键规则：

- 标签表达的是“该类目版次下可归的类”。v1 的父类标签（fruit/processed）不是
  v2 子类（fresh/dry/juice）的真值，只能通过 ``compatible`` 给出候选集合。
- ``label is None`` 的人工裁决表示“现有适用类都进不去”，是有效的未知类真值，
  用于评估弃权质量，不能当成某个叶类的训练样本。
- v2 中 ``processed`` 是父节点，叶类只有 fresh/dry/juice；预测只允许落在叶类
  或显式弃权，不允许落在父节点。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

UNKNOWN = None  # 显式未知/弃权的标签值，序列化为 null


@dataclass(frozen=True)
class Taxonomy:
    name: str
    effective: str
    nodes: dict[str, list[str]]
    compatible_out: dict[str, list[str]]

    @property
    def leaves(self) -> tuple[str, ...]:
        # 叶类 = 拥有独立条目且自身没有子节点的节点，加上仅作为子节点出现
        # 的节点（如 v2 的 dry/juice 只列在 processed 的 children 里）。
        declared_parents = {
            node for node, children in self.nodes.items() if children
        }
        all_children = {
            child for children in self.nodes.values() for child in children
        }
        leaf_set = {
            node for node in self.nodes if node not in declared_parents
        } | all_children
        return tuple(sorted(leaf_set))

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(sorted(node for node, children in self.nodes.items() if children))

    def is_leaf(self, label: Optional[str]) -> bool:
        return label in self.leaves


def build_taxonomies(taxonomies_doc: dict) -> dict[str, Taxonomy]:
    compatible = taxonomies_doc["compatible"]
    out = {}
    for name, spec in taxonomies_doc.items():
        if name == "compatible":
            continue
        out[name] = Taxonomy(
            name=name,
            effective=spec["effective"],
            nodes={node: list(children) for node, children in spec["nodes"].items()},
            compatible_out=dict(compatible),
        )
    return out


def map_label(
    label: Optional[str], source: Taxonomy, target: Taxonomy
) -> tuple[Optional[str], ...]:
    """把旧版标签映射为新版的*候选*叶类元组。

    返回空元组表示无候选（含未知裁决）；返回多元素表示只能圈定候选集合，
    不能把其中任一候选当作确定真值。
    """
    if label is None:
        return ()
    if source.name == target.name:
        return (label,) if target.is_leaf(label) else tuple(
            sorted(
                child
                for child in target.nodes.get(label, [])
            )
        )
    candidates = source.compatible_out.get(label, [])
    return tuple(sorted(candidate for candidate in candidates if target.is_leaf(candidate)))


@dataclass(frozen=True)
class Classification:
    """一个 SKU 在某类目版次、某文案版次下的判定结果。"""

    sku: str
    taxonomy: str
    revision: int
    label: Optional[str]  # None = 弃权/未知，绝不硬塞叶类
    confidence: str  # high | medium | low
    basis: str  # human | rule | model
    reasons: tuple[str, ...]
    evidence: tuple[dict, ...]
    abstain_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "taxonomy": self.taxonomy,
            "revision": self.revision,
            "label": self.label,
            "confidence": self.confidence,
            "basis": self.basis,
            "reasons": list(self.reasons),
            "evidence": list(self.evidence),
            "abstain_reason": self.abstain_reason,
        }
