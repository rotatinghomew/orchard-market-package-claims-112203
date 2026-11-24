"""按历史截止点的分组回测与可靠度视图。

回测纪律（见 资料格式.md 的 policy.boundaries）：

- 每个截止点一轮：``known < boundary`` 的裁决可用于训练，
  ``known >= boundary`` 的用于考核。
- 连通组（同 family 或同 package 同件）只要在两侧都出现，整组从本轮剔除，
  同一商品的改标题（多 revision）和组合子件因此不可能被拆进训练和考核。
- 训练与考核按类目版次分别进行：v1 模型只对 v1 真值打分，v2 同理。
  旧版父类标签不是新版子类真值，只通过 compatible 进入“软训练”候选集合，
  且单独标记、单独报告，绝不与硬真值混算。
- 人工 label=None 是未知类真值，用于统计“弃权是否正确”，不进训练。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .classifier import (
    NaiveBayesClassifier,
    RuleReference,
)
from .evidence import Packet
from .taxonomy import Taxonomy, map_label


@dataclass
class Split:
    boundary: str
    taxonomy: str
    train: list  # list[(Product, label)]，硬真值，label 均为该版叶类
    train_soft: list  # list[(Product, (候选叶类,...))]，旧版父类映射，仅候选集
    test: list  # list[(Product, label|None)]，None 是未知类真值
    excluded_groups: list[str]
    train_groups: int
    test_groups: int


def make_splits(packet: Packet, taxonomies: dict[str, Taxonomy]) -> list[Split]:
    splits: list[Split] = []
    boundaries = packet.data["policy"]["boundaries"]
    for boundary in boundaries:
        train_groups, test_groups, crossing = set(), set(), set()
        for group_id, skus in packet.skus_by_group.items():
            in_train = any(
                ruling["known"] < boundary
                for sku in skus
                for ruling in packet.products[sku].rulings
            )
            in_test = any(
                ruling["known"] >= boundary
                for sku in skus
                for ruling in packet.products[sku].rulings
            )
            if in_train and in_test:
                crossing.add(group_id)
            elif in_train:
                train_groups.add(group_id)
            elif in_test:
                test_groups.add(group_id)

        for tax_name, taxonomy in taxonomies.items():
            train_rows, soft_rows, test_rows = [], [], []
            for product in packet.products.values():
                group_id = packet.groups[product.sku]
                if group_id in crossing:
                    continue
                for ruling in product.rulings:
                    if ruling["known"] >= boundary:
                        if ruling["taxonomy"] == tax_name and group_id in test_groups:
                            test_rows.append((product, ruling["label"]))
                        continue
                    if group_id not in train_groups:
                        continue
                    if ruling["taxonomy"] == tax_name and taxonomy.is_leaf(ruling["label"]):
                        train_rows.append((product, ruling["label"]))
                    elif ruling["taxonomy"] != tax_name and ruling["label"] is not None:
                        source_tax = taxonomies[ruling["taxonomy"]]
                        candidates = map_label(ruling["label"], source_tax, taxonomy)
                        if candidates:
                            soft_rows.append((product, candidates))
            splits.append(
                Split(
                    boundary=boundary,
                    taxonomy=tax_name,
                    train=train_rows,
                    train_soft=soft_rows,
                    test=test_rows,
                    excluded_groups=sorted(crossing),
                    train_groups=len(train_groups),
                    test_groups=len(test_groups),
                )
            )
    return splits


@dataclass
class RoundReport:
    boundary: str
    taxonomy: str
    method: str
    n_train: int
    n_train_soft: int
    n_test: int
    n_leaf_test: int
    n_unknown_test: int
    labels_in_train: dict[str, int]
    labels_in_test: dict
    accuracy: float | None
    abstain_rate: float
    abstain_correct: int  # 真未知且弃权
    abstain_wrong: int  # 真叶类却弃权（覆盖不足）
    forced_error: int  # 真未知却硬判叶类（严重错误）
    per_label: dict[str, dict] = field(default_factory=dict)
    rare_labels: list[str] = field(default_factory=list)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    n_excluded_groups: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "boundary": self.boundary,
            "taxonomy": self.taxonomy,
            "method": self.method,
            "n_train": self.n_train,
            "n_train_soft": self.n_train_soft,
            "n_test": self.n_test,
            "n_leaf_test": self.n_leaf_test,
            "n_unknown_test": self.n_unknown_test,
            "labels_in_train": self.labels_in_train,
            "labels_in_test": self.labels_in_test,
            "accuracy": self.accuracy,
            "abstain_rate": round(self.abstain_rate, 4),
            "abstain_correct": self.abstain_correct,
            "abstain_wrong": self.abstain_wrong,
            "forced_error": self.forced_error,
            "per_label": self.per_label,
            "rare_labels": self.rare_labels,
            "confusion": self.confusion,
            "n_excluded_groups": self.n_excluded_groups,
            "note": self.note,
        }


def _soft_expand(split: Split) -> tuple[list, list[float]]:
    """软训练行展开：硬真值权重 1；父类候选按候选数均分权重。

    显式弱权重用法，报告里标注 n_train_soft，不把父类标签当作人工子类真值；
    仅用于让模型在新版冷启动时具备候选区分能力。
    """
    rows = list(split.train)
    weights = [1.0] * len(rows)
    for product, candidates in split.train_soft:
        share = 1.0 / len(candidates)
        for candidate in candidates:
            rows.append((product, candidate))
            weights.append(share)
    return rows, weights


def _evaluate_method(
    split: Split,
    taxonomy: Taxonomy,
    method: str,
    rare_threshold: int,
    use_soft: bool,
) -> RoundReport:
    rule = RuleReference()
    note_parts = []
    if method == "rule":
        model = None
    else:
        if use_soft:
            train_rows, weights = _soft_expand(split)
        else:
            train_rows, weights = split.train, None
        model = (
            NaiveBayesClassifier().fit(train_rows, weights=weights)
            if train_rows
            else NaiveBayesClassifier()
        )
        if use_soft and split.train_soft:
            note_parts.append(
                f"含 {len(split.train_soft)} 条旧版父类软样本（仅候选集，非子类真值）"
            )
        if len(split.train) < 2:
            note_parts.append("硬真值不足，模型冷启动，主要依赖规则护栏")

    predictions: list[tuple[str | None, str | None]] = []
    for product, truth in split.test:
        if method == "rule":
            decision = rule.classify(product, taxonomy)
        else:
            decision = model.classify(product, taxonomy, rule_reference=rule)
        predictions.append((truth, decision.label))

    leaf_pairs = [(t, p) for t, p in predictions if t is not None]
    n_unknown = sum(1 for t, _ in predictions if t is None)
    abstain_correct = sum(1 for t, p in predictions if t is None and p is None)
    abstain_wrong = sum(1 for t, p in predictions if t is not None and p is None)
    forced_error = sum(1 for t, p in predictions if t is None and p is not None)
    correct = sum(1 for t, p in leaf_pairs if t == p)
    accuracy = correct / len(leaf_pairs) if leaf_pairs else None

    per_label = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for truth, pred in leaf_pairs:
        confusion[truth][pred if pred is not None else "ABSTAIN"] += 1
    for label in sorted(taxonomy.leaves):
        total = sum(1 for t, _ in leaf_pairs if t == label)
        if total == 0:
            continue
        hit = sum(1 for t, p in leaf_pairs if t == label and p == label)
        per_label[label] = {
            "support": total,
            "precision_support": hit,
            "recall": round(hit / total, 4),
            "abstained": sum(1 for t, p in leaf_pairs if t == label and p is None),
            "misclassified_as": {
                pred: count
                for pred, count in confusion[label].items()
                if pred != label
            },
            "rare": False,
        }
    train_counts = Counter(label for _, label in split.train)
    rare_labels = sorted(
        label for label, count in train_counts.items() if count <= rare_threshold
    )
    for label in rare_labels:
        if label in per_label:
            per_label[label]["rare"] = True
    if not split.test:
        note_parts.append("该截止点后无同版考核样本，旧版真值不可借作新版考核")

    return RoundReport(
        boundary=split.boundary,
        taxonomy=taxonomy.name,
        method=method + ("+soft" if use_soft and split.train_soft else ""),
        n_train=len(split.train),
        n_train_soft=len(split.train_soft),
        n_test=len(split.test),
        n_leaf_test=len(leaf_pairs),
        n_unknown_test=n_unknown,
        labels_in_train=dict(sorted(train_counts.items())),
        labels_in_test=dict(
            sorted(
                Counter(label for _, label in split.test).items(),
                key=lambda item: (item[0] is None, str(item[0])),
            )
        ),
        accuracy=None if accuracy is None else round(accuracy, 4),
        abstain_rate=round((abstain_correct + abstain_wrong) / len(predictions), 4)
        if predictions
        else 0.0,
        abstain_correct=abstain_correct,
        abstain_wrong=abstain_wrong,
        forced_error=forced_error,
        per_label=per_label,
        rare_labels=rare_labels,
        confusion={k: dict(v) for k, v in confusion.items()},
        n_excluded_groups=len(split.excluded_groups),
        note="；".join(note_parts),
    )


def run_backtests(
    packet: Packet,
    taxonomies: dict[str, Taxonomy],
    rare_threshold: int = 5,
) -> list[RoundReport]:
    """同一组切分上的做法对比。

    每种做法两个口径：仅硬真值、硬真值加旧版父类软候选（后者单独标注）。
    n_test=0 的轮次保留，用于换版前后视图说明旧版无考核、不能借旧真值。
    """
    reports = []
    for split in make_splits(packet, taxonomies):
        taxonomy = taxonomies[split.taxonomy]
        reports.append(_evaluate_method(split, taxonomy, "rule_reference",
                                        rare_threshold, use_soft=False))
        reports.append(_evaluate_method(split, taxonomy, "naive_bayes",
                                        rare_threshold, use_soft=False))
        if split.train_soft:
            reports.append(_evaluate_method(split, taxonomy, "naive_bayes",
                                            rare_threshold, use_soft=True))
    return reports


def reliability_views(reports: list[RoundReport]) -> dict:
    """运营要的三个可靠度视图：稀有类、换版前后、未知类。"""
    rare_view = []
    for report in reports:
        for label in report.rare_labels:
            if label in report.per_label:
                rare_view.append(
                    {
                        "boundary": report.boundary,
                        "taxonomy": report.taxonomy,
                        "method": report.method,
                        "label": label,
                        **report.per_label[label],
                    }
                )
    version_view = {}
    for report in reports:
        key = (report.taxonomy, report.method)
        version_view.setdefault(key, []).append(
            {
                "boundary": report.boundary,
                "n_train": report.n_train,
                "n_train_soft": report.n_train_soft,
                "n_test": report.n_test,
                "accuracy": report.accuracy,
                "abstain_rate": report.abstain_rate,
                "labels_in_train": report.labels_in_train,
                "note": report.note,
            }
        )
    unknown_view = [
        {
            "boundary": report.boundary,
            "taxonomy": report.taxonomy,
            "method": report.method,
            "unknown_support": report.n_unknown_test,
            "abstain_correct": report.abstain_correct,
            "abstain_wrong_on_leaf": report.abstain_wrong,
            "forced_error_on_unknown": report.forced_error,
        }
        for report in reports
    ]
    return {
        "rare_classes": rare_view,
        "version_transition": {
            f"{tax}::{method}": rows
            for (tax, method), rows in version_view.items()
        },
        "unknown_class": unknown_view,
    }
