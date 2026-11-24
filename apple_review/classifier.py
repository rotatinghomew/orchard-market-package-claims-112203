"""两种可比较的分类做法。

1. :class:`RuleReference` —— 可解释参照做法。只使用可逐条展示的配料事实与
   文案 token，遇到信息不全、互斥冲突一律弃权，输出理由码与证据列表。
2. :class:`NaiveBayesClassifier` —— 选用做法。多分类词项朴素贝叶斯（标准库
   纯计数实现），带弃权机制：后验优势不足或与规则参照冲突时不硬判。

两种做法共享同一套证据 token，保证对比公平；两者都只在叶类上输出，未知类
（人工 label=None）永远不进入训练，只用于评估弃权质量。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .evidence import (
    COPY_TOKEN_SIGNALS,
    INGREDIENT_FACT_SIGNALS,
    UNKNOWN_COPY_TOKENS,
    Packet,
    Product,
)
from .taxonomy import Classification, Taxonomy, UNKNOWN, map_label

# 弃权理由码
ABSTAIN_NO_FACT = "配料事实缺失且文案无品类词"
ABSTAIN_UNKNOWN_COPY = "文案声明信息不全且无配料事实"
ABSTAIN_CONFLICT = "配料事实与文案指向互斥叶类"
ABSTAIN_OLD_PARENT = "只有旧版父类标签，不能当作新版子类真值"
ABSTAIN_NO_RULING = "该版次无人工裁决"
ABSTAIN_MODEL_LOW_MARGIN = "模型后验优势低于阈值"
ABSTAIN_MODEL_DISAGREE = "模型与规则参照互斥冲突"
ABSTAIN_MODEL_UNSEEN = "模型对全部候选后验相同（无有效词项）"
ABSTAIN_MODEL_SINGLE_CLASS = "训练集仅含单一叶类，模型无法区分其他类，弃权转人工"

# 后验对数优势阈值：top1 与 top2 的 log-prob 差小于该值则弃权
DEFAULT_MARGIN = 1.5


def tokenize(product: Product, revision: int | None = None) -> list[str]:
    """模型与规则共用的特征：配料事实 token 加文案 token，带来源前缀防串义。"""
    revision = revision or product.latest_revision
    tokens: list[str] = []
    if product.ingredients is not None:
        tokens.extend(f"ing:{item}" for item in product.ingredients)
    row = product.texts[revision]
    tokens.extend(f"copy:{token}" for token in row["title"].split())
    tokens.extend(f"copy:{token}" for token in row["body"].split())
    return tokens


@dataclass
class RuleDecision:
    label: str | None
    confidence: str
    reasons: tuple[str, ...]
    abstain_reason: str | None


class RuleReference:
    """可解释规则参照：配料事实优先，文案仅作佐证，冲突即弃权。"""

    name = "rule_reference"

    def decide_v2(self, product: Product, revision: int | None = None) -> RuleDecision:
        revision = revision or product.latest_revision
        ingredient_signals: set[str] = set()
        if product.ingredients is not None:
            for item in product.ingredients:
                signal = INGREDIENT_FACT_SIGNALS.get(item)
                if signal:
                    ingredient_signals.add(signal)

        copy_signals: set[str] = set()
        copy_unknown = False
        row = product.texts[revision]
        for token in row["title"].split() + row["body"].split():
            if token in UNKNOWN_COPY_TOKENS:
                copy_unknown = True
            signal = COPY_TOKEN_SIGNALS.get(token)
            if signal:
                copy_signals.add(signal)

        # 互斥只在“跨来源”或“同一来源内出现多个不同叶类”时成立；
        # 文案里干制/果脯/开袋 同时指 dry 属于同源一致，不是冲突。
        if len(ingredient_signals) > 1:
            return RuleDecision(None, "low", (), ABSTAIN_CONFLICT)
        if len(copy_signals) > 1:
            return RuleDecision(
                None,
                "low",
                (f"文案内部指向互斥叶类 {sorted(copy_signals)}",),
                ABSTAIN_CONFLICT,
            )
        if ingredient_signals and copy_signals and ingredient_signals != copy_signals:
            return RuleDecision(
                None,
                "low",
                (
                    f"配料事实指向 {sorted(ingredient_signals)}",
                    f"文案指向 {sorted(copy_signals)}",
                ),
                ABSTAIN_CONFLICT,
            )
        if ingredient_signals:
            label = next(iter(ingredient_signals))
            if copy_signals == ingredient_signals:
                return RuleDecision(
                    label, "high", ("配料事实与文案一致",), None
                )
            return RuleDecision(
                label,
                "medium",
                ("仅有配料事实支撑，文案无一致佐证",),
                None,
            )
        # 无配料事实
        if copy_unknown:
            return RuleDecision(None, "low", (), ABSTAIN_UNKNOWN_COPY)
        if not copy_signals:
            return RuleDecision(None, "low", (), ABSTAIN_NO_FACT)
        if len(copy_signals) > 1:
            return RuleDecision(None, "low", (), ABSTAIN_CONFLICT)
        # 只有文案弱证据：可给候选但低置信，业务侧按“待核”处理
        return RuleDecision(
            next(iter(copy_signals)),
            "low",
            ("仅文案弱证据，无配料事实",),
            None,
        )

    def decide(
        self, product: Product, taxonomy: Taxonomy, revision: int | None = None
    ) -> RuleDecision:
        """在指定版次的叶类空间内判定；v1 叶类为 fruit/processed。"""
        decision = self.decide_v2(product, revision)
        if taxonomy.name == "v1" and decision.label is not None:
            mapped = "fruit" if decision.label == "fresh" else "processed"
            return RuleDecision(mapped, decision.confidence, decision.reasons,
                                decision.abstain_reason)
        return decision

    def classify(
        self,
        product: Product,
        taxonomy: Taxonomy,
        revision: int | None = None,
    ) -> Classification:
        decision = self.decide(product, taxonomy, revision)
        label = decision.label
        if label is not None and not taxonomy.is_leaf(label):
            # 理论上不会发生，双保险
            label = UNKNOWN
            decision = RuleDecision(None, "low", decision.reasons, ABSTAIN_CONFLICT)
        evidence = tuple(
            item.to_dict()
            for item in (
                *product.ingredient_evidence(),
                *product.copy_evidence(revision),
            )
        )
        return Classification(
            sku=product.sku,
            taxonomy=taxonomy.name,
            revision=revision or product.latest_revision,
            label=label,
            confidence=decision.confidence,
            basis="rule",
            reasons=decision.reasons,
            evidence=evidence,
            abstain_reason=decision.abstain_reason,
        )


class NaiveBayesClassifier:
    """带拉普拉斯平滑的多分类词项朴素贝叶斯，含弃权。"""

    name = "naive_bayes"

    def __init__(self, margin: float = DEFAULT_MARGIN):
        self.margin = margin
        self.labels: tuple[str, ...] = ()
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.vocabulary: frozenset[str] = frozenset()
        self.n_train = 0

    def fit(
        self,
        rows: list[tuple[Product, str]],
        revision: int | None = None,
        weights: list[float] | None = None,
    ) -> "NaiveBayesClassifier":
        """rows 为 (product, 叶类标签)；label=None 的样本不得进入。

        weights 为每行的证据权重：硬真值为 1.0；旧版父类软样本按候选数均分，
        权重小于 1，保证父类标签不会被当成完整子类真值。
        """
        label_counts: Counter = Counter()
        token_counts: dict[str, Counter] = defaultdict(Counter)
        token_totals: Counter = Counter()
        vocabulary: set[str] = set()
        if weights is None:
            weights = [1.0] * len(rows)
        if len(weights) != len(rows):
            raise ValueError("权重数量与训练行不一致")
        for (product, label), weight in zip(rows, weights):
            if label is None:
                raise ValueError("未知类样本不能作为叶类训练真值")
            label_counts[label] += weight
            tokens = tokenize(product, revision)
            for token in tokens:
                token_counts[label][token] += weight
            token_totals[label] += weight * len(tokens)
            vocabulary.update(tokens)
        total = sum(label_counts.values())
        self.n_train = len(rows)
        self.labels = tuple(sorted(label_counts))
        self.vocabulary = frozenset(vocabulary)
        vocab_size = len(vocabulary)
        for label in self.labels:
            self.log_prior[label] = math.log(label_counts[label] / total)
            denom = token_totals[label] + vocab_size
            self.log_likelihood[label] = {
                token: math.log((token_counts[label][token] + 1) / denom)
                for token in vocabulary
            }
        return self

    def predict_log_scores(
        self, product: Product, revision: int | None = None
    ) -> dict[str, float]:
        scores = dict(self.log_prior)
        for token in tokenize(product, revision):
            if token not in self.vocabulary:
                continue
            for label in self.labels:
                scores[label] += self.log_likelihood[label][token]
        return scores

    def explain(
        self,
        product: Product,
        revision: int | None = None,
        rule_reference: RuleReference | None = None,
    ) -> tuple[str | None, str, str | None, dict[str, float], list[str]]:
        """返回 (标签, 置信度, 弃权理由, 各叶类 log 分, 解释条目)。"""
        rule = rule_reference or RuleReference()
        if not self.labels:
            fallback = rule.decide_v2(product, revision)
            return (
                fallback.label,
                fallback.confidence,
                fallback.abstain_reason,
                {},
                ["模型无训练样本，回退规则参照", *fallback.reasons],
            )
        scores = self.predict_log_scores(product, revision)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else top_score
        explain_terms = self._term_contributions(product, top_label, revision)
        if len(self.labels) < 2:
            # 冷启动：训练仅覆盖单叶类，模型不具备多类区分能力。
            # 显式回退到可解释规则参照，不把模型弃权当成终判，也不伪造多类后验。
            fallback = rule.decide_v2(product, revision)
            return (
                fallback.label,
                fallback.confidence,
                fallback.abstain_reason,
                scores,
                [
                    f"训练叶类仅 {list(self.labels)}，单类模型回退规则参照",
                    *fallback.reasons,
                ],
            )
        if all(score == top_score for _, score in ordered):
            return None, "low", ABSTAIN_MODEL_UNSEEN, scores, ["没有任何已知词项命中"]
        if top_score - runner_up < self.margin:
            return (
                None,
                "low",
                ABSTAIN_MODEL_LOW_MARGIN,
                scores,
                [
                    f"top1 {top_label} 与次选 log 分差 {top_score - runner_up:.2f} "
                    f"< 阈值 {self.margin}",
                    *explain_terms,
                ],
            )
        confidence = "high" if top_score - runner_up >= self.margin * 2 else "medium"
        return (
            top_label,
            confidence,
            None,
            scores,
            [
                f"后验优势 {top_score - runner_up:.2f} >= 阈值 {self.margin}",
                *explain_terms,
            ],
        )

    def _term_contributions(
        self, product: Product, label: str, revision: int | None
    ) -> list[str]:
        contributions = []
        for token in tokenize(product, revision):
            if token in self.log_likelihood.get(label, {}):
                contributions.append(
                    f"{token} 对 {label} 的 log 似然 "
                    f"{self.log_likelihood[label][token]:.2f}"
                )
        contributions.sort(reverse=True)
        return contributions[:5]

    def classify(
        self,
        product: Product,
        taxonomy: Taxonomy,
        revision: int | None = None,
        rule_reference: RuleReference | None = None,
    ) -> Classification:
        revision = revision or product.latest_revision
        rule = rule_reference or RuleReference()
        label, confidence, abstain, scores, terms = self.explain(
            product, revision, rule_reference=rule
        )
        reasons = list(terms)
        fallback_used = any("回退规则参照" in term for term in terms)
        rule_decision = rule.decide_v2(product, revision)
        if rule_decision.label is None and not fallback_used:
            # 可解释参照因信息不全/互斥冲突而弃权时，模型不得用弱后验硬判，
            # 跟随弃权并保留模型分数作为人工参考。
            label = None
            confidence = "low"
            abstain = rule_decision.abstain_reason
            reasons.append(
                f"规则参照弃权（{rule_decision.abstain_reason}），"
                "模型不得仅凭弱文案词硬判，转人工"
            )
        elif label is not None and rule_decision.label is not None and label != rule_decision.label:
            # 模型与可解释参照互斥：不硬判，弃权并保留双方依据
            label, confidence, abstain = (
                None,
                "low",
                ABSTAIN_MODEL_DISAGREE,
            )
            reasons.append(
                f"模型候选与规则参照 {rule_decision.label} 互斥，转人工"
            )
        if label is not None and taxonomy.name == "v1":
            label = "fruit" if label == "fresh" else "processed"
        evidence = tuple(
            item.to_dict()
            for item in (
                *product.ingredient_evidence(),
                *product.copy_evidence(revision),
            )
        )
        score_view = {key: round(value, 4) for key, value in scores.items()}
        if scores:
            reasons.append(f"模型各类 log 分：{score_view}")
        return Classification(
            sku=product.sku,
            taxonomy=taxonomy.name,
            revision=revision,
            label=label,
            confidence=confidence,
            basis="model_fallback" if fallback_used else "model",
            reasons=tuple(reasons),
            evidence=evidence,
            abstain_reason=abstain,
        )


def classify_with_human_override(
    packet: Packet,
    sku: str,
    taxonomy: Taxonomy,
    method: str = "model",
    revision: int | None = None,
    on_or_before: str | None = None,
    as_of: str | None = None,
) -> Classification:
    """统一入口：人工裁决优先且只在同版次内生效；否则走选定做法。

    - 同版次人工 label 为叶类 -> high/human。
    - 同版次人工 label=None -> 弃权（人工未知类），模型不得覆盖。
    - 只有旧版父类标签 -> 弃权，abstain_reason 标明不能当新版子类真值，
      兼容映射仅作为候选写进理由。
    """
    on_or_before = on_or_before or as_of
    product = packet.products[sku]
    revision = revision or product.latest_revision
    ruling = product.ruling_on(taxonomy.name, on_or_before)
    if ruling is not None:
        if ruling["label"] is None:
            return Classification(
                sku=sku,
                taxonomy=taxonomy.name,
                revision=revision,
                label=UNKNOWN,
                confidence="high",
                basis="human",
                reasons=(f"人工裁决（{ruling['known']}）：现有适用类均不可归",),
                evidence=tuple(
                    item.to_dict()
                    for item in (
                        *product.ingredient_evidence(),
                        *product.copy_evidence(revision),
                    )
                ),
                abstain_reason=ABSTAIN_NO_RULING,
            )
        if taxonomy.is_leaf(ruling["label"]):
            return Classification(
                sku=sku,
                taxonomy=taxonomy.name,
                revision=revision,
                label=ruling["label"],
                confidence="high",
                basis="human",
                reasons=(f"人工裁决（{ruling['known']}）：{ruling['label']}",),
                evidence=tuple(
                    item.to_dict()
                    for item in (
                        *product.ingredient_evidence(),
                        *product.copy_evidence(revision),
                    )
                ),
            )
    # 只有旧版父类标签的情形
    older = [
        name for name in packet.data["taxonomies"] if name != "compatible" and name != taxonomy.name
    ]
    for old_name in sorted(older):
        old_ruling = product.ruling_on(old_name, on_or_before)
        if old_ruling is not None and old_ruling["label"] is not None:
            old_tax = Taxonomy(
                old_name,
                packet.data["taxonomies"][old_name]["effective"],
                packet.data["taxonomies"][old_name]["nodes"],
                packet.data["taxonomies"]["compatible"],
            )
            candidates = map_label(old_ruling["label"], old_tax, taxonomy)
            return Classification(
                sku=sku,
                taxonomy=taxonomy.name,
                revision=revision,
                label=UNKNOWN,
                confidence="low",
                basis="rule",
                reasons=(
                    f"仅有 {old_name} 父类标签 {old_ruling['label']}，"
                    f"新版候选集合 {list(candidates)} 不能当作确定真值",
                ),
                evidence=tuple(
                    item.to_dict()
                    for item in (
                        *product.ingredient_evidence(),
                        *product.copy_evidence(revision),
                    )
                ),
                abstain_reason=ABSTAIN_OLD_PARENT,
            )

    rule = RuleReference()
    if method == "rule":
        return rule.classify(product, taxonomy, revision)
    model = _fit_model_for_context(packet, taxonomy, on_or_before)
    return model.classify(product, taxonomy, revision, rule_reference=rule)


def _fit_model_for_context(
    packet: Packet,
    taxonomy: Taxonomy,
    on_or_before: str | None,
) -> NaiveBayesClassifier:
    """用该版次、截至时点的全部已知叶类裁决训练模型（供单条判定复用）。"""
    rows = []
    for product in packet.products.values():
        ruling = product.ruling_on(taxonomy.name, on_or_before)
        if ruling is not None and taxonomy.is_leaf(ruling["label"]):
            rows.append((product, ruling["label"]))
    return NaiveBayesClassifier().fit(rows)
