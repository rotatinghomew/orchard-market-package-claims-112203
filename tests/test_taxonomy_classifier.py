"""类目版次、标签版本语义与分类器护栏测试。"""
import unittest
from pathlib import Path

from apple_review.classifier import (
    ABSTAIN_CONFLICT,
    ABSTAIN_OLD_PARENT,
    NaiveBayesClassifier,
    RuleReference,
    classify_with_human_override,
)
from apple_review.evidence import Packet
from apple_review.taxonomy import UNKNOWN, build_taxonomies, map_label

PACKET = Path(__file__).resolve().parent.parent / "资料" / "演练.json"


class TaxonomyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)
        cls.tax = build_taxonomies(cls.packet.data["taxonomies"])

    def test_v2_leaves_include_declared_children(self):
        self.assertEqual(set(self.tax["v2"].leaves), {"fresh", "dry", "juice"})
        self.assertEqual(self.tax["v2"].parents, ("processed",))

    def test_v1_leaves(self):
        self.assertEqual(set(self.tax["v1"].leaves), {"fruit", "processed"})

    def test_old_parent_only_gives_candidates_not_truth(self):
        # fruit -> (fresh,) 候选；processed -> (dry, juice) 多候选
        self.assertEqual(map_label("fruit", self.tax["v1"], self.tax["v2"]), ("fresh",))
        self.assertEqual(
            map_label("processed", self.tax["v1"], self.tax["v2"]),
            ("dry", "juice"),
        )
        # 未知裁决不产生候选
        self.assertEqual(map_label(None, self.tax["v1"], self.tax["v2"]), ())

    def test_old_parent_label_abstains_in_v2(self):
        # S001 只有 v1 fruit 裁决，在 v2 下必须弃权，不能直接当 fresh 真值
        result = classify_with_human_override(self.packet, "S001", self.tax["v2"])
        self.assertIsNone(result.label)
        self.assertEqual(result.abstain_reason, ABSTAIN_OLD_PARENT)

    def test_human_null_is_abstention_not_leaf(self):
        result = classify_with_human_override(self.packet, "S004", self.tax["v2"])
        self.assertIsNone(result.label)
        self.assertEqual(result.basis, "human")

    def test_human_leaf_overrides_model(self):
        result = classify_with_human_override(self.packet, "S006", self.tax["v2"])
        self.assertEqual(result.label, "dry")
        self.assertEqual(result.basis, "human")
        self.assertEqual(result.confidence, "high")


class RuleReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)
        cls.tax = build_taxonomies(cls.packet.data["taxonomies"])
        cls.rule = RuleReference()

    def test_consistent_fact_and_copy_high(self):
        decision = self.rule.decide_v2(self.packet.products["S074"])
        self.assertEqual(decision.label, "fresh")
        self.assertEqual(decision.confidence, "high")

    def test_missing_ingredients_unknown_copy_abstains(self):
        decision = self.rule.decide_v2(self.packet.products["S073"])
        self.assertIsNone(decision.label)
        self.assertIsNotNone(decision.abstain_reason)

    def test_generic_giftbox_with_ingredient_uses_fact(self):
        # S060 文案只有包装词，但配料切片 -> dry
        decision = self.rule.decide_v2(self.packet.products["S060"])
        self.assertEqual(decision.label, "dry")

    def test_human_ruling_overrides_consistent_but_wrong_signals(self):
        # S006：配料与文案都指 juice，但人工 v2 裁决为 dry。
        # 规则给出 juice（依据可追溯），人工裁决在统一入口覆盖为 dry。
        product = self.packet.products["S006"]
        decision = self.rule.decide_v2(product)
        self.assertEqual(decision.label, "juice")
        overridden = classify_with_human_override(
            self.packet, "S006", self.tax["v2"]
        )
        self.assertEqual(overridden.label, "dry")
        self.assertEqual(overridden.basis, "human")

    def test_synthetic_cross_source_conflict_abstains(self):
        # 构造一个配料指 fresh、文案指 dry 的互斥件，规则必须弃权
        from apple_review.evidence import Product
        product = Product(
            sku="X1", merchant="M", family="X", ingredients=["纯苹果"],
            net_g=100, launched="2026-10-01",
            texts={2: {"sku": "X1", "revision": 2, "published": "2026-10-10",
                       "title": "果脯 X1", "body": "干制 果脯 开袋"}},
            rulings=[],
        )
        decision = self.rule.decide_v2(product)
        self.assertIsNone(decision.label)
        self.assertEqual(decision.abstain_reason, ABSTAIN_CONFLICT)


class ModelGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)
        cls.tax = build_taxonomies(cls.packet.data["taxonomies"])

    def test_model_cannot_force_leaf_when_rule_abstains(self):
        # 用 v2 已有人工叶类训练，再对无配料无品类词的 S076 判定，
        # 规则弃权时模型必须跟随弃权
        rows = []
        for product in self.packet.products.values():
            ruling = product.ruling_on("v2")
            if ruling is not None and self.tax["v2"].is_leaf(ruling["label"]):
                rows.append((product, ruling["label"]))
        model = NaiveBayesClassifier().fit(rows)
        result = model.classify(
            self.packet.products["S076"], self.tax["v2"],
            rule_reference=RuleReference(),
        )
        self.assertIsNone(result.label)
        self.assertIsNotNone(result.abstain_reason)

    def test_single_class_training_falls_back_to_rule(self):
        rows = [
            (self.packet.products["S074"], "fresh"),
            (self.packet.products["S077"], "fresh"),
        ]
        model = NaiveBayesClassifier().fit(rows)
        result = model.classify(
            self.packet.products["S075"], self.tax["v2"],
            rule_reference=RuleReference(),
        )
        # 单类模型回退规则，S075 配料开袋 -> dry
        self.assertEqual(result.label, "dry")
        self.assertEqual(result.basis, "model_fallback")

    def test_unknown_label_never_enters_training(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().fit(
                [(self.packet.products["S004"], UNKNOWN)]
            )


if __name__ == "__main__":
    unittest.main()
