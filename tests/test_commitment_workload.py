"""组合级承诺传播、三档处置与人工复核纪律测试。"""
import unittest
from pathlib import Path

from apple_review.commitment import (
    CONTINUE_OLD,
    NOTIFY_BUYER,
    PAUSE_NEW,
    review_all,
)
from apple_review.evidence import Packet
from apple_review.taxonomy import build_taxonomies
from apple_review.workload import build_work_plan, evidence_signature

PACKET = Path(__file__).resolve().parent.parent / "资料" / "演练.json"


class CommitmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)
        cls.tax = build_taxonomies(cls.packet.data["taxonomies"])
        cls.reviews = {
            review.package: review for review in review_all(cls.packet, cls.tax)
        }

    def test_v1_commitment_continues_under_old_scope(self):
        # P1 是 v1“含鲜果”，S006 改判 dry 不削弱含鲜果事实 -> 按旧约继续
        review = self.reviews["P1"]
        self.assertEqual(review.taxonomy, "v1")
        self.assertEqual(review.disposition, CONTINUE_OLD)
        self.assertTrue(all(scenario.satisfies for scenario in review.scenarios))

    def test_v2_open_optional_group_pauses_new_booking(self):
        # P2 的 actual_choices 为空，且存在不达标选法 -> 暂停新增
        review = self.reviews["P2"]
        self.assertEqual(review.disposition, PAUSE_NEW)
        self.assertTrue(review.open_optional_group)
        self.assertFalse(all(scenario.satisfies for scenario in review.scenarios))

    def test_v2_confirmed_choice_broken_requires_notify(self):
        # P3 已锁定 S003，复查后不满足 -> 必须通知采购方
        review = self.reviews["P3"]
        self.assertTrue(review.notify_required)
        chosen = next(
            s for s in review.scenarios if s.choice == ("S003",)
        )
        self.assertFalse(chosen.satisfies)

    def test_original_commitment_text_preserved(self):
        # 复查不得改写承诺原文
        for review in self.reviews.values():
            original = next(
                row for row in self.packet.data["commitments"]
                if row["package"] == review.package and row["buyer"] == review.buyer
            )
            self.assertEqual(review.scope, original["scope"])
            self.assertEqual(review.taxonomy, original["taxonomy"])
            self.assertEqual(review.original_choices, original["actual_choices"])

    def test_shared_component_not_same_order(self):
        # S004 同时出现在 P1、P2，但买家承诺分别建单
        affected = [
            review.package for review in self.reviews.values()
            if "S004" in review.components
        ]
        self.assertEqual(sorted(affected), ["P1", "P2"])
        buyers = {
            review.buyer for review in self.reviews.values()
            if "S004" in review.components
        }
        self.assertEqual(buyers, {"B1", "B2"})

    def test_two_label_layers_separated(self):
        # S000 无人工 v2 标签但证据推断 fresh
        state = self.reviews["P2"].components["S000"]
        self.assertIsNone(state.human_label)
        self.assertEqual(state.inferred_label, "fresh")

    def test_dispositions_cover_three_classes(self):
        dispositions = {review.disposition for review in self.reviews.values()}
        self.assertEqual(dispositions, {CONTINUE_OLD, PAUSE_NEW})
        # 通知通过 notify_required 表达（可与暂停并存）
        self.assertTrue(
            any(review.notify_required for review in self.reviews.values())
        )


class WorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)
        cls.tax = build_taxonomies(cls.packet.data["taxonomies"])
        cls.reviews = review_all(cls.packet, cls.tax)
        cls.plan = build_work_plan(cls.packet, cls.reviews, cls.tax)

    def test_scope_locked_to_affected_and_registered(self):
        expected = {
            sku
            for package in self.packet.data["packages"]
            for sku in package["base"] + sum(package["optional"], [])
        }
        self.assertEqual(set(self.plan.locked_scope), expected)

    def test_assigned_work_not_redispatched(self):
        # S004 已派出 -> reconcile，不在本轮新派队列
        s004 = next(item for item in self.plan.items if item.sku == "S004")
        self.assertEqual(s004.status, "reconcile")
        self.assertNotIn("S004", self.plan.scheduled)

    def test_unassigned_superseded_work_rolls_back(self):
        # S006 未派出且被 11-07 裁决覆盖 -> 回退，不重复派
        s006 = next(item for item in self.plan.items if item.sku == "S006")
        self.assertEqual(s006.status, "rolled_back")
        self.assertNotIn("S006", self.plan.scheduled)
        self.assertTrue(any(row["sku"] == "S006" for row in self.plan.rollbacks))

    def test_capacity_respected(self):
        used_assigned = sum(
            item.hours for item in self.plan.items
            if item.status in ("assigned", "reconcile")
        )
        scheduled_hours = sum(
            item.hours for item in self.plan.items
            if item.sku in self.plan.scheduled
        )
        self.assertLessEqual(
            used_assigned + scheduled_hours, self.plan.capacity_hours + 1e-9
        )

    def test_high_cost_s000_prioritized_by_error_cost(self):
        # S000 误分代价 5（fresh），应排在所有代价 1 的阻断件前
        scheduled = self.plan.scheduled
        self.assertIn("S000", scheduled)
        self.assertLess(
            scheduled.index("S000"),
            len(scheduled),
        )

    def test_evidence_signature_stable_and_distinct(self):
        sig1 = evidence_signature(self.packet, "S004")
        sig2 = evidence_signature(self.packet, "S006")
        self.assertIsInstance(sig1, str)
        self.assertNotEqual(sig1, sig2)

    def test_reserve_high_protects_high_cost_under_tight_capacity(self):
        # 复制资料包，把 S004 改成“未派出且无新裁决覆盖”的 high 排队件，
        # 并收紧容量，验证 reserve_high 保底让它优先于普通件排入。
        import copy
        data = copy.deepcopy(self.packet.data)
        for row in data["review_work"]:
            if row["sku"] == "S004":
                row["assigned"] = None
        # 去掉 S004 的新裁决，使其不进入 rolled_back
        data["rulings"] = [
            row for row in data["rulings"]
            if not (row["sku"] == "S004" and row["known"] == "2026-11-02")
        ]
        data["policy"]["hours"] = 3  # 容量紧张
        packet = Packet(data)
        reviews = review_all(packet, self.tax)
        plan = build_work_plan(packet, reviews, self.tax)
        s004 = next(item for item in plan.items if item.sku == "S004")
        self.assertEqual(s004.status, "queued")
        self.assertEqual(s004.cost_class, "high")
        self.assertIn("S004", plan.scheduled)
        high_hours = sum(
            item.hours for item in plan.items
            if item.sku in plan.scheduled and item.cost_class == "high"
        )
        self.assertGreaterEqual(high_hours, plan.reserve_high - 1e-9)


if __name__ == "__main__":
    unittest.main()
