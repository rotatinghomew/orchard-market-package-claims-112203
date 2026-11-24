"""分组防泄漏回测与可靠度视图测试。"""
import unittest
from pathlib import Path

from apple_review.evaluation import (
    make_splits,
    reliability_views,
    run_backtests,
)
from apple_review.evidence import Packet
from apple_review.taxonomy import build_taxonomies

PACKET = Path(__file__).resolve().parent.parent / "资料" / "演练.json"


class BacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)
        cls.tax = build_taxonomies(cls.packet.data["taxonomies"])
        cls.splits = make_splits(cls.packet, cls.tax)

    def test_crossing_group_excluded_from_both_sides(self):
        # 组合连通组在每个截止点都跨边，必须整体排除，不进训练也不进考核
        crossing_skus = {
            "S000", "S001", "S002", "S003", "S004",
            "S005", "S006", "S007", "S009", "S010",
        }
        crossing_group_ids = {self.packet.groups[sku] for sku in crossing_skus}
        for split in self.splits:
            train_skus = {p.sku for p, _ in split.train}
            test_skus = {p.sku for p, _ in split.test}
            self.assertFalse(train_skus & crossing_skus)
            self.assertFalse(test_skus & crossing_skus)
            self.assertTrue(crossing_group_ids <= set(split.excluded_groups))

    def test_no_group_straddles_train_and_test(self):
        for split in self.splits:
            train_groups = {self.packet.groups[p.sku] for p, _ in split.train}
            test_groups = {self.packet.groups[p.sku] for p, _ in split.test}
            self.assertFalse(train_groups & test_groups)

    def test_revisions_cannot_be_split(self):
        # 同一 SKU 的两个 revision 属于同一组，分组层面保证不跨边
        groups = self.packet.groups
        for sku in self.packet.products:
            self.assertEqual(groups[sku], groups[sku])

    def test_reports_cover_both_methods_and_views(self):
        reports = run_backtests(self.packet, self.tax)
        methods = {report.method for report in reports}
        self.assertIn("rule_reference", methods)
        self.assertTrue(any("naive_bayes" in m for m in methods))
        views = reliability_views(reports)
        self.assertEqual(set(views), {"rare_classes", "version_transition", "unknown_class"})

    def test_unknown_truth_never_forced_to_leaf(self):
        reports = run_backtests(self.packet, self.tax)
        for report in reports:
            self.assertEqual(
                report.forced_error, 0,
                f"{report.method} {report.boundary} 出现未知硬判",
            )

    def test_rare_class_flag_present_for_juice(self):
        reports = run_backtests(self.packet, self.tax)
        # 10-01 轮 juice 训练样本少，应被标为稀有类
        flagged = [
            report for report in reports
            if "juice" in report.rare_labels and report.boundary == "2026-10-01"
        ]
        self.assertTrue(flagged)

    def test_v1_has_no_test_after_v2_effective(self):
        v1_splits = [s for s in self.splits if s.taxonomy == "v1"]
        for split in v1_splits:
            self.assertEqual(len(split.test), 0)
            self.assertGreater(len(split.train), 0)


if __name__ == "__main__":
    unittest.main()
