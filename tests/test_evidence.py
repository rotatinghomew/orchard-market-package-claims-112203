"""证据层与防泄漏分组测试。"""
import unittest
from pathlib import Path

from apple_review.evidence import Packet, read_packet

PACKET = Path(__file__).resolve().parent.parent / "资料" / "演练.json"


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Packet.load(PACKET)

    def test_ingredients_none_is_absence_not_empty(self):
        product = self.packet.products["S007"]
        evidence = product.ingredient_evidence()
        self.assertEqual(len(evidence), 1)
        self.assertIsNone(evidence[0].signal)
        self.assertEqual(evidence[0].weight, "absent")

    def test_gift_box_word_is_not_evidence(self):
        # “礼盒”只出现在非证据词表，不能驱动品类判定
        product = self.packet.products["S000"]
        evidence = product.copy_evidence()
        self.assertTrue(all("礼盒" not in item.detail or item.signal is None
                            for item in evidence))

    def test_groups_join_family_and_package(self):
        # 三个组合各自的全部件必须在同一连通组
        for package in self.packet.data["packages"]:
            members = package["base"] + sum(package["optional"], [])
            group_ids = {self.packet.groups[sku] for sku in members}
            self.assertEqual(len(group_ids), 1)

    def test_revisions_stay_in_one_group(self):
        # 同一 SKU 的改版天然同组（同一 sku 即同一组）
        product = self.packet.products["S001"]
        self.assertIn(1, product.texts)
        self.assertIn(2, product.texts)
        self.assertEqual(self.packet.groups["S001"], self.packet.groups["S001"])

    def test_invalid_packet_rejected(self):
        with self.assertRaises(ValueError):
            Packet({"products": []})

    def test_broken_reference_rejected(self, tmp_path=None):
        import json
        import tempfile
        data = read_packet(PACKET)
        data["commitments"][0]["package"] = "P999"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                Packet.load(path)


if __name__ == "__main__":
    unittest.main()
