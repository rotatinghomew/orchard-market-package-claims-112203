"""核对演练资料的公开输入约定。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inspect_packet import read_packet


class PacketTest(unittest.TestCase):
    def test_related_records(self):
        doc = read_packet(Path("资料/演练.json"))
        sku={row['sku'] for row in doc['products']}
        self.assertTrue(len(sku)==len(doc['products']) and {row['sku'] for row in doc['text']}==sku)
        self.assertTrue({row['sku'] for row in doc['rulings'] + doc['review_work']} <= sku)
        self.assertTrue(len({(row['sku'],row['revision']) for row in doc['text']})==len(doc['text']))
        self.assertTrue(any(row['label'] is None for row in doc['rulings']))
        self.assertTrue(all(set(row['base'] + sum(row['optional'],[]))<=sku for row in doc['packages']))
        self.assertTrue({row['package'] for row in doc['commitments']} <= {row['package'] for row in doc['packages']})

    def test_reject_wrong_container(self):
        with TemporaryDirectory() as folder:
            packet = Path(folder) / "unexpected.json"
            packet.write_text(json.dumps({"invalid": 3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_packet(packet)
