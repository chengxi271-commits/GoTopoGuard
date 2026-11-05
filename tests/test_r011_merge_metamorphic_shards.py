import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r011_merge_metamorphic_shards.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r011_merge_metamorphic_shards", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R011MergeMetamorphicShardsTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_merge_query_rows_rejects_duplicate_query_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard_a = root / "a.jsonl"
            shard_b = root / "b.jsonl"
            shard_a.write_text('{"query_id":"Q","position_id":"P"}\n', encoding="utf-8")
            shard_b.write_text('{"query_id":"Q","position_id":"P"}\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                self.mod.merge_query_rows([shard_a, shard_b])

    def test_merge_query_rows_keeps_unique_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard_a = root / "a.jsonl"
            shard_b = root / "b.jsonl"
            shard_a.write_text('{"query_id":"Q1","position_id":"P1"}\n', encoding="utf-8")
            shard_b.write_text('{"query_id":"Q2","position_id":"P2"}\n', encoding="utf-8")

            rows = self.mod.merge_query_rows([shard_a, shard_b])

            self.assertEqual([row["query_id"] for row in rows], ["Q1", "Q2"])


if __name__ == "__main__":
    unittest.main()
