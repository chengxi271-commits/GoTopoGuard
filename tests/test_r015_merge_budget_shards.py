import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r015_merge_budget_shards.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r015_merge_budget_shards", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R015MergeBudgetShardsTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_merge_query_rows_rejects_duplicate_query_id(self):
        rows = [{"query_id": "q1", "position_id": "P", "visits": 1024}, {"query_id": "q1", "position_id": "P", "visits": 4096}]

        with self.assertRaises(ValueError):
            self.mod.merge_query_rows(rows)

    def test_collect_shard_rows_reads_all_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard0 = root / "shard_00"
            shard1 = root / "shard_01"
            shard0.mkdir()
            shard1.mkdir()
            (shard0 / "query_budget_features.jsonl").write_text(
                '{"query_id":"q0","position_id":"P0","visits":1024}\n', encoding="utf-8"
            )
            (shard1 / "query_budget_features.jsonl").write_text(
                '{"query_id":"q1","position_id":"P1","visits":1024}\n', encoding="utf-8"
            )

            rows = self.mod.collect_shard_rows(root)

            self.assertEqual({row["query_id"] for row in rows}, {"q0", "q1"})


if __name__ == "__main__":
    unittest.main()
