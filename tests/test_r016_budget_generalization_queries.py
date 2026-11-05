import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r016_budget_generalization_queries.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r016_budget_generalization_queries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R016BudgetGeneralizationQueryTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_stable_sample_is_deterministic_and_limited(self):
        rows = [{"position_id": f"P{i:03d}"} for i in range(20)]

        first = self.mod.stable_sample(rows, limit=5, key_name="position_id")
        second = self.mod.stable_sample(list(reversed(rows)), limit=5, key_name="position_id")

        self.assertEqual([row["position_id"] for row in first], [row["position_id"] for row in second])
        self.assertEqual(len(first), 5)

    def test_shard_records_keeps_position_visits_together(self):
        records = [{"position_id": f"P{i:03d}"} for i in range(30)]

        shard0 = self.mod.shard_records(records, shard_index=0, shard_count=3)
        shard1 = self.mod.shard_records(records, shard_index=1, shard_count=3)
        shard2 = self.mod.shard_records(records, shard_index=2, shard_count=3)
        all_ids = {row["position_id"] for row in shard0 + shard1 + shard2}

        self.assertEqual(all_ids, {row["position_id"] for row in records})
        self.assertFalse({row["position_id"] for row in shard0} & {row["position_id"] for row in shard1})

    def test_aggregate_budget_features_prefixes_all_visits(self):
        rows = [
            {"position_id": "P1", "visits": 1024, "top_move": "A1", "scoreLead": 1.0, "winrate": 0.6},
            {"position_id": "P1", "visits": 8192, "top_move": "B2", "scoreLead": 2.5, "winrate": 0.7},
        ]

        out = self.mod.aggregate_budget_features(rows)

        self.assertEqual(out[0]["v1024_top_move"], "A1")
        self.assertEqual(out[0]["v8192_top_move"], "B2")
        self.assertAlmostEqual(out[0]["budget_score_abs_delta_1024_8192"], 1.5)


if __name__ == "__main__":
    unittest.main()
