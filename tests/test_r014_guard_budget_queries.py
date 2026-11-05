import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r014_guard_budget_queries.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r014_guard_budget_queries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R014GuardBudgetQueryTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_aggregate_budget_features_prefixes_visits(self):
        rows = [
            {"position_id": "P", "visits": 1024, "top_move": "D4", "winrate": 0.6, "scoreLead": 1.0},
            {"position_id": "P", "visits": 4096, "top_move": "Q16", "winrate": 0.7, "scoreLead": 2.0},
        ]

        agg = self.mod.aggregate_budget_features(rows)[0]

        self.assertEqual(agg["v1024_top_move"], "D4")
        self.assertEqual(agg["v4096_top_move"], "Q16")
        self.assertAlmostEqual(agg["v4096_winrate"], 0.7)


if __name__ == "__main__":
    unittest.main()
