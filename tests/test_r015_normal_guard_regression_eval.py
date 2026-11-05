import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r015_normal_guard_regression_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r015_normal_guard_regression_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R015NormalGuardRegressionEvalTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_guard_choice_uses_high_budget_only_when_triggered(self):
        row = {"v256_top_move": "D4", "v1024_top_move": "Q16"}

        self.assertEqual(self.mod.method_top_move(row, triggered=False), "D4")
        self.assertEqual(self.mod.method_top_move(row, triggered=True), "Q16")

    def test_summarize_method_counts_normal_regressions_and_corrections(self):
        rows = [
            {"oracle_top_move": "A", "g0_top_move": "B", "method_top_move": "A", "method_visits": 1024, "triggered": True},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "B", "method_visits": 1024, "triggered": True},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "A", "method_visits": 256, "triggered": False},
        ]

        summary = self.mod.summarize_method(rows)

        self.assertEqual(summary["normal_correction_count"], 1)
        self.assertEqual(summary["normal_regression_count"], 1)
        self.assertEqual(summary["net_normal_gain"], 0)
        self.assertAlmostEqual(summary["oracle_agreement_rate"], 2 / 3)

    def test_attach_budget_features_keeps_only_rows_with_budget(self):
        feature_rows = [
            {"position_id": "A", "dataset": "D1", "v256_top_move": "D4"},
            {"position_id": "B", "dataset": "D2", "v256_top_move": "Q16"},
        ]
        budget_rows = [{"position_id": "A", "v1024_top_move": "C3", "v4096_top_move": "C3"}]

        rows = self.mod.attach_normal_budget_features(feature_rows, budget_rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["position_id"], "A")
        self.assertEqual(rows[0]["oracle_top_move"], "C3")

    def test_stable_natural_fold_falls_back_to_position_id(self):
        row = {"position_id": "R006_jgdb_00001"}

        self.assertEqual(
            self.mod.stable_natural_fold(row, fold_count=5),
            self.mod.stable_natural_fold(row, fold_count=5),
        )


if __name__ == "__main__":
    unittest.main()
