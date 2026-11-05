import importlib.util
import os
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r015b_normal_aware_utility_guard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r015b_normal_aware_utility_guard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R015BNormalAwareUtilityGuardTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_select_threshold_can_trade_agreement_for_lower_trigger_rate(self):
        rows = [
            {"score": 0.90, "effect": "correction"},
            {"score": 0.80, "effect": "correction"},
            {"score": 0.70, "effect": "regression"},
            {"score": 0.60, "effect": "preserved_correct"},
        ]

        no_penalty = self.mod.select_threshold_by_objective(rows, visit_penalty=0.0, max_trigger_rate=None)
        with_penalty = self.mod.select_threshold_by_objective(rows, visit_penalty=1.10, max_trigger_rate=None)

        self.assertGreater(no_penalty["trigger_rate"], with_penalty["trigger_rate"])
        self.assertGreater(no_penalty["agreement_rate"], with_penalty["agreement_rate"])
        self.assertEqual(with_penalty["correction_count"], 0)
        self.assertEqual(with_penalty["threshold"], float("inf"))

    def test_group_key_uses_dataset_specific_holdout(self):
        self.assertEqual(self.mod.group_key({"dataset": "D3", "game_index": 7}), ("D3", 7))
        self.assertEqual(self.mod.group_key({"dataset": "D5", "family": "cyclic"}), ("D5", "cyclic"))
        key = self.mod.group_key({"dataset": "D1", "position_id": "P"})
        self.assertEqual(key[0], "N")
        self.assertIsInstance(key[1], int)

    def test_group_parallel_jobs_is_env_controlled(self):
        old_value = os.environ.get("GOTOPOGUARD_GROUP_N_JOBS")
        try:
            os.environ.pop("GOTOPOGUARD_GROUP_N_JOBS", None)
            self.assertEqual(self.mod.group_parallel_jobs(), 1)
            os.environ["GOTOPOGUARD_GROUP_N_JOBS"] = "4"
            self.assertEqual(self.mod.group_parallel_jobs(), 4)
        finally:
            if old_value is None:
                os.environ.pop("GOTOPOGUARD_GROUP_N_JOBS", None)
            else:
                os.environ["GOTOPOGUARD_GROUP_N_JOBS"] = old_value

    def test_summarize_records_counts_trigger_rate(self):
        rows = [
            {"oracle_top_move": "A", "g0_top_move": "B", "method_top_move": "A", "method_visits": 1024, "triggered": True},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "A", "method_visits": 256, "triggered": False},
        ]

        summary = self.mod.summarize_method(rows)

        self.assertEqual(summary["correction_count"], 1)
        self.assertEqual(summary["regression_count"], 0)
        self.assertAlmostEqual(summary["trigger_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
