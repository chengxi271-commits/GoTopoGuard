import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r014b_utility_guard_redesign.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r014b_utility_guard_redesign", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R014BUtilityGuardRedesignTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_extra_search_effect_labels_correction_and_regression(self):
        self.assertEqual(
            self.mod.extra_search_effect({"v256_top_move": "A", "v1024_top_move": "B", "v4096_top_move": "B"}),
            "correction",
        )
        self.assertEqual(
            self.mod.extra_search_effect({"v256_top_move": "A", "v1024_top_move": "B", "v4096_top_move": "A"}),
            "regression",
        )
        self.assertEqual(
            self.mod.extra_search_effect({"v256_top_move": "A", "v1024_top_move": "A", "v4096_top_move": "A"}),
            "preserved_correct",
        )
        self.assertEqual(
            self.mod.extra_search_effect({"v256_top_move": "A", "v1024_top_move": "B", "v4096_top_move": "C"}),
            "unresolved_error",
        )

    def test_select_threshold_prefers_corrections_over_regressions(self):
        rows = [
            {"score": 0.90, "effect": "correction"},
            {"score": 0.70, "effect": "correction"},
            {"score": 0.55, "effect": "regression"},
            {"score": 0.10, "effect": "preserved_correct"},
        ]

        selected = self.mod.select_threshold_by_training_utility(rows, max_trigger_rate=None)

        self.assertGreater(selected["threshold"], 0.55)
        self.assertEqual(selected["correction_count"], 2)
        self.assertEqual(selected["regression_count"], 0)
        self.assertAlmostEqual(selected["oracle_agreement_rate"], 1.0)

    def test_summarize_utility_method_counts_net_gain(self):
        rows = [
            {"oracle_top_move": "A", "g0_top_move": "B", "method_top_move": "A", "method_visits": 1024, "triggered": True},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "B", "method_visits": 1024, "triggered": True},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "A", "method_visits": 256, "triggered": False},
        ]

        summary = self.mod.summarize_method(rows)

        self.assertEqual(summary["correction_count"], 1)
        self.assertEqual(summary["regression_count"], 1)
        self.assertEqual(summary["net_correction_gain"], 0)
        self.assertAlmostEqual(summary["trigger_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
