import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r014_guard_equal_compute_pilot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r014_guard_equal_compute_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R014GuardEqualComputePilotTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_guard_top_move_uses_high_budget_only_when_triggered(self):
        row = {"v256_top_move": "D4", "v1024_top_move": "Q16"}

        self.assertEqual(self.mod.guard_top_move(row, triggered=False), "D4")
        self.assertEqual(self.mod.guard_top_move(row, triggered=True), "Q16")

    def test_summarize_method_counts_corrections_and_regressions(self):
        rows = [
            {"oracle_top_move": "A", "g0_top_move": "B", "method_top_move": "A", "method_visits": 1024},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "B", "method_visits": 1024},
            {"oracle_top_move": "A", "g0_top_move": "A", "method_top_move": "A", "method_visits": 256},
        ]

        summary = self.mod.summarize_method(rows)

        self.assertEqual(summary["position_count"], 3)
        self.assertEqual(summary["g0_error_count"], 1)
        self.assertEqual(summary["correction_count"], 1)
        self.assertEqual(summary["regression_count"], 1)
        self.assertAlmostEqual(summary["oracle_agreement_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
