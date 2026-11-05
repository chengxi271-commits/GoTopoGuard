import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r013_natural_false_positive_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r013_natural_false_positive_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R013NaturalFalsePositiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_summarize_false_positive_group_counts_triggers(self):
        rows = [
            {"score": 0.9},
            {"score": 0.2},
            {"score": 0.8},
        ]

        summary = self.mod.summarize_false_positive_group(rows, threshold=0.75)

        self.assertEqual(summary["position_count"], 3)
        self.assertEqual(summary["trigger_count"], 2)
        self.assertAlmostEqual(summary["trigger_rate"], 2 / 3)

    def test_natural_fold_is_stable_in_range(self):
        fold = self.mod.natural_fold("P", fold_count=5)

        self.assertGreaterEqual(fold, 0)
        self.assertLess(fold, 5)


if __name__ == "__main__":
    unittest.main()
