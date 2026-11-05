import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r013_operating_point_tradeoff.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r013_operating_point_tradeoff", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R013OperatingPointTradeoffTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_threshold_at_fpr_cap_allows_grouped_threshold(self):
        threshold = self.mod.threshold_at_fpr_cap([0.9, 0.7, 0.7, 0.1], 0.75)

        self.assertEqual(threshold, 0.7)

    def test_threshold_at_fpr_cap_can_choose_no_trigger_threshold(self):
        threshold = self.mod.threshold_at_fpr_cap([0.9, 0.8], 0.0)

        self.assertGreater(threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
