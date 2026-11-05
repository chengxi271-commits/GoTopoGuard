import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r012_lead_time_evaluation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r012_lead_time_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R012LeadTimeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_threshold_at_recall_groups_ties(self):
        threshold = self.mod.threshold_at_recall([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.1], 0.9)

        self.assertEqual(threshold, 0.5)

    def test_warning_lead_uses_earliest_trigger(self):
        positions = [
            {"lead_moves_to_terminal": 80, "score": 0.2},
            {"lead_moves_to_terminal": 20, "score": 0.8},
            {"lead_moves_to_terminal": 5, "score": 0.9},
        ]

        result = self.mod.summarize_game_warning(positions, threshold=0.7)

        self.assertTrue(result["warning_detected"])
        self.assertEqual(result["earliest_warning_lead_moves"], 20)
        self.assertEqual(result["triggered_position_count"], 2)

    def test_select_best_detector_prefers_mean_auroc_then_auprc(self):
        metrics = [
            {"baseline": "A", "status": "done", "auroc": 0.8, "auprc": 0.7, "fpr_at_recall_90": 0.2},
            {"baseline": "A", "status": "done", "auroc": 0.8, "auprc": 0.7, "fpr_at_recall_90": 0.2},
            {"baseline": "B", "status": "done", "auroc": 0.8, "auprc": 0.75, "fpr_at_recall_90": 0.3},
            {"baseline": "B", "status": "done", "auroc": 0.8, "auprc": 0.75, "fpr_at_recall_90": 0.3},
        ]

        selected, summary = self.mod.select_best_detector(metrics)

        self.assertEqual(selected, "B")
        self.assertEqual(summary["B"]["metric_count"], 2)


if __name__ == "__main__":
    unittest.main()
