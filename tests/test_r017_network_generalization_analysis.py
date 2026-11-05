import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r017_network_generalization_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r017_network_generalization_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R017NetworkGeneralizationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_network_record_compares_to_base_budget(self):
        base = {"position_id": "P1", "v1024_top_move": "A", "v1024_scoreLead": 1.0, "v1024_winrate": 0.6}
        alt = {"position_id": "P1", "network_alias": "b18", "top_move": "B", "scoreLead": 2.5, "winrate": 0.7, "dataset": "D5", "family": "cyclic"}

        out = self.mod.network_stability_record(base, alt)

        self.assertEqual(out["top_move_agrees_with_b28_1024"], 0)
        self.assertAlmostEqual(out["score_abs_delta_vs_b28_1024"], 1.5)
        self.assertEqual(out["network_alias"], "b18")

    def test_summarize_records_by_network_and_dataset(self):
        rows = [
            {"network_alias": "b18", "dataset": "D5", "family": "cyclic", "top_move_agrees_with_b28_1024": 1, "score_abs_delta_vs_b28_1024": 0.1, "winrate_abs_delta_vs_b28_1024": 0.01},
            {"network_alias": "b18", "dataset": "D5", "family": "cyclic", "top_move_agrees_with_b28_1024": 0, "score_abs_delta_vs_b28_1024": 0.3, "winrate_abs_delta_vs_b28_1024": 0.03},
        ]

        summary = self.mod.summarize_records(rows)
        row = next(item for item in summary if item["group"] == "network_dataset")

        self.assertEqual(row["name"], "b18/D5")
        self.assertAlmostEqual(row["top_move_agreement_rate_vs_b28_1024"], 0.5)


if __name__ == "__main__":
    unittest.main()
