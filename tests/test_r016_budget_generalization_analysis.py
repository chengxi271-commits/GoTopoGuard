import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r016_budget_generalization_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r016_budget_generalization_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R016BudgetGeneralizationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_budget_record_computes_reference_agreement(self):
        row = {
            "position_id": "P1",
            "dataset": "D5",
            "family": "cyclic",
            "v64_top_move": "A",
            "v256_top_move": "B",
            "v1024_top_move": "C",
            "v8192_top_move": "C",
            "v64_scoreLead": 0.0,
            "v256_scoreLead": 1.0,
            "v1024_scoreLead": 1.5,
            "v8192_scoreLead": 2.0,
            "v64_winrate": 0.1,
            "v256_winrate": 0.2,
            "v1024_winrate": 0.3,
            "v8192_winrate": 0.4,
        }

        out = self.mod.budget_stability_record(row)

        self.assertEqual(out["agree_1024_8192"], 1)
        self.assertEqual(out["agree_64_8192"], 0)
        self.assertAlmostEqual(out["score_abs_delta_256_8192"], 1.0)

    def test_summarize_records_groups_dataset_and_family(self):
        rows = [
            {"dataset": "D5", "family": "cyclic", "agree_1024_8192": 1, "disagree_1024_8192": 0, "score_abs_delta_1024_8192": 0.5, "winrate_abs_delta_1024_8192": 0.1},
            {"dataset": "D5", "family": "cyclic", "agree_1024_8192": 0, "disagree_1024_8192": 1, "score_abs_delta_1024_8192": 1.5, "winrate_abs_delta_1024_8192": 0.3},
        ]

        summary = self.mod.summarize_records(rows)
        dataset_row = next(row for row in summary if row["group"] == "dataset" and row["name"] == "D5")
        family_row = next(row for row in summary if row["group"] == "family" and row["name"] == "D5/cyclic")

        self.assertEqual(dataset_row["position_count"], 2)
        self.assertAlmostEqual(dataset_row["agreement_rate_1024_8192"], 0.5)
        self.assertEqual(family_row["position_count"], 2)


if __name__ == "__main__":
    unittest.main()
