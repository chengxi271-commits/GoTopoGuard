import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r011_detection_baseline_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r011_detection_baseline_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R011DetectionBaselineSweepTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_fpr_at_recall_threshold(self):
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.6, 0.7, 0.1]

        fpr = self.mod.fpr_at_recall(y_true, scores, 0.9)

        self.assertEqual(fpr, 0.5)

    def test_fpr_at_recall_handles_tied_scores_by_threshold_group(self):
        y_true = [1, 0, 1, 0]
        scores = [0.5, 0.5, 0.5, 0.5]

        fpr = self.mod.fpr_at_recall(y_true, scores, 0.9)

        self.assertEqual(fpr, 1.0)

    def test_feature_row_from_board_adds_label_and_family(self):
        row = self.mod.feature_row_from_board(
            position_id="P",
            board_rows=[
                "B....",
                ".....",
                "..W..",
                ".....",
                ".....",
            ],
            label=1,
            dataset="D5",
            family="cyclic_enclosure",
        )

        self.assertEqual(row["label"], 1)
        self.assertEqual(row["family"], "cyclic_enclosure")
        self.assertIn("black_group_count", row)
        self.assertIn("stone_count", row)

    def test_family_heldout_split_excludes_heldout_from_train(self):
        rows = [
            {"position_id": "n1", "label": 0, "family": "natural", "dataset": "D1"},
            {"position_id": "n2", "label": 0, "family": "natural", "dataset": "D2"},
            {"position_id": "p1", "label": 1, "family": "A", "dataset": "D5"},
            {"position_id": "p2", "label": 1, "family": "B", "dataset": "D5"},
        ]

        train, test = self.mod.family_heldout_split(rows, "A")

        self.assertNotIn("p1", {row["position_id"] for row in train})
        self.assertIn("p1", {row["position_id"] for row in test})
        self.assertIn("p2", {row["position_id"] for row in train})

    def test_attach_search_features_merges_by_position_id(self):
        rows = [{"position_id": "P1", "label": 1}, {"position_id": "P2", "label": 0}]
        search = {"P1": {"position_id": "P1", "v256_winrate": 0.7}}

        merged, count = self.mod.attach_search_features(rows, search)

        self.assertEqual(count, 1)
        self.assertEqual(merged[0]["v256_winrate"], 0.7)
        self.assertNotIn("v256_winrate", merged[1])

    def test_attach_metamorphic_features_merges_by_position_id(self):
        rows = [{"position_id": "P1", "label": 1}, {"position_id": "P2", "label": 0}]
        metamorphic = {"P2": {"position_id": "P2", "sym_v256_top_move_disagreement_rate": 0.5}}

        merged, count = self.mod.attach_metamorphic_features(rows, metamorphic)

        self.assertEqual(count, 1)
        self.assertEqual(merged[1]["sym_v256_top_move_disagreement_rate"], 0.5)
        self.assertNotIn("sym_v256_top_move_disagreement_rate", merged[0])

    def test_b5_runs_when_search_and_metamorphic_are_available(self):
        rows = []
        for index in range(6):
            rows.append(
                {
                    "position_id": f"n{index}",
                    "label": 0,
                    "family": "natural",
                    "dataset": "D1",
                    **{name: float(index) for name in self.mod.BOARD_FEATURES + self.mod.TOPOLOGY_FEATURES},
                    **{name: float(index) for name in self.mod.ENGINE_CONFIDENCE_FEATURES + self.mod.SEARCH_BRITTLENESS_FEATURES},
                    **{name: float(index) for name in self.mod.METAMORPHIC_FEATURES},
                }
            )
        for index in range(6):
            rows.append(
                {
                    "position_id": f"p{index}",
                    "label": 1,
                    "family": "A",
                    "dataset": "D5",
                    **{name: float(index + 10) for name in self.mod.BOARD_FEATURES + self.mod.TOPOLOGY_FEATURES},
                    **{name: float(index + 10) for name in self.mod.ENGINE_CONFIDENCE_FEATURES + self.mod.SEARCH_BRITTLENESS_FEATURES},
                    **{name: float(index + 10) for name in self.mod.METAMORPHIC_FEATURES},
                }
            )
        for index in range(6):
            rows.append(
                {
                    "position_id": f"q{index}",
                    "label": 1,
                    "family": "B",
                    "dataset": "D5",
                    **{name: float(index + 20) for name in self.mod.BOARD_FEATURES + self.mod.TOPOLOGY_FEATURES},
                    **{name: float(index + 20) for name in self.mod.ENGINE_CONFIDENCE_FEATURES + self.mod.SEARCH_BRITTLENESS_FEATURES},
                    **{name: float(index + 20) for name in self.mod.METAMORPHIC_FEATURES},
                }
            )

        metrics = self.mod.run_family_heldout(rows, include_search_baselines=True, include_metamorphic_baselines=True)

        baselines = {row["baseline"] for row in metrics}
        self.assertIn("B5_topology_plus_search_plus_metamorphic_logreg", baselines)
        self.assertIn("B5_topology_plus_search_plus_metamorphic_gbdt", baselines)


if __name__ == "__main__":
    unittest.main()
