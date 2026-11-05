import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r011_metamorphic_symmetry_features.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r011_metamorphic_symmetry_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R011MetamorphicSymmetryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_transform_board_and_inverse_move(self):
        board = [
            "B..",
            ".W.",
            "..B",
        ]

        transformed = self.mod.transform_board(board, "rot90")

        self.assertEqual(transformed, ["..B", ".W.", "B.."])
        self.assertEqual(self.mod.map_gtp_move("A3", 3, "rot90"), "C3")
        self.assertEqual(self.mod.canonicalize_transformed_gtp_move("C3", 3, "rot90"), "A3")

    def test_pass_move_is_stable_under_symmetry(self):
        self.assertEqual(self.mod.map_gtp_move("pass", 19, "flip_x"), "pass")
        self.assertEqual(self.mod.canonicalize_transformed_gtp_move("PASS", 19, "rot180"), "PASS")

    def test_aggregate_symmetry_features_detects_disagreement(self):
        base_rows = [
            {"position_id": "P", "visits": 256, "top_move": "D4", "winrate": 0.60, "scoreLead": 1.0},
        ]
        symmetry_rows = [
            {
                "position_id": "P",
                "symmetry": "rot180",
                "visits": 256,
                "top_move": "Q16",
                "winrate": 0.63,
                "scoreLead": 1.4,
            },
            {
                "position_id": "P",
                "symmetry": "flip_x",
                "visits": 256,
                "top_move": "Q4",
                "winrate": 0.55,
                "scoreLead": 0.4,
            },
        ]

        features = self.mod.aggregate_metamorphic_features(base_rows, symmetry_rows, board_size=19)[0]

        self.assertEqual(features["position_id"], "P")
        self.assertEqual(features["sym_v256_valid_transform_count"], 2)
        self.assertEqual(features["sym_v256_top_move_disagreement_count"], 0)
        self.assertAlmostEqual(features["sym_v256_winrate_abs_delta_mean"], 0.04)
        self.assertAlmostEqual(features["sym_v256_score_abs_delta_max"], 0.6)

    def test_select_shard_records_is_deterministic_modulo_split(self):
        records = [{"position_id": f"P{i}"} for i in range(6)]

        shard = self.mod.select_shard_records(records, shard_index=1, shard_count=3)

        self.assertEqual([row["position_id"] for row in shard], ["P1", "P4"])


if __name__ == "__main__":
    unittest.main()
