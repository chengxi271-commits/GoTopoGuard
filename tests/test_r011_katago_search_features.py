import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r011_katago_search_features.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r011_katago_search_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R011KataGoSearchFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_initial_player_uses_prefix_parity(self):
        self.assertEqual(self.mod.initial_player_for_record({"move_prefix_len": 10})[0], "B")
        self.assertEqual(self.mod.initial_player_for_record({"move_prefix_len": 11})[0], "W")
        self.assertEqual(self.mod.initial_player_for_record({"source_move_prefix_len": 11})[0], "W")

    def test_build_query_from_board_uses_initial_stones_and_komi(self):
        record = {
            "position_id": "P",
            "board": ["B.", ".W"],
            "komi": "6.5",
            "move_prefix_len": 2,
        }

        query, metadata = self.mod.build_query_from_board_record(record, board_key="board", visits=64)

        self.assertEqual(query["initialPlayer"], "B")
        self.assertEqual(query["initialStones"], [["B", "A2"], ["W", "B1"]])
        self.assertEqual(query["komi"], 6.5)
        self.assertEqual(metadata["initial_player_basis"], "move_prefix_len_parity")

    def test_extract_query_features_from_response(self):
        response = {
            "id": "P_v64",
            "rootInfo": {"winrate": 0.6, "scoreLead": 3.0, "scoreStdev": 5.0},
            "moveInfos": [
                {"move": "D4", "visits": 30, "prior": 0.2},
                {"move": "Q16", "visits": 10, "prior": 0.1},
            ],
            "policy": [0.5, 0.5],
        }

        features = self.mod.extract_query_features(response, visits=64)

        self.assertEqual(features["top_move"], "D4")
        self.assertEqual(features["top_visit_count"], 30)
        self.assertAlmostEqual(features["visit_entropy"], 0.811278, places=5)

    def test_aggregate_position_features_marks_budget_disagreement(self):
        rows = [
            {"position_id": "P", "visits": 64, "top_move": "D4", "winrate": 0.4, "scoreLead": 1.0},
            {"position_id": "P", "visits": 256, "top_move": "Q16", "winrate": 0.7, "scoreLead": 4.0},
        ]

        agg = self.mod.aggregate_position_features(rows)[0]

        self.assertEqual(agg["budget_top_move_disagreement_64_256"], 1)
        self.assertAlmostEqual(agg["budget_winrate_abs_delta_64_256"], 0.3)


if __name__ == "__main__":
    unittest.main()
