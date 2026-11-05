import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r009_perturbation_split.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r009_perturbation_split", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R009PerturbationSplitTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_select_candidate_prefers_empty_point_away_from_stones(self):
        board = [
            "B....",
            ".....",
            ".....",
            ".....",
            "....W",
        ]

        candidate = self.mod.select_perturbation_candidate(board, "B")

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["occupied_neighbor_count"], 0)
        self.assertGreaterEqual(candidate["nearest_stone_distance"], 3)

    def test_apply_single_stone_perturbation_rejects_occupied(self):
        board = [
            "B....",
            ".....",
            ".....",
            ".....",
            "....W",
        ]

        changed = self.mod.apply_single_stone_perturbation(board, "W", 1, 1)

        self.assertEqual(changed[1][1], "W")
        with self.assertRaises(ValueError):
            self.mod.apply_single_stone_perturbation(board, "W", 0, 0)

    def test_build_perturbation_pairs_keeps_original_and_perturbed_signatures(self):
        source = {
            "position_id": "P1",
            "split_id": "D1",
            "source_id": "unit",
            "source_name": "unit.sgf",
            "source_url": "unit://",
            "komi": "6.5",
            "rules": "Chinese",
            "move_prefix_len": 42,
            "game_move_count": 100,
            "board": [
                "B....",
                ".....",
                ".....",
                ".....",
                "....W",
            ],
        }

        records, summary = self.mod.build_perturbation_pairs([source], run_id="R009", split_id="D4_TEST")

        self.assertEqual(summary["input_position_count"], 1)
        self.assertEqual(summary["pair_count"], 1)
        self.assertNotEqual(records[0]["original_position_signature"], records[0]["perturbed_position_signature"])
        self.assertEqual(records[0]["perturbation_stone_count"], 1)
        self.assertEqual(records[0]["komi"], "6.5")
        self.assertEqual(records[0]["rules"], "Chinese")
        self.assertEqual(records[0]["source_move_prefix_len"], 42)


if __name__ == "__main__":
    unittest.main()
