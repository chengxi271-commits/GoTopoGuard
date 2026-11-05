import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r008_adversarial_replay_split.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r008_adversarial_replay_split", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R008AdversarialReplaySplitTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_lead_time_prefixes_skip_unavailable_early_leads(self):
        prefixes = self.mod.lead_time_prefixes(12, [80, 10, 1, 0])

        self.assertEqual(prefixes, [(10, 2), (1, 11), (0, 12)])

    def test_failure_window_bucket_labels_terminal_and_leads(self):
        self.assertEqual(self.mod.failure_window_bucket(0), "terminal")
        self.assertEqual(self.mod.failure_window_bucket(5), "near_1_5")
        self.assertEqual(self.mod.failure_window_bucket(20), "mid_6_20")
        self.assertEqual(self.mod.failure_window_bucket(40), "early_21_40")
        self.assertEqual(self.mod.failure_window_bucket(80), "early_41_80")

    def test_build_records_from_sgf_replays_prefix_positions(self):
        sgf = "(;SZ[5]KM[6.5];B[aa];W[bb];B[cc];W[dd])"
        records, summary = self.mod.build_adversarial_records(
            [("case/row1/example.sgf", sgf)],
            lead_moves=[2, 0],
            run_id="R008",
            split_id="D3_TEST",
            required_board_size=5,
        )

        self.assertEqual(summary["valid_game_count"], 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["lead_moves_to_terminal"], 2)
        self.assertEqual(records[0]["move_prefix_len"], 2)
        self.assertEqual(records[1]["lead_moves_to_terminal"], 0)
        self.assertEqual(records[1]["move_prefix_len"], 4)
        self.assertNotEqual(records[0]["position_signature"], records[1]["position_signature"])


if __name__ == "__main__":
    unittest.main()
