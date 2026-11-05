import importlib.util
import pathlib
import types
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r001_sgf_replay_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r001_sgf_replay_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R001SgfReplaySmokeTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_parse_mainline_moves_and_pass(self):
        game = self.mod.parse_sgf_mainline("(;GM[1]FF[4]SZ[19]KM[6.5]RU[Chinese];B[pd];W[dd];B[])")

        self.assertEqual(game.board_size, 19)
        self.assertEqual(game.komi, "6.5")
        self.assertEqual(game.rules, "Chinese")
        self.assertEqual([(move.color, move.coord) for move in game.moves], [("B", (15, 3)), ("W", (3, 3)), ("B", None)])

    def test_replay_captures_surrounded_stone(self):
        game = self.mod.parse_sgf_mainline("(;SZ[5];B[bb];W[ab];B[ee];W[ba];B[ed];W[bc];B[de];W[cb])")

        result = self.mod.replay_game(game)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.captures["W"], 1)
        self.assertEqual(result.final_board[1][1], ".")

    def test_replay_rejects_occupied_point(self):
        game = self.mod.parse_sgf_mainline("(;SZ[5];B[aa];W[aa])")

        result = self.mod.replay_game(game)

        self.assertFalse(result.ok)
        self.assertIn("occupied", result.errors[0])

    def test_replay_allows_suicide_when_rules_allow_it(self):
        game = self.mod.parse_sgf_mainline("(;SZ[3]RU[koPOSITIONALscoreAREAtaxNONEsui1]AB[ab][ba][bc][cb];W[bb])")

        result = self.mod.replay_game(game)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.final_board[1][1], ".")
        self.assertEqual(result.captures["B"], 1)

    def test_setup_stones_are_applied_before_moves(self):
        game = self.mod.parse_sgf_mainline("(;SZ[5]AB[aa][bb]AW[cc];B[dd])")

        result = self.mod.replay_game(game)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.final_board[0][0], "B")
        self.assertEqual(result.final_board[1][1], "B")
        self.assertEqual(result.final_board[2][2], "W")
        self.assertEqual(result.final_board[3][3], "B")

    def test_position_signature_is_stable_and_distinguishes_positions(self):
        game_a = self.mod.parse_sgf_mainline("(;SZ[5];B[aa];W[bb])")
        game_b = self.mod.parse_sgf_mainline("(;SZ[5];B[aa];W[cc])")

        result_a1 = self.mod.replay_game(game_a)
        result_a2 = self.mod.replay_game(game_a)
        result_b = self.mod.replay_game(game_b)

        self.assertEqual(self.mod.position_signature(result_a1.final_board), self.mod.position_signature(result_a2.final_board))
        self.assertNotEqual(self.mod.position_signature(result_a1.final_board), self.mod.position_signature(result_b.final_board))

    def test_extract_go_attack_site_dir_names(self):
        script = 'content:[{title:"A",dir_name:"no_search_hardened"},{title:"B",dir_name:"human_vs_kata100k"}]'

        self.assertEqual(self.mod.extract_go_attack_site_dir_names(script), ["no_search_hardened", "human_vs_kata100k"])

    def test_site_sgf_url_uses_section_basename(self):
        url = self.mod.go_attack_site_sgf_url("no_search_hardened", "public/sgfs/no_search_hardened/C240D2DE4B3BBBE6-L9.sgfs")

        self.assertEqual(url, "https://goattack.far.ai/sgfs/no_search_hardened/C240D2DE4B3BBBE6-L9.sgfs")

    def test_completion_failures_require_all_sources(self):
        args = types.SimpleNamespace(jgdb_limit=50, katago_limit=50, go_attack_limit=20)
        report = {
            "by_source": {
                "jgdb": {"valid_count": 50},
                "katago_kata1_2020_12_12": {"valid_count": 0},
                "go_attack_site_examples": {"valid_count": 20},
            }
        }

        failures = self.mod.completion_failures(report, args)

        self.assertEqual(failures, {"katago_kata1_2020_12_12": {"required": 50, "valid": 0}})


if __name__ == "__main__":
    unittest.main()
