import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r006_r007_natural_split_build.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r006_r007_natural_split_build", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R006R007NaturalSplitBuildTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_window_target_counts_are_balanced(self):
        targets = self.mod.window_target_counts(5, ["early", "mid", "late"])

        self.assertEqual(targets, {"early": 2, "mid": 2, "late": 1})

    def test_candidate_prefixes_use_window_centers(self):
        prefixes = self.mod.candidate_prefixes_for_game(200)

        self.assertEqual(prefixes, [("early_40_80", 60), ("mid_80_140", 110), ("late_140_220", 180)])

    def test_prefix_signature_uses_prefix_not_full_game(self):
        game = self.mod.R001.parse_sgf_mainline("(;SZ[5];B[aa];W[bb];B[cc])")

        sig1, board1 = self.mod.prefix_position_signature(game, 1)
        sig3, board3 = self.mod.prefix_position_signature(game, 3)

        self.assertNotEqual(sig1, sig3)
        self.assertEqual(board1[0][0], "B")
        self.assertEqual(board3[2][2], "B")

    def test_katago_default_sources_include_multiple_public_archives(self):
        urls = self.mod.katago_source_urls(None)

        self.assertGreaterEqual(len(urls), 2)
        self.assertIn(self.mod.R001.DEFAULT_KATAGO_URL, urls)

    def test_katago_source_id_uses_archive_date(self):
        source_id = self.mod.katago_source_id(
            "https://katagoarchive.org/kata1/traininggames/2021-03-15sgfs.tar.bz2"
        )

        self.assertEqual(source_id, "katago_kata1_2021_03_15")


if __name__ == "__main__":
    unittest.main()
