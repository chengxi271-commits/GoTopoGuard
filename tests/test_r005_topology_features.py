import importlib.util
import pathlib
import unittest


MODULE = pathlib.Path(__file__).resolve().parents[1] / "src" / "gotopobench" / "topology_features.py"


def load_module():
    spec = importlib.util.spec_from_file_location("topology_features", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R005TopologyFeatureTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_stone_groups_count_size_and_liberties(self):
        board = self.mod.parse_board(
            [
                ".....",
                ".BB..",
                ".B.W.",
                "....W",
                ".....",
            ]
        )

        black_groups = self.mod.stone_groups(board, "B")
        white_groups = self.mod.stone_groups(board, "W")

        self.assertEqual(len(black_groups), 1)
        self.assertEqual(black_groups[0]["size"], 3)
        self.assertEqual(black_groups[0]["liberty_count"], 7)
        self.assertEqual(len(white_groups), 2)

    def test_empty_regions_detect_enclosed_eye_like_region(self):
        board = self.mod.parse_board(
            [
                ".....",
                ".BBB.",
                ".B.B.",
                ".BBB.",
                ".....",
            ]
        )

        regions = self.mod.empty_regions(board)
        enclosed = [region for region in regions if not region["edge_reachable"]]

        self.assertEqual(len(enclosed), 1)
        self.assertEqual(enclosed[0]["size"], 1)
        self.assertEqual(enclosed[0]["boundary_colors"], ["B"])

    def test_articulation_points_detect_plus_center(self):
        board = self.mod.parse_board(
            [
                ".....",
                "..B..",
                ".BBB.",
                "..B..",
                ".....",
            ]
        )

        articulations = self.mod.stone_articulation_points(board, "B")

        self.assertEqual(articulations, [(2, 2)])

    def test_extract_topology_features_summarizes_controlled_board(self):
        board = self.mod.parse_board(
            [
                ".....",
                ".BBB.",
                ".B.B.",
                ".BBB.",
                ".....",
            ]
        )

        features = self.mod.extract_topology_features(board)

        self.assertEqual(features["black_group_count"], 1)
        self.assertEqual(features["white_group_count"], 0)
        self.assertEqual(features["enclosed_empty_region_count"], 1)
        self.assertEqual(features["black_articulation_count"], 0)


if __name__ == "__main__":
    unittest.main()
