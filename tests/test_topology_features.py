# Copyright 2026 Yixi Chen
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gotopoguard import (  # noqa: E402
    empty_regions,
    extract_topology_features,
    parse_board,
    stone_articulation_points,
    stone_groups,
)


class TopologyFeatureTests(unittest.TestCase):
    def test_stone_groups_count_size_and_liberties(self):
        board = parse_board(
            [
                ".....",
                ".BB..",
                ".B.W.",
                "....W",
                ".....",
            ]
        )
        black_groups = stone_groups(board, "B")
        white_groups = stone_groups(board, "W")
        self.assertEqual(len(black_groups), 1)
        self.assertEqual(black_groups[0]["size"], 3)
        self.assertEqual(black_groups[0]["liberty_count"], 7)
        self.assertEqual(len(white_groups), 2)

    def test_empty_regions_detect_enclosed_region(self):
        board = parse_board(
            [
                ".....",
                ".BBB.",
                ".B.B.",
                ".BBB.",
                ".....",
            ]
        )
        enclosed = [r for r in empty_regions(board) if not r["edge_reachable"]]
        self.assertEqual(len(enclosed), 1)
        self.assertEqual(enclosed[0]["size"], 1)
        self.assertEqual(enclosed[0]["boundary_colors"], ["B"])

    def test_articulation_points_detect_plus_center(self):
        board = parse_board(
            [
                ".....",
                "..B..",
                ".BBB.",
                "..B..",
                ".....",
            ]
        )
        self.assertEqual(stone_articulation_points(board, "B"), [(2, 2)])

    def test_extract_topology_features(self):
        board = parse_board(
            [
                ".....",
                ".BBB.",
                ".B.B.",
                ".BBB.",
                ".....",
            ]
        )
        features = extract_topology_features(board)
        self.assertEqual(features["black_group_count"], 1)
        self.assertEqual(features["white_group_count"], 0)
        self.assertEqual(features["enclosed_empty_region_count"], 1)
        self.assertEqual(features["black_articulation_count"], 0)


if __name__ == "__main__":
    unittest.main()
