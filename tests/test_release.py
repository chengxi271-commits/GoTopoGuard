# Copyright 2026 Yixi Chen
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gotopoguard import extract_topology_features, parse_board, stone_groups


class ReleaseTests(unittest.TestCase):
    def test_invalid_board_encoding(self):
        for rows in ([], [""], ["..", "."], ["BX"]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_board(rows)

    def test_empty_standard_board(self):
        result = extract_topology_features(parse_board(["." * 19] * 19))
        self.assertEqual(len(result), 14)
        self.assertEqual(result["board_width"], 19)
        self.assertEqual(result["board_height"], 19)
        self.assertEqual(result["empty_region_count"], 1)
        self.assertTrue(all(value == 0 for key, value in result.items()
                            if key not in {"board_width", "board_height", "empty_region_count"}))

    def test_shared_liberty_is_counted_once(self):
        group = stone_groups(parse_board(["BB", "B."]), "B")[0]
        self.assertEqual(group["liberties"], [(1, 1)])
        self.assertEqual(group["liberty_count"], 1)

    def test_edge_region_is_not_enclosed(self):
        result = extract_topology_features(parse_board(["B.B", "BBB", "..."]))
        self.assertEqual(result["empty_region_count"], 2)
        self.assertEqual(result["enclosed_empty_region_count"], 0)

    def test_mixed_boundary_is_not_single_color_bounded(self):
        result = extract_topology_features(parse_board(["BBB", "B.W", "WWW"]))
        self.assertEqual(result["enclosed_empty_region_count"], 1)
        self.assertEqual(result["black_bounded_empty_region_count"], 0)
        self.assertEqual(result["white_bounded_empty_region_count"], 0)

    def test_reflection_invariance(self):
        rows = ["B...W", "BB.WW", "..B..", "....."]
        expected = extract_topology_features(parse_board(rows))
        for reflected in (rows[::-1], [row[::-1] for row in rows]):
            self.assertEqual(extract_topology_features(parse_board(reflected)), expected)

    def test_color_exchange(self):
        rows = [".....", ".BBB.", ".B.B.", ".BBB.", "....W"]
        original = extract_topology_features(parse_board(rows))
        swapped = extract_topology_features(parse_board([row.translate(str.maketrans("BW", "WB")) for row in rows]))
        for key, value in original.items():
            target = key.replace("black_", "white_") if key.startswith("black_") else key.replace("white_", "black_")
            self.assertEqual(swapped[target], value)

    def test_example_matches_expected_json(self):
        expected = json.loads((ROOT / "examples/expected_features.json").read_text())
        actual = subprocess.check_output([sys.executable, str(ROOT / "examples/minimal_features.py")], text=True)
        self.assertEqual(json.loads(actual), expected)

    def test_module_cli_matches_expected_json(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        actual = subprocess.check_output([sys.executable, "-m", "gotopoguard", str(ROOT / "examples/board.json")], env=env, text=True)
        expected = json.loads((ROOT / "examples/expected_features.json").read_text())
        self.assertEqual(json.loads(actual), expected)

    def test_module_cli_reports_invalid_input(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"board": "not an array"}')
            result = subprocess.run([sys.executable, "-m", "gotopoguard", str(path)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected a JSON array", result.stderr)


if __name__ == "__main__":
    unittest.main()
