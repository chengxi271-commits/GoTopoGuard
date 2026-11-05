# Copyright 2026 Yixi Chen
# SPDX-License-Identifier: Apache-2.0

"""Extract reference topology features from a JSON array of board rows."""

import argparse
import json
import sys
from pathlib import Path

from . import extract_topology_features, parse_board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path, help="JSON file containing an array of B/W/. row strings")
    args = parser.parse_args()
    try:
        rows = json.loads(args.board.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError("expected a JSON array of board-row strings")
        board = parse_board(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"board": rows, "features": extract_topology_features(board)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
