# Copyright 2026 Yixi Chen
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gotopoguard import extract_topology_features, parse_board  # noqa: E402


BOARD = [
    ".....",
    ".BBB.",
    ".B.B.",
    ".BBB.",
    ".....",
]


def main() -> int:
    payload = {
        "board": BOARD,
        "features": extract_topology_features(parse_board(BOARD)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
