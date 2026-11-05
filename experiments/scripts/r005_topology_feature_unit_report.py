#!/usr/bin/env python
"""R005 controlled-board topology feature report."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src" / "gotopobench" / "topology_features.py"


def load_topology_module():
    spec = importlib.util.spec_from_file_location("topology_features", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    topo = load_topology_module()
    cases = {
        "enclosed_empty_region": [
            ".....",
            ".BBB.",
            ".B.B.",
            ".BBB.",
            ".....",
        ],
        "plus_articulation": [
            ".....",
            "..B..",
            ".BBB.",
            "..B..",
            ".....",
        ],
        "mixed_groups": [
            ".....",
            ".BB..",
            ".B.W.",
            "....W",
            ".....",
        ],
    }
    snapshots = []
    for case_id, rows in cases.items():
        board = topo.parse_board(rows)
        snapshots.append(
            {
                "case_id": case_id,
                "board": rows,
                "features": topo.extract_topology_features(board),
                "black_groups": topo.stone_groups(board, "B"),
                "white_groups": topo.stone_groups(board, "W"),
                "empty_regions": topo.empty_regions(board),
                "black_articulations": topo.stone_articulation_points(board, "B"),
                "white_articulations": topo.stone_articulation_points(board, "W"),
            }
        )
    report = {
        "run_id": "R005",
        "generated_at_unix": int(time.time()),
        "evidence_class": "controlled feature unit tests only",
        "status": "done",
        "test_command": "python -m unittest tests.test_r005_topology_features -v",
        "joint_test_command": "python -m unittest tests.test_r001_sgf_replay_smoke tests.test_r002_position_signature_validation tests.test_r003_katago_schema_smoke tests.test_r004_katago_100_position_smoke tests.test_r005_topology_features -v",
        "case_count": len(snapshots),
        "snapshots": snapshots,
        "boundary": "Controlled deterministic feature tests only; not detector or benchmark evidence.",
    }
    output_root = ROOT / "experiments" / "results" / "R005_TOPOLOGY_FEATURE_UNIT_TESTS"
    write_json(output_root / "r005_topology_feature_unit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
