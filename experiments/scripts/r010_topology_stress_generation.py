#!/usr/bin/env python
"""Build R010 controlled topology stress positions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from collections import Counter


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPO = load_module("topology_features", ROOT_DIR / "src" / "gotopobench" / "topology_features.py")
R001 = load_module("r001_sgf_replay_smoke", SCRIPT_DIR / "r001_sgf_replay_smoke.py")

BOARD_SIZE = 19
FAMILIES = [
    "cyclic_enclosure",
    "cut_connect_corridor",
    "ladder_net_escape",
    "boundary_escape_route",
    "false_eye_enclosure",
    "shared_liberty_race",
    "pass_scoring_instability",
]


def ring(color: str, width: int, height: int) -> list[tuple[str, int, int]]:
    stones = []
    for y in range(height):
        for x in range(width):
            if x in (0, width - 1) or y in (0, height - 1):
                stones.append((color, x, y))
    return stones


PATTERNS = {
    "cyclic_enclosure": {
        "width": 5,
        "height": 5,
        "stones": ring("B", 5, 5),
        "proxy": "closed same-color ring around enclosed empty region",
    },
    "cut_connect_corridor": {
        "width": 5,
        "height": 5,
        "stones": [("B", 2, 0), ("B", 2, 1), ("B", 2, 2), ("B", 1, 2), ("B", 3, 2), ("B", 2, 3), ("B", 2, 4)],
        "proxy": "single articulation connector in same-color group",
    },
    "ladder_net_escape": {
        "width": 6,
        "height": 5,
        "stones": [
            ("B", 2, 2),
            ("B", 3, 2),
            ("W", 1, 2),
            ("W", 4, 2),
            ("W", 2, 1),
            ("W", 3, 1),
            ("B", 4, 3),
            ("W", 5, 3),
        ],
        "proxy": "low-liberty group with diagonal escape pressure",
    },
    "boundary_escape_route": {
        "width": 5,
        "height": 5,
        "stones": [("B", 2, 0), ("B", 2, 1), ("W", 1, 1), ("W", 3, 1), ("W", 1, 2), ("W", 3, 2)],
        "proxy": "low-liberty group touching board boundary",
    },
    "false_eye_enclosure": {
        "width": 5,
        "height": 5,
        "stones": ring("B", 5, 5) + [("B", 2, 1), ("B", 2, 2), ("B", 2, 3)],
        "proxy": "enclosed same-color eye-like regions split by a connector",
    },
    "shared_liberty_race": {
        "width": 5,
        "height": 5,
        "stones": [
            ("B", 1, 2),
            ("B", 1, 3),
            ("W", 3, 2),
            ("W", 3, 3),
            ("W", 0, 2),
            ("W", 0, 3),
            ("B", 4, 2),
            ("B", 4, 3),
        ],
        "proxy": "opposing groups with shared liberties",
    },
    "pass_scoring_instability": {
        "width": 7,
        "height": 3,
        "stones": ring("B", 3, 3) + [("W", x + 4, y) for color, x, y in ring("W", 3, 3)],
        "proxy": "adjacent enclosed territories plus dame-like corridor",
    },
}


def transform_point(x: int, y: int, width: int, height: int, variant: int) -> tuple[int, int]:
    if variant % 2:
        x = width - 1 - x
    if (variant // 2) % 2:
        y = height - 1 - y
    return x, y


def transformed_stones(family: str, seed: int) -> tuple[list[tuple[str, int, int]], int, int]:
    pattern = PATTERNS[family]
    width = pattern["width"]
    height = pattern["height"]
    max_x = BOARD_SIZE - width
    max_y = BOARD_SIZE - height
    ox = (seed * 5 + seed // 7) % (max_x + 1)
    oy = (seed * 3 + seed // 11) % (max_y + 1)
    swap_colors = (seed // ((max_x + 1) * (max_y + 1))) % 2 == 1
    variant = seed // 13
    stones = []
    for color, x, y in pattern["stones"]:
        tx, ty = transform_point(x, y, width, height, variant)
        final_color = color
        if swap_colors:
            final_color = "W" if color == "B" else "B"
        stones.append((final_color, ox + tx, oy + ty))
    return stones, ox, oy


def board_from_stones(stones: list[tuple[str, int, int]]) -> list[str]:
    board = [["." for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for color, col, row in stones:
        if board[row][col] != ".":
            raise ValueError(f"duplicate generated point {(col, row)}")
        board[row][col] = color
    return ["".join(row) for row in board]


def legal_board(board: list[list[str]]) -> bool:
    for color in ("B", "W"):
        for group in TOPO.stone_groups(board, color):
            if group["liberty_count"] == 0:
                return False
    return True


def shared_liberty_count(board: list[list[str]]) -> int:
    black_liberties = set()
    white_liberties = set()
    for group in TOPO.stone_groups(board, "B"):
        black_liberties.update(group["liberties"])
    for group in TOPO.stone_groups(board, "W"):
        white_liberties.update(group["liberties"])
    return len(black_liberties & white_liberties)


def min_group_liberty_count(board: list[list[str]]) -> int:
    counts = [group["liberty_count"] for color in ("B", "W") for group in TOPO.stone_groups(board, color)]
    return min(counts) if counts else 0


def boundary_low_liberty_group_exists(board: list[list[str]]) -> bool:
    width, height = TOPO.board_size(board)
    for color in ("B", "W"):
        for group in TOPO.stone_groups(board, color):
            min_boundary_distance = min(
                min(col, row, width - 1 - col, height - 1 - row)
                for col, row in group["stones"]
            )
            if min_boundary_distance <= 2 and group["liberty_count"] <= 4:
                return True
    return False


def family_face_valid(family: str, board: list[list[str]], features: dict) -> bool:
    if family == "cyclic_enclosure":
        return features["enclosed_empty_region_count"] >= 1
    if family == "cut_connect_corridor":
        return features["black_articulation_count"] + features["white_articulation_count"] >= 1
    if family == "ladder_net_escape":
        return min_group_liberty_count(board) <= 3
    if family == "boundary_escape_route":
        return boundary_low_liberty_group_exists(board)
    if family == "false_eye_enclosure":
        return features["black_bounded_empty_region_count"] + features["white_bounded_empty_region_count"] >= 2
    if family == "shared_liberty_race":
        return shared_liberty_count(board) >= 1
    if family == "pass_scoring_instability":
        return features["black_bounded_empty_region_count"] >= 1 and features["white_bounded_empty_region_count"] >= 1
    raise ValueError(f"unknown family {family}")


def generate_position(family: str, *, seed: int, run_id: str, split_id: str) -> dict:
    stones, offset_x, offset_y = transformed_stones(family, seed)
    board_rows = board_from_stones(stones)
    board = TOPO.parse_board(board_rows)
    features = TOPO.extract_topology_features(board)
    legal = legal_board(board)
    face_valid = legal and family_face_valid(family, board, features)
    return {
        "position_id": f"{run_id}_{family}_{seed:04d}",
        "run_id": run_id,
        "split_id": split_id,
        "family": family,
        "family_proxy": PATTERNS[family]["proxy"],
        "seed": seed,
        "offset": {"col": offset_x, "row": offset_y},
        "board_size": BOARD_SIZE,
        "komi": "6.5",
        "rules": "generated_topology_stress_chinese_proxy",
        "board": board_rows,
        "position_signature": R001.position_signature(TOPO.parse_board(board_rows)),
        "features": features,
        "face_validity": {
            "legal_board": legal,
            "family_face_valid": face_valid,
            "face_validity_basis": "deterministic topology-feature proxy",
        },
        "evidence_class": "controlled topology stress construction only",
    }


def generate_stress_split(*, per_family_count: int, run_id: str, split_id: str) -> tuple[list[dict], dict]:
    records = []
    family_counts = Counter()
    invalid_counts = Counter()
    seen_signatures = set()
    duplicate_count = 0
    for family in FAMILIES:
        seed = 0
        attempts = 0
        while family_counts[family] < per_family_count and attempts < per_family_count * 50:
            record = generate_position(family, seed=seed, run_id=run_id, split_id=split_id)
            seed += 1
            attempts += 1
            signature = record["position_signature"]
            if signature in seen_signatures:
                duplicate_count += 1
                continue
            if not record["face_validity"]["family_face_valid"]:
                invalid_counts[family] += 1
                continue
            seen_signatures.add(signature)
            records.append(record)
            family_counts[family] += 1
    status = "done"
    if any(family_counts[family] != per_family_count for family in FAMILIES):
        status = "incomplete"
    summary = {
        "status": status,
        "per_family_target": per_family_count,
        "position_count": len(records),
        "family_counts": dict(family_counts),
        "invalid_face_validity_counts": dict(invalid_counts),
        "duplicate_signature_skipped": duplicate_count,
        "unique_signature_count": len(seen_signatures),
    }
    return records, summary


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    records, summary = generate_stress_split(
        per_family_count=args.per_family_count,
        run_id=args.run_id,
        split_id=args.split_id,
    )
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "controlled topology stress construction only",
        "split_id": args.split_id,
        "families": FAMILIES,
        **summary,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "positions.jsonl", records)
    write_json(output_root / "split_report.json", report)
    write_json(
        output_root / "source_manifest.json",
        {
            "run_id": args.run_id,
            "split_id": args.split_id,
            "source_id": "generated_topology_templates",
            "provenance_state": "generated-controlled-stress",
            "families": FAMILIES,
            "per_family_target": args.per_family_count,
            "template_proxies": {family: PATTERNS[family]["proxy"] for family in FAMILIES},
            "evidence_class": "controlled topology stress construction only",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_issue and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R010")
    parser.add_argument("--split-id", default="D5_TOPOLOGY_STRESS_TEMPLATES")
    parser.add_argument("--per-family-count", type=int, default=144)
    parser.add_argument("--output-root", default="experiments/results/R010_TOPOLOGY_STRESS_SPLIT")
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
