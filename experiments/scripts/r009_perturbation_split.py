#!/usr/bin/env python
"""Build R009 meaningless-stone perturbation split from D1/D2 positions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import time
from collections import Counter


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT_PATHS = [
    "experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl",
    "experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl",
]


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R001 = load_script_module("r001_sgf_replay_smoke", SCRIPT_DIR / "r001_sgf_replay_smoke.py")
R004 = load_script_module("r004_katago_100_position_smoke", SCRIPT_DIR / "r004_katago_100_position_smoke.py")


def neighbors(col: int, row: int, size: int):
    if col > 0:
        yield col - 1, row
    if col + 1 < size:
        yield col + 1, row
    if row > 0:
        yield col, row - 1
    if row + 1 < size:
        yield col, row + 1


def board_to_matrix(board_rows: list[str]) -> list[list[str]]:
    return [list(row) for row in board_rows]


def board_to_rows(board: list[list[str]]) -> list[str]:
    return ["".join(row) for row in board]


def group_and_liberties(board: list[list[str]], col: int, row: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    color = board[row][col]
    group = {(col, row)}
    liberties = set()
    stack = [(col, row)]
    while stack:
        c, r = stack.pop()
        for nc, nr in neighbors(c, r, len(board)):
            stone = board[nr][nc]
            if stone == ".":
                liberties.add((nc, nr))
            elif stone == color and (nc, nr) not in group:
                group.add((nc, nr))
                stack.append((nc, nr))
    return group, liberties


def opponent(color: str) -> str:
    return "W" if color == "B" else "B"


def apply_single_stone_perturbation(board_rows: list[str], color: str, col: int, row: int) -> list[str]:
    board = board_to_matrix(board_rows)
    if board[row][col] != ".":
        raise ValueError("perturbation point is occupied")
    board[row][col] = color
    captured = []
    for nc, nr in neighbors(col, row, len(board)):
        if board[nr][nc] != opponent(color):
            continue
        group, liberties = group_and_liberties(board, nc, nr)
        if not liberties:
            captured.extend(group)
    for c, r in captured:
        board[r][c] = "."
    own_group, own_liberties = group_and_liberties(board, col, row)
    if not own_liberties:
        raise ValueError("perturbation would be suicide")
    return board_to_rows(board)


def nearest_stone_distance(board_rows: list[str], col: int, row: int) -> int:
    distances = [
        abs(col - c) + abs(row - r)
        for r, board_row in enumerate(board_rows)
        for c, stone in enumerate(board_row)
        if stone in ("B", "W")
    ]
    if not distances:
        return len(board_rows) * 2
    return min(distances)


def capture_count(before_rows: list[str], after_rows: list[str]) -> int:
    before = sum(1 for row in before_rows for stone in row if stone in ("B", "W"))
    after = sum(1 for row in after_rows for stone in row if stone in ("B", "W"))
    return max(0, before + 1 - after)


def select_perturbation_candidate(board_rows: list[str], color: str) -> dict | None:
    candidates = []
    size = len(board_rows)
    for row in range(size):
        for col in range(size):
            if board_rows[row][col] != ".":
                continue
            occupied_neighbors = sum(1 for nc, nr in neighbors(col, row, size) if board_rows[nr][nc] in ("B", "W"))
            try:
                after_rows = apply_single_stone_perturbation(board_rows, color, col, row)
            except ValueError:
                continue
            captures = capture_count(board_rows, after_rows)
            nearest = nearest_stone_distance(board_rows, col, row)
            edge_distance = min(col, row, size - 1 - col, size - 1 - row)
            candidates.append(
                {
                    "col": col,
                    "row": row,
                    "color": color,
                    "occupied_neighbor_count": occupied_neighbors,
                    "capture_count": captures,
                    "nearest_stone_distance": nearest,
                    "edge_distance": edge_distance,
                    "score_tuple": (captures, occupied_neighbors, -nearest, -edge_distance, row, col),
                }
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["score_tuple"])


def preferred_color_for_position(position_id: str) -> str:
    digest = hashlib.sha1(position_id.encode("utf-8")).hexdigest()
    return "B" if int(digest[:2], 16) % 2 == 0 else "W"


def board_signature(board_rows: list[str]) -> str:
    return R001.position_signature(board_to_matrix(board_rows))


def build_perturbation_pairs(source_positions: list[dict], *, run_id: str, split_id: str) -> tuple[list[dict], dict]:
    records = []
    status_counts = Counter()
    source_split_counts = Counter()
    color_counts = Counter()
    low_impact_proxy_counts = Counter()
    for source in source_positions:
        original_id = source["position_id"]
        original_board = source["board"]
        source_split_counts[source.get("split_id", "unknown")] += 1
        colors = [preferred_color_for_position(original_id)]
        colors.append(opponent(colors[0]))
        candidate = None
        for color in colors:
            candidate = select_perturbation_candidate(original_board, color)
            if candidate is not None:
                break
        if candidate is None:
            status_counts["skipped_no_legal_candidate"] += 1
            continue
        perturbed_board = apply_single_stone_perturbation(
            original_board,
            candidate["color"],
            candidate["col"],
            candidate["row"],
        )
        original_signature = source.get("position_signature") or board_signature(original_board)
        perturbed_signature = board_signature(perturbed_board)
        if perturbed_signature == original_signature:
            status_counts["skipped_same_signature"] += 1
            continue
        color_counts[candidate["color"]] += 1
        proxy_label = "isolated_no_capture" if candidate["occupied_neighbor_count"] == 0 and candidate["capture_count"] == 0 else "legal_low_adjacency"
        low_impact_proxy_counts[proxy_label] += 1
        record_index = len(records) + 1
        records.append(
            {
                "pair_id": f"{run_id}_pair_{record_index:05d}",
                "run_id": run_id,
                "split_id": split_id,
                "source_position_id": original_id,
                "source_split_id": source.get("split_id"),
                "source_id": source.get("source_id"),
                "source_name": source.get("source_name"),
                "source_url": source.get("source_url"),
                "board_size": source.get("board_size", len(original_board)),
                "komi": source.get("komi"),
                "rules": source.get("rules"),
                "source_move_prefix_len": source.get("move_prefix_len"),
                "source_game_move_count": source.get("game_move_count"),
                "katago_query_compatible_komi": source.get("katago_query_compatible_komi"),
                "original_board": original_board,
                "perturbed_board": perturbed_board,
                "original_position_signature": original_signature,
                "perturbed_position_signature": perturbed_signature,
                "perturbation_stone_count": 1,
                "perturbation_color": candidate["color"],
                "perturbation_coord": {"col": candidate["col"], "row": candidate["row"]},
                "perturbation_gtp": R004.sgf_coord_to_gtp((candidate["col"], candidate["row"]), len(original_board)),
                "occupied_neighbor_count": candidate["occupied_neighbor_count"],
                "capture_count": candidate["capture_count"],
                "nearest_stone_distance": candidate["nearest_stone_distance"],
                "edge_distance": candidate["edge_distance"],
                "low_impact_proxy": proxy_label,
                "label_basis": "deterministic_low_adjacency_single_stone_perturbation",
                "evidence_class": "perturbation split construction only",
            }
        )
        status_counts["valid_pair"] += 1
    perturbed_signature_counts = Counter(record["perturbed_position_signature"] for record in records)
    duplicate_perturbed = sum(count - 1 for count in perturbed_signature_counts.values() if count > 1)
    summary = {
        "input_position_count": len(source_positions),
        "pair_count": len(records),
        "status_counts": dict(status_counts),
        "source_split_counts": dict(source_split_counts),
        "perturbation_color_counts": dict(color_counts),
        "low_impact_proxy_counts": dict(low_impact_proxy_counts),
        "duplicate_perturbed_signature_count": duplicate_perturbed,
    }
    return records, summary


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def input_paths(args: argparse.Namespace) -> list[str]:
    return args.input_paths or list(DEFAULT_INPUT_PATHS)


def run(args: argparse.Namespace) -> int:
    paths = [pathlib.Path(path) for path in input_paths(args)]
    source_positions = []
    source_manifest = []
    for path in paths:
        rows = read_jsonl(path)
        source_positions.extend(rows)
        source_manifest.append({"path": str(path), "row_count": len(rows)})
    records, summary = build_perturbation_pairs(source_positions, run_id=args.run_id, split_id=args.split_id)
    status = "done"
    if summary["pair_count"] != summary["input_position_count"]:
        status = "incomplete"
    if summary["duplicate_perturbed_signature_count"]:
        status = "signature_warning"
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "perturbation split construction only",
        "status": status,
        "split_id": args.split_id,
        "input_paths": [str(path) for path in paths],
        **summary,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "perturbation_pairs.jsonl", records)
    write_json(output_root / "split_report.json", report)
    write_json(
        output_root / "source_manifest.json",
        {
            "run_id": args.run_id,
            "split_id": args.split_id,
            "input_paths": source_manifest,
            "provenance_state": "derived-from-D1-D2-splits",
            "perturbation_rule": "deterministic low-adjacency single-stone perturbation",
            "evidence_class": "perturbation split construction only",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_issue and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R009")
    parser.add_argument("--split-id", default="D4_MEANINGLESS_STONE_PERTURBATION")
    parser.add_argument("--input", dest="input_paths", action="append", default=None)
    parser.add_argument("--output-root", default="experiments/results/R009_PERTURBATION_SPLIT")
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
