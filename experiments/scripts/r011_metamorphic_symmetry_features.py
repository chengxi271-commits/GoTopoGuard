#!/usr/bin/env python
"""Extract R011 B5 metamorphic symmetry-consistency features.

The full B5 detector uses the same query-compatible D1-D5 positions as the
R011 KataGo search-feature run. For each position, this script queries all
seven non-identity dihedral board symmetries and compares their search outputs
against the original position after canonicalizing moves back to the original
board coordinates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import time
from collections import defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R004 = load_script_module("r004_katago_100_position_smoke", SCRIPT_DIR / "r004_katago_100_position_smoke.py")
R011_SEARCH = load_script_module("r011_katago_search_features", SCRIPT_DIR / "r011_katago_search_features.py")

SYMMETRIES = [
    "rot90",
    "rot180",
    "rot270",
    "flip_x",
    "flip_y",
    "transpose",
    "anti_transpose",
]

INVERSE_SYMMETRY = {
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_x": "flip_x",
    "flip_y": "flip_y",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}

GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_pass_move(move: str | None) -> bool:
    return isinstance(move, str) and move.lower() == "pass"


def gtp_to_point(move: str, board_size: int) -> tuple[int, int]:
    if is_pass_move(move):
        raise ValueError("pass has no board coordinate")
    col_letter = move[0].upper()
    col = GTP_COLUMNS.index(col_letter)
    row_from_bottom = int(move[1:])
    row = board_size - row_from_bottom
    return col, row


def point_to_gtp(point: tuple[int, int], board_size: int) -> str:
    return R004.sgf_coord_to_gtp(point, board_size)


def transform_point(point: tuple[int, int], board_size: int, symmetry: str) -> tuple[int, int]:
    col, row = point
    last = board_size - 1
    if symmetry == "identity":
        return col, row
    if symmetry == "rot90":
        return last - row, col
    if symmetry == "rot180":
        return last - col, last - row
    if symmetry == "rot270":
        return row, last - col
    if symmetry == "flip_x":
        return last - col, row
    if symmetry == "flip_y":
        return col, last - row
    if symmetry == "transpose":
        return row, col
    if symmetry == "anti_transpose":
        return last - row, last - col
    raise ValueError(f"unknown symmetry {symmetry}")


def transform_board(board_rows: list[str], symmetry: str) -> list[str]:
    board_size = len(board_rows)
    transformed = [["." for _ in range(board_size)] for _ in range(board_size)]
    for row, cells in enumerate(board_rows):
        for col, stone in enumerate(cells):
            new_col, new_row = transform_point((col, row), board_size, symmetry)
            transformed[new_row][new_col] = stone
    return ["".join(row) for row in transformed]


def map_gtp_move(move: str | None, board_size: int, symmetry: str) -> str | None:
    if move is None or is_pass_move(move):
        return move
    return point_to_gtp(transform_point(gtp_to_point(move, board_size), board_size, symmetry), board_size)


def canonicalize_transformed_gtp_move(move: str | None, board_size: int, symmetry: str) -> str | None:
    if move is None or is_pass_move(move):
        return move
    inverse = INVERSE_SYMMETRY[symmetry]
    return map_gtp_move(move, board_size, inverse)


def stones_from_board(board_rows: list[str]) -> list[list[str]]:
    stones = []
    size = len(board_rows)
    for row, board_row in enumerate(board_rows):
        for col, stone in enumerate(board_row):
            if stone in ("B", "W"):
                stones.append([stone, point_to_gtp((col, row), size)])
    return stones


def build_symmetry_query(record: dict, *, symmetry: str, visits: int) -> tuple[dict, dict]:
    board_rows = transform_board(record["board"], symmetry)
    initial_player, basis = R011_SEARCH.initial_player_for_record(record)
    komi = R011_SEARCH.parse_komi(record.get("komi"))
    if komi is None:
        komi = 6.5
    query_id = f"{record['position_id']}__sym_{symmetry}__v{visits}"
    query = {
        "id": query_id,
        "boardXSize": len(board_rows),
        "boardYSize": len(board_rows),
        "initialPlayer": initial_player,
        "initialStones": stones_from_board(board_rows),
        "moves": [],
        "rules": "Chinese",
        "komi": komi,
        "maxVisits": visits,
        "includePolicy": True,
        "includeOwnership": True,
    }
    metadata = {
        "query_id": query_id,
        "position_id": record["position_id"],
        "dataset": record.get("dataset"),
        "family": record.get("family"),
        "symmetry": symmetry,
        "visits": visits,
        "initial_player": initial_player,
        "initial_player_basis": basis,
        "komi": komi,
    }
    return query, metadata


def select_shard_records(records: list[dict], *, shard_index: int, shard_count: int) -> list[dict]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if shard_count == 1:
        return records
    return [record for index, record in enumerate(records) if index % shard_count == shard_index]


def existing_query_ids(path: pathlib.Path) -> set[str]:
    return {row["query_id"] for row in read_jsonl(path) if row.get("query_id")}


def run_katago_queries(
    katago_exe: pathlib.Path,
    config: pathlib.Path,
    model: pathlib.Path,
    query_metadata: list[tuple[dict, dict]],
    output_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> int:
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [str(katago_exe), "analysis", "-config", str(config), "-model", str(model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    rows = []
    try:
        for query, metadata in query_metadata:
            process.stdin.write(json.dumps(query) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("KataGo analysis ended before returning all symmetry responses")
            response = json.loads(line)
            features = R011_SEARCH.extract_query_features(response, visits=metadata["visits"])
            rows.append({**metadata, **features, "schema_status": "valid"})
            if len(rows) >= 128:
                append_jsonl(output_path, rows)
                rows = []
        if rows:
            append_jsonl(output_path, rows)
    finally:
        if process.stdin:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        stderr_handle.close()
    return len(query_metadata)


def aggregate_metamorphic_features(base_query_rows: list[dict], symmetry_query_rows: list[dict], *, board_size: int) -> list[dict]:
    base_by_position_visit = {(row["position_id"], int(row["visits"])): row for row in base_query_rows}
    symmetry_by_position_visit = defaultdict(list)
    for row in symmetry_query_rows:
        symmetry_by_position_visit[(row["position_id"], int(row["visits"]))].append(row)
    outputs = {}
    for key, rows in symmetry_by_position_visit.items():
        position_id, visits = key
        base = base_by_position_visit.get(key)
        if not base:
            continue
        base_top = base.get("top_move")
        canonical_top_moves = []
        winrate_deltas = []
        score_deltas = []
        for row in rows:
            symmetry = row["symmetry"]
            canonical_top = canonicalize_transformed_gtp_move(row.get("top_move"), board_size, symmetry)
            canonical_top_moves.append(canonical_top)
            if row.get("winrate") is not None and base.get("winrate") is not None:
                winrate_deltas.append(abs(float(row["winrate"]) - float(base["winrate"])))
            if row.get("scoreLead") is not None and base.get("scoreLead") is not None:
                score_deltas.append(abs(float(row["scoreLead"]) - float(base["scoreLead"])))
        disagreement_count = sum(1 for move in canonical_top_moves if move != base_top)
        transform_count = len(rows)
        prefix = f"sym_v{visits}"
        features = outputs.setdefault(position_id, {"position_id": position_id})
        features[f"{prefix}_valid_transform_count"] = transform_count
        features[f"{prefix}_top_move_disagreement_count"] = disagreement_count
        features[f"{prefix}_top_move_disagreement_rate"] = disagreement_count / transform_count if transform_count else 0.0
        features[f"{prefix}_top_move_unique_count"] = len({move for move in canonical_top_moves if move is not None})
        features[f"{prefix}_winrate_abs_delta_mean"] = sum(winrate_deltas) / len(winrate_deltas) if winrate_deltas else 0.0
        features[f"{prefix}_winrate_abs_delta_max"] = max(winrate_deltas) if winrate_deltas else 0.0
        features[f"{prefix}_score_abs_delta_mean"] = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
        features[f"{prefix}_score_abs_delta_max"] = max(score_deltas) if score_deltas else 0.0
    return sorted(outputs.values(), key=lambda row: row["position_id"])


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    query_feature_path = output_root / "symmetry_query_features.jsonl"
    position_feature_path = output_root / "position_metamorphic_features.jsonl"
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    symmetries = [value for value in args.symmetries.split(",") if value.strip()]
    unknown = sorted(set(symmetries) - set(SYMMETRIES))
    if unknown:
        raise ValueError(f"unknown symmetries: {unknown}")
    records, skipped = R011_SEARCH.load_engine_position_records()
    if args.max_positions:
        records = records[: args.max_positions]
    all_position_count = len(records)
    records = select_shard_records(records, shard_index=args.shard_index, shard_count=args.shard_count)
    existing = existing_query_ids(query_feature_path)
    query_metadata = []
    for record in records:
        for symmetry in symmetries:
            for visits in visits_values:
                query, metadata = build_symmetry_query(record, symmetry=symmetry, visits=visits)
                if metadata["query_id"] in existing:
                    continue
                query_metadata.append((query, metadata))
    if args.aggregate_only:
        completed = 0
    elif args.prepare_only:
        append_jsonl(output_root / "pending_symmetry_queries.jsonl", [query for query, _metadata in query_metadata])
        completed = 0
    else:
        completed = run_katago_queries(
            pathlib.Path(args.katago_exe),
            pathlib.Path(args.config),
            pathlib.Path(args.model),
            query_metadata,
            query_feature_path,
            output_root / "katago_symmetry_stderr.log",
        )
    symmetry_rows = read_jsonl(query_feature_path)
    base_rows = read_jsonl(pathlib.Path(args.base_query_features))
    if symmetry_rows and base_rows:
        aggregated = aggregate_metamorphic_features(base_rows, symmetry_rows, board_size=args.board_size)
        position_feature_path.write_text("", encoding="utf-8")
        append_jsonl(position_feature_path, aggregated)
    else:
        aggregated = []
    expected_query_count = len(records) * len(symmetries) * len(visits_values)
    expected_position_count = len(records)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "KataGo metamorphic symmetry-consistency feature extraction",
        "status": "done" if len(symmetry_rows) == expected_query_count and len(aggregated) == expected_position_count else "running_or_incomplete",
        "all_query_compatible_position_count": all_position_count,
        "query_compatible_position_count": len(records),
        "skipped_position_count": len(skipped),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "symmetries": symmetries,
        "visits": visits_values,
        "expected_query_count": expected_query_count,
        "completed_query_count": len(symmetry_rows),
        "newly_completed_query_count": completed,
        "expected_position_feature_count": expected_position_count,
        "position_feature_count": len(aggregated),
        "base_query_feature_path": args.base_query_features,
        "skipped_sample": skipped[:20],
    }
    write_json(output_root / "r011_metamorphic_symmetry_feature_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R011_METAMORPHIC_SYMMETRY_FEATURES")
    parser.add_argument("--katago-exe", default="external/engines/katago/v1.16.5-opencl/katago.exe")
    parser.add_argument("--config", default="external/engines/katago/v1.16.5-opencl/analysis_example.cfg")
    parser.add_argument("--model", default="external/engines/downloads/networks/katago/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz")
    parser.add_argument("--visits", default="64,256")
    parser.add_argument("--symmetries", default=",".join(SYMMETRIES))
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--base-query-features", default="experiments/results/R011_KATAGO_SEARCH_FEATURES/query_search_features.jsonl")
    parser.add_argument("--output-root", default="experiments/results/R011_METAMORPHIC_SYMMETRY_FEATURES")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
