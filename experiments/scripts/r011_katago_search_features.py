#!/usr/bin/env python
"""Extract KataGo search features for R011 with checkpoint/resume support."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

DEFAULT_PATHS = {
    "D1": "experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl",
    "D2": "experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl",
    "D3": "experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl",
    "D4": "experiments/results/R009_PERTURBATION_SPLIT/perturbation_pairs.jsonl",
    "D5": "experiments/results/R010_TOPOLOGY_STRESS_SPLIT/positions.jsonl",
}


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


def append_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def initial_player_for_record(record: dict) -> tuple[str, str]:
    if record.get("move_prefix_len") is not None:
        return ("B" if int(record["move_prefix_len"]) % 2 == 0 else "W", "move_prefix_len_parity")
    if record.get("source_move_prefix_len") is not None:
        return ("B" if int(record["source_move_prefix_len"]) % 2 == 0 else "W", "source_move_prefix_len_parity")
    return "B", "generated_default_black"


def stones_from_board(board_rows: list[str]) -> list[list[str]]:
    stones = []
    size = len(board_rows)
    for row, board_row in enumerate(board_rows):
        for col, stone in enumerate(board_row):
            if stone in ("B", "W"):
                stones.append([stone, R004.sgf_coord_to_gtp((col, row), size)])
    return stones


def parse_komi(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_query_compatible_komi(value) -> bool:
    return R004.is_katago_komi_compatible(value)


def build_query_from_board_record(record: dict, *, board_key: str, visits: int) -> tuple[dict, dict]:
    board_rows = record[board_key]
    initial_player, basis = initial_player_for_record(record)
    komi = parse_komi(record.get("komi"))
    if komi is None:
        komi = 6.5
    query_id = f"{record['position_id']}_p{initial_player}_v{visits}"
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
        "visits": visits,
        "initial_player": initial_player,
        "initial_player_basis": basis,
        "komi": komi,
        "board_key": board_key,
    }
    return query, metadata


def entropy(values: list[float]) -> float:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0:
        return 0.0
    result = 0.0
    for value in values:
        p = max(0.0, float(value)) / total
        if p > 0:
            result -= p * math.log2(p)
    return result


def extract_query_features(response: dict, *, visits: int) -> dict:
    move_infos = response.get("moveInfos") or []
    root = response.get("rootInfo") or {}
    top = move_infos[0] if move_infos else {}
    second = move_infos[1] if len(move_infos) > 1 else {}
    visit_values = [float(info.get("visits", 0.0)) for info in move_infos]
    prior_values = [float(info.get("prior", 0.0)) for info in move_infos]
    policy_values = [float(value) for value in response.get("policy", []) if value is not None]
    top_visits = float(top.get("visits", 0.0))
    second_visits = float(second.get("visits", 0.0))
    return {
        "query_id": response.get("id"),
        "visits": visits,
        "winrate": root.get("winrate"),
        "scoreLead": root.get("scoreLead"),
        "scoreStdev": root.get("scoreStdev"),
        "utility": root.get("utility"),
        "top_move": top.get("move"),
        "top_visit_count": top_visits,
        "second_visit_count": second_visits,
        "top_visit_gap": top_visits - second_visits,
        "top_prior": top.get("prior"),
        "second_prior": second.get("prior"),
        "move_info_count": len(move_infos),
        "visit_entropy": entropy(visit_values),
        "prior_entropy": entropy(prior_values),
        "policy_entropy": entropy(policy_values),
    }


def aggregate_position_features(query_feature_rows: list[dict]) -> list[dict]:
    by_position = defaultdict(list)
    for row in query_feature_rows:
        by_position[row["position_id"]].append(row)
    aggregated = []
    for position_id, rows in by_position.items():
        out = {"position_id": position_id}
        by_visit = {int(row["visits"]): row for row in rows}
        for visits, row in sorted(by_visit.items()):
            prefix = f"v{visits}"
            for key in [
                "winrate",
                "scoreLead",
                "scoreStdev",
                "utility",
                "top_visit_gap",
                "top_prior",
                "move_info_count",
                "visit_entropy",
                "prior_entropy",
                "policy_entropy",
            ]:
                out[f"{prefix}_{key}"] = row.get(key)
            out[f"{prefix}_top_move"] = row.get("top_move")
        if 64 in by_visit and 256 in by_visit:
            low = by_visit[64]
            high = by_visit[256]
            out["budget_top_move_disagreement_64_256"] = int(low.get("top_move") != high.get("top_move"))
            out["budget_winrate_abs_delta_64_256"] = abs(float(low.get("winrate") or 0.0) - float(high.get("winrate") or 0.0))
            out["budget_score_abs_delta_64_256"] = abs(float(low.get("scoreLead") or 0.0) - float(high.get("scoreLead") or 0.0))
        aggregated.append(out)
    return sorted(aggregated, key=lambda row: row["position_id"])


def load_engine_position_records() -> tuple[list[dict], list[dict]]:
    records = []
    skipped = []
    for dataset, path in DEFAULT_PATHS.items():
        for row in read_jsonl(pathlib.Path(path)):
            if dataset in ("D1", "D2", "D3"):
                record = {
                    **row,
                    "position_id": row["position_id"],
                    "dataset": dataset,
                    "family": "natural" if dataset in ("D1", "D2") else f"go_attack_{row['failure_window_bucket']}",
                    "board": row["board"],
                }
            elif dataset == "D4":
                record = {
                    **row,
                    "position_id": row["pair_id"],
                    "dataset": dataset,
                    "family": "perturbation_low_adjacency",
                    "board": row["perturbed_board"],
                }
            else:
                record = {
                    **row,
                    "position_id": row["position_id"],
                    "dataset": dataset,
                    "family": row["family"],
                    "board": row["board"],
                }
            if not is_query_compatible_komi(record.get("komi")):
                skipped.append({"position_id": record["position_id"], "dataset": dataset, "reason": "katago_incompatible_komi", "komi": record.get("komi")})
                continue
            records.append(record)
    return records, skipped


def existing_query_ids(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("query_id"):
            ids.add(row["query_id"])
    return ids


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
                raise RuntimeError("KataGo analysis ended before returning all responses")
            response = json.loads(line)
            features = extract_query_features(response, visits=metadata["visits"])
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


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    query_feature_path = output_root / "query_search_features.jsonl"
    position_feature_path = output_root / "position_search_features.jsonl"
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    records, skipped = load_engine_position_records()
    if args.max_positions:
        records = records[: args.max_positions]
    existing = existing_query_ids(query_feature_path)
    query_metadata = []
    for record in records:
        for visits in visits_values:
            query, metadata = build_query_from_board_record(record, board_key="board", visits=visits)
            if metadata["query_id"] in existing:
                continue
            query_metadata.append((query, metadata))
    if args.prepare_only:
        append_jsonl(output_root / "pending_queries.jsonl", [query for query, _metadata in query_metadata])
        completed = 0
    else:
        completed = run_katago_queries(
            pathlib.Path(args.katago_exe),
            pathlib.Path(args.config),
            pathlib.Path(args.model),
            query_metadata,
            query_feature_path,
            output_root / "katago_stderr.log",
        )
    query_rows = read_jsonl(query_feature_path) if query_feature_path.exists() else []
    if query_rows:
        aggregated = aggregate_position_features(query_rows)
        position_feature_path.parent.mkdir(parents=True, exist_ok=True)
        position_feature_path.write_text("", encoding="utf-8")
        append_jsonl(position_feature_path, aggregated)
    expected_query_count = len(records) * len(visits_values)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "KataGo search feature extraction",
        "status": "done" if len(query_rows) == expected_query_count else "running_or_incomplete",
        "query_compatible_position_count": len(records),
        "skipped_position_count": len(skipped),
        "expected_query_count": expected_query_count,
        "completed_query_count": len(query_rows),
        "newly_completed_query_count": completed,
        "visits": visits_values,
        "skipped_sample": skipped[:20],
    }
    write_json(output_root / "r011_katago_search_feature_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R011_SEARCH_FEATURES")
    parser.add_argument("--katago-exe", default="external/engines/katago/v1.16.5-opencl/katago.exe")
    parser.add_argument("--config", default="external/engines/katago/v1.16.5-opencl/analysis_example.cfg")
    parser.add_argument("--model", default="external/engines/downloads/networks/katago/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz")
    parser.add_argument("--visits", default="64,256")
    parser.add_argument("--output-root", default="experiments/results/R011_KATAGO_SEARCH_FEATURES")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
