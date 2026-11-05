#!/usr/bin/env python
"""R004 KataGo 100-position analysis smoke over a D0 subset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import subprocess
import time
from collections import Counter


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R001 = load_script_module("r001_sgf_replay_smoke", SCRIPT_DIR / "r001_sgf_replay_smoke.py")
R003 = load_script_module("r003_katago_schema_smoke", SCRIPT_DIR / "r003_katago_schema_smoke.py")

DEFAULT_KATAGO_EXE = "external/engines/katago/v1.16.5-opencl/katago.exe"
DEFAULT_KATAGO_CONFIG = "external/engines/katago/v1.16.5-opencl/analysis_example.cfg"
DEFAULT_KATAGO_MODEL = R003.DEFAULT_KATAGO_MODEL
DEFAULT_VISITS = [64, 256]
DEFAULT_TARGET_COUNTS = {
    "jgdb": 40,
    "katago_kata1_2020_12_12": 40,
    "go_attack_site_examples": 20,
}


def sgf_coord_to_gtp(coord: tuple[int, int] | None, board_size: int) -> str:
    if coord is None:
        return "pass"
    col, row = coord
    columns = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{columns[col]}{board_size - row}"


def safe_float(raw, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def is_katago_komi_compatible(raw) -> bool:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return True
    return -150.0 <= value <= 150.0 and abs(value * 2 - round(value * 2)) < 1e-9


def choose_move_prefix_len(game) -> int:
    move_count = len(game.moves)
    if move_count >= 80:
        return 80
    if move_count >= 40:
        return 40
    return max(1, move_count // 2)


def build_query_from_game(position_id: str, game, *, move_prefix_len: int, max_visits: int) -> dict:
    moves = [[move.color, sgf_coord_to_gtp(move.coord, game.board_size)] for move in game.moves[:move_prefix_len]]
    initial_stones = [[color, sgf_coord_to_gtp(coord, game.board_size)] for color, coord in game.setup_stones]
    return {
        "id": f"{position_id}_v{max_visits}",
        "boardXSize": game.board_size,
        "boardYSize": game.board_size,
        "initialPlayer": "B",
        "initialStones": initial_stones,
        "moves": moves,
        "rules": game.rules or "Chinese",
        "komi": safe_float(game.komi, 7.5),
        "maxVisits": max_visits,
        "includePolicy": True,
        "includeOwnership": True,
    }


def fetch_go_attack_site_records_with_retries(limit: int, *, attempts: int, fetch_fn=R001.fetch_go_attack_site_sgf_records):
    errors = []
    for attempt in range(1, max(1, attempts) + 1):
        records, summary = fetch_fn(limit)
        if records:
            summary = {**summary, "attempts_used": attempt}
            if errors:
                summary["previous_errors"] = errors
            return records, summary
        errors.append({"attempt": attempt, "summary": summary})
    return [], {
        "status": "site_fetch_failed_after_retries",
        "attempts_used": max(1, attempts),
        "errors": errors,
    }


def replay_valid_games(source_id: str, records, *, required_board_size: int, limit: int):
    accepted = []
    rejected = []
    scanned = 0
    for name, sgf in records:
        scanned += 1
        try:
            game = R001.parse_sgf_mainline(sgf)
            if game.board_size != required_board_size:
                rejected.append({"source_id": source_id, "name": name, "status": "rejected_board_size", "board_size": game.board_size})
                continue
            result = R001.replay_game(game)
            if not result.ok:
                rejected.append({"source_id": source_id, "name": name, "status": "invalid_replay", "errors": result.errors})
                continue
            if not is_katago_komi_compatible(game.komi):
                rejected.append({"source_id": source_id, "name": name, "status": "rejected_katago_incompatible_komi", "komi": game.komi})
                continue
            accepted.append((name, game, R001.position_signature(result.final_board)))
            if len(accepted) >= limit:
                break
        except Exception as exc:
            rejected.append({"source_id": source_id, "name": name, "status": "parse_error", "errors": [str(exc)]})
    return accepted, {"source_id": source_id, "scanned_sgf_count": scanned, "accepted_valid_count": len(accepted), "rejected_count": len(rejected), "rejections_sample": rejected[:10]}


def build_position_records(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    source_summaries = []
    position_records = []
    source_specs = [
        (
            "jgdb",
            R001.iter_tar_sgfs(args.jgdb_url, "r|gz", max_scan=args.jgdb_max_scan),
            args.jgdb_count,
        ),
        (
            "katago_kata1_2020_12_12",
            R001.iter_tar_sgfs(args.katago_url, "r|bz2", max_scan=args.katago_max_scan),
            args.katago_count,
        ),
    ]
    for source_id, records, target_count in source_specs:
        games, summary = replay_valid_games(source_id, records, required_board_size=args.required_board_size, limit=target_count)
        source_summaries.append(summary)
        for index, (name, game, final_signature) in enumerate(games, start=1):
            position_records.append(
                {
                    "position_id": f"R004_{source_id}_{index:03d}",
                    "source_id": source_id,
                    "source_name": name,
                    "move_prefix_len": choose_move_prefix_len(game),
                    "game_move_count": len(game.moves),
                    "board_size": game.board_size,
                    "komi": game.komi,
                    "rules": game.rules,
                    "final_position_signature": final_signature,
                    "query_basis": "sgf_move_prefix",
                    "game": game,
                }
            )
    go_attack_records, go_attack_summary = fetch_go_attack_site_records_with_retries(args.go_attack_count, attempts=args.source_fetch_retries)
    source_summaries.append({"source_id": "go_attack_site_examples", **go_attack_summary})
    games, summary = replay_valid_games("go_attack_site_examples", go_attack_records, required_board_size=args.required_board_size, limit=args.go_attack_count)
    source_summaries.append(summary)
    for index, (name, game, final_signature) in enumerate(games, start=1):
        position_records.append(
            {
                "position_id": f"R004_go_attack_site_examples_{index:03d}",
                "source_id": "go_attack_site_examples",
                "source_name": name,
                "move_prefix_len": choose_move_prefix_len(game),
                "game_move_count": len(game.moves),
                "board_size": game.board_size,
                "komi": game.komi,
                "rules": game.rules,
                "final_position_signature": final_signature,
                "query_basis": "sgf_move_prefix",
                "game": game,
            }
        )
    return position_records, source_summaries


def serialize_position_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "game"}


def run_katago_batch(katago_exe: pathlib.Path, config: pathlib.Path, model: pathlib.Path, queries: list[dict], output_root: pathlib.Path, *, timeout_seconds: int):
    command = [str(katago_exe), "analysis", "-config", str(config), "-model", str(model)]
    stderr_path = output_root / "raw_katago_stderr.log"
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(katago_exe.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    responses = []
    try:
        for query in queries:
            start = time.perf_counter()
            proc.stdin.write(json.dumps(query, separators=(",", ":")) + "\n")
            proc.stdin.flush()
            deadline = time.perf_counter() + timeout_seconds
            response_line = ""
            while time.perf_counter() < deadline:
                line = proc.stdout.readline()
                if line.strip().startswith("{"):
                    response_line = line
                    break
            if not response_line:
                raise RuntimeError(f"no JSON response for query {query['id']}")
            response = json.loads(response_line)
            response["_latency_seconds"] = time.perf_counter() - start
            responses.append(response)
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        stderr_handle.close()
    return responses


def summarize_batch(schema_records: list[dict], *, expected_positions: int, expected_visits: list[int]) -> dict:
    position_ids = sorted({record["position_id"] for record in schema_records})
    visit_values = sorted({record["visits"] for record in schema_records})
    status_counts = Counter(record["schema_status"] for record in schema_records)
    latencies = [record["latency_seconds"] for record in schema_records]
    by_visit = {}
    for visits in visit_values:
        visit_latencies = [record["latency_seconds"] for record in schema_records if record["visits"] == visits]
        by_visit[str(visits)] = {
            "query_count": len(visit_latencies),
            "mean_latency_seconds": statistics.fmean(visit_latencies) if visit_latencies else None,
            "max_latency_seconds": max(visit_latencies) if visit_latencies else None,
        }
    status = "done"
    if len(position_ids) != expected_positions or visit_values != sorted(expected_visits):
        status = "completion_failure"
    if status_counts.get("valid", 0) != expected_positions * len(expected_visits):
        status = "schema_invalid"
    return {
        "run_id": "R004",
        "generated_at_unix": int(time.time()),
        "evidence_class": "engine analysis smoke only",
        "status": status,
        "position_count": len(position_ids),
        "expected_position_count": expected_positions,
        "query_count": len(schema_records),
        "expected_query_count": expected_positions * len(expected_visits),
        "visits": visit_values,
        "expected_visits": sorted(expected_visits),
        "schema_valid_count": status_counts.get("valid", 0),
        "schema_status_counts": dict(status_counts),
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "by_visit": by_visit,
    }


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visits = [int(value) for value in args.visits.split(",") if value.strip()]
    expected_positions = args.jgdb_count + args.katago_count + args.go_attack_count
    position_records, source_summaries = build_position_records(args)
    write_json(output_root / "source_manifest.json", source_summaries)
    write_jsonl(output_root / "position_records.jsonl", [serialize_position_record(record) for record in position_records])
    queries = []
    query_metadata = []
    for record in position_records:
        for max_visits in visits:
            query = build_query_from_game(record["position_id"], record["game"], move_prefix_len=record["move_prefix_len"], max_visits=max_visits)
            queries.append(query)
            query_metadata.append({**serialize_position_record(record), "query_id": query["id"], "visits": max_visits})
    write_jsonl(output_root / "katago_queries.jsonl", queries)
    backend_manifest = {
        "run_id": "R004",
        "generated_at_unix": int(time.time()),
        "evidence_class": "engine analysis smoke only",
        "katago_exe": str(pathlib.Path(args.katago_exe).resolve()),
        "config": str(pathlib.Path(args.config).resolve()),
        "model": str(pathlib.Path(args.model).resolve()),
        "katago_exe_sha256": R003.sha256_file(pathlib.Path(args.katago_exe).resolve()),
        "config_sha256": R003.sha256_file(pathlib.Path(args.config).resolve()),
        "model_sha256": R003.sha256_file(pathlib.Path(args.model).resolve()),
        "visits": visits,
        "target_counts": {
            "jgdb": args.jgdb_count,
            "katago_kata1_2020_12_12": args.katago_count,
            "go_attack_site_examples": args.go_attack_count,
        },
    }
    write_json(output_root / "backend_manifest.json", backend_manifest)
    if len(position_records) != expected_positions:
        report = {
            "run_id": "R004",
            "generated_at_unix": int(time.time()),
            "evidence_class": "engine analysis smoke only",
            "status": "position_build_failure",
            "position_count": len(position_records),
            "expected_position_count": expected_positions,
        }
        write_json(output_root / "r004_katago_100_position_smoke_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2 if args.fail_on_issue else 0
    responses = run_katago_batch(
        pathlib.Path(args.katago_exe).resolve(),
        pathlib.Path(args.config).resolve(),
        pathlib.Path(args.model).resolve(),
        queries,
        output_root,
        timeout_seconds=args.per_query_timeout_seconds,
    )
    write_jsonl(output_root / "raw_katago_responses.jsonl", responses)
    schema_records = []
    metadata_by_query = {item["query_id"]: item for item in query_metadata}
    for response in responses:
        query_id = response["id"]
        metadata = metadata_by_query[query_id]
        schema = R003.validate_response_schema(response, expected_id=query_id)
        schema_records.append(
            {
                "query_id": query_id,
                "position_id": metadata["position_id"],
                "source_id": metadata["source_id"],
                "source_name": metadata["source_name"],
                "visits": metadata["visits"],
                "schema_status": schema["status"],
                "move_info_count": schema["move_info_count"],
                "latency_seconds": response.get("_latency_seconds"),
                "schema_validation": schema,
            }
        )
    write_jsonl(output_root / "schema_records.jsonl", schema_records)
    report = summarize_batch(schema_records, expected_positions=expected_positions, expected_visits=visits)
    report["backend_manifest_path"] = str(output_root / "backend_manifest.json")
    report["source_manifest_path"] = str(output_root / "source_manifest.json")
    write_json(output_root / "r004_katago_100_position_smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_issue and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--katago-exe", default=DEFAULT_KATAGO_EXE)
    parser.add_argument("--config", default=DEFAULT_KATAGO_CONFIG)
    parser.add_argument("--model", default=DEFAULT_KATAGO_MODEL)
    parser.add_argument("--output-root", default="experiments/results/R004_KATAGO_100_POSITION_SMOKE")
    parser.add_argument("--jgdb-url", default=R001.DEFAULT_JGDB_URL)
    parser.add_argument("--katago-url", default=R001.DEFAULT_KATAGO_URL)
    parser.add_argument("--jgdb-count", type=int, default=DEFAULT_TARGET_COUNTS["jgdb"])
    parser.add_argument("--katago-count", type=int, default=DEFAULT_TARGET_COUNTS["katago_kata1_2020_12_12"])
    parser.add_argument("--go-attack-count", type=int, default=DEFAULT_TARGET_COUNTS["go_attack_site_examples"])
    parser.add_argument("--jgdb-max-scan", type=int, default=300)
    parser.add_argument("--katago-max-scan", type=int, default=300)
    parser.add_argument("--source-fetch-retries", type=int, default=3)
    parser.add_argument("--required-board-size", type=int, default=19)
    parser.add_argument("--visits", default="64,256")
    parser.add_argument("--per-query-timeout-seconds", type=int, default=240)
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
