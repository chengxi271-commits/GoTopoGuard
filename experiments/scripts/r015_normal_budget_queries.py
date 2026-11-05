#!/usr/bin/env python
"""Run R015 high-budget KataGo queries for all D1/D2 compatible positions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import time
from collections import Counter, defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011_SEARCH = load_script_module("r011_katago_search_features", SCRIPT_DIR / "r011_katago_search_features.py")
R014_BUDGET = load_script_module("r014_guard_budget_queries", SCRIPT_DIR / "r014_guard_budget_queries.py")


DEFAULT_PATHS = {
    "D1": "experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl",
    "D2": "experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl",
}


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


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_normal_position_records(paths: dict[str, str]) -> tuple[list[dict], Counter]:
    records = []
    skipped = Counter()
    for dataset, path in paths.items():
        family = "natural_human" if dataset == "D1" else "natural_engine"
        for row in read_jsonl(pathlib.Path(path)):
            if not R011_SEARCH.is_query_compatible_komi(row.get("komi")):
                skipped["incompatible_komi"] += 1
                continue
            records.append(
                {
                    **row,
                    "position_id": row["position_id"],
                    "dataset": dataset,
                    "family": family,
                    "board": row["board"],
                }
            )
    return records, skipped


def existing_query_ids(path: pathlib.Path) -> set[str]:
    return {row["query_id"] for row in read_jsonl(path) if row.get("query_id")}


def build_budget_query(record: dict, *, visits: int) -> tuple[dict, dict]:
    query, metadata = R011_SEARCH.build_query_from_board_record(record, board_key="board", visits=visits)
    query["id"] = f"{record['position_id']}__r015_v{visits}"
    metadata["query_id"] = query["id"]
    metadata["run_context"] = "R015_normal_budget_regression"
    return query, metadata


def select_shard(records: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count <= 1:
        return records
    return [record for idx, record in enumerate(records) if idx % shard_count == shard_index]


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
                raise RuntimeError("KataGo analysis ended before returning all R015 responses")
            response = json.loads(line)
            features = R011_SEARCH.extract_query_features(response, visits=metadata["visits"])
            rows.append({**metadata, **features, "schema_status": "valid"})
            if len(rows) >= 64:
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


def aggregate_budget_features(query_rows: list[dict]) -> list[dict]:
    return R014_BUDGET.aggregate_budget_features(query_rows)


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    query_feature_path = output_root / "query_budget_features.jsonl"
    position_feature_path = output_root / "position_budget_features.jsonl"
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    records, skipped = load_normal_position_records(DEFAULT_PATHS)
    full_record_count = len(records)
    records = select_shard(records, args.shard_index, args.shard_count)
    if args.max_positions:
        records = records[: args.max_positions]
    existing = existing_query_ids(query_feature_path)
    query_metadata = []
    for record in records:
        for visits in visits_values:
            query, metadata = build_budget_query(record, visits=visits)
            if metadata["query_id"] in existing:
                continue
            query_metadata.append((query, metadata))
    if args.prepare_only:
        write_jsonl(output_root / "pending_budget_queries.jsonl", [query for query, _metadata in query_metadata])
        completed = 0
    else:
        completed = run_katago_queries(
            pathlib.Path(args.katago_exe),
            pathlib.Path(args.config),
            pathlib.Path(args.model),
            query_metadata,
            query_feature_path,
            output_root / "katago_budget_stderr.log",
        )
    query_rows = read_jsonl(query_feature_path)
    aggregated = aggregate_budget_features(query_rows) if query_rows else []
    if aggregated:
        write_jsonl(position_feature_path, aggregated)
    expected_query_count = len(records) * len(visits_values)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R015 high-budget KataGo query extraction for normal-position guard regression",
        "status": "done" if len(query_rows) == expected_query_count and len(aggregated) == len(records) else "running_or_incomplete",
        "full_compatible_position_count": full_record_count,
        "position_count": len(records),
        "dataset_counts": dict(Counter(row["dataset"] for row in records)),
        "skipped_counts": dict(skipped),
        "visits": visits_values,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "expected_query_count": expected_query_count,
        "completed_query_count": len(query_rows),
        "newly_completed_query_count": completed,
        "position_feature_count": len(aggregated),
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth",
    }
    write_json(output_root / "r015_normal_budget_query_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R015_BUDGET_QUERIES")
    parser.add_argument("--katago-exe", default="external/engines/katago/v1.16.5-opencl/katago.exe")
    parser.add_argument("--config", default="external/engines/katago/v1.16.5-opencl/analysis_example.cfg")
    parser.add_argument("--model", default="external/engines/downloads/networks/katago/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz")
    parser.add_argument("--visits", default="1024,4096")
    parser.add_argument("--output-root", default="experiments/results/R015_NORMAL_BUDGET_QUERIES")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
