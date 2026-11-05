#!/usr/bin/env python
"""Merge R015 normal-budget query shards."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R015_QUERIES = load_script_module("r015_normal_budget_queries", SCRIPT_DIR / "r015_normal_budget_queries.py")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_shard_rows(shard_root: pathlib.Path) -> list[dict]:
    rows = []
    for shard_dir in sorted(shard_root.glob("shard_*")):
        rows.extend(read_jsonl(shard_dir / "query_budget_features.jsonl"))
    return rows


def merge_query_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    merged = []
    for row in rows:
        query_id = row.get("query_id")
        if query_id in seen:
            raise ValueError(f"duplicate query_id in R015 shard merge: {query_id}")
        seen.add(query_id)
        merged.append(row)
    return sorted(merged, key=lambda row: row.get("query_id", ""))


def run(args: argparse.Namespace) -> int:
    shard_root = pathlib.Path(args.shard_root)
    output_root = pathlib.Path(args.output_root)
    rows = merge_query_rows(collect_shard_rows(shard_root))
    aggregated = R015_QUERIES.aggregate_budget_features(rows) if rows else []
    write_jsonl(output_root / "query_budget_features.jsonl", rows)
    write_jsonl(output_root / "position_budget_features.jsonl", aggregated)
    expected_queries = int(args.expected_queries)
    expected_positions = int(args.expected_positions)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R015 merged normal-budget query shards",
        "status": "done" if len(rows) == expected_queries and len(aggregated) == expected_positions else "running_or_incomplete",
        "shard_root": str(shard_root),
        "completed_query_count": len(rows),
        "position_feature_count": len(aggregated),
        "expected_query_count": expected_queries,
        "expected_position_count": expected_positions,
    }
    write_json(output_root / "r015_normal_budget_query_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R015_MERGE_BUDGET_SHARDS")
    parser.add_argument("--shard-root", default="experiments/results/R015_NORMAL_BUDGET_QUERIES_SHARDS")
    parser.add_argument("--output-root", default="experiments/results/R015_NORMAL_BUDGET_QUERIES")
    parser.add_argument("--expected-queries", type=int, default=13562)
    parser.add_argument("--expected-positions", type=int, default=6781)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
