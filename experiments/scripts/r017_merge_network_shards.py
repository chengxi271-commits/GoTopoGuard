#!/usr/bin/env python
"""Merge R017 network-generalization query shards."""

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


R017Q = load_script_module("r017_network_generalization_queries", SCRIPT_DIR / "r017_network_generalization_queries.py")


def run(args: argparse.Namespace) -> int:
    shards_root = pathlib.Path(args.shards_root)
    output_root = pathlib.Path(args.output_root)
    shard_dirs = sorted(path for path in shards_root.glob("*_shard_*") if path.is_dir())
    query_by_id = {}
    reports = []
    for shard_dir in shard_dirs:
        report_path = shard_dir / "r017_network_generalization_query_report.json"
        if report_path.exists():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        for row in R017Q.read_jsonl(shard_dir / "query_network_features.jsonl"):
            query_by_id[row["query_id"]] = row
    query_rows = sorted(query_by_id.values(), key=lambda row: row["query_id"])
    R017Q.R016Q.write_jsonl(output_root / "query_network_features.jsonl", query_rows)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R017 merged network-generalization query shards",
        "status": "done",
        "shard_count": len(shard_dirs),
        "completed_query_count": len(query_rows),
        "expected_query_count": args.expected_queries,
        "network_counts": {},
        "shard_reports": reports,
    }
    for row in query_rows:
        report["network_counts"][row["network_alias"]] = report["network_counts"].get(row["network_alias"], 0) + 1
    if args.expected_queries and len(query_rows) != args.expected_queries:
        report["status"] = "incomplete"
    R017Q.write_json(output_root / "r017_network_generalization_query_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R017_NETWORK_QUERY_MERGE")
    parser.add_argument("--shards-root", default="experiments/results/R017_NETWORK_GENERALIZATION_QUERIES_SHARDS")
    parser.add_argument("--output-root", default="experiments/results/R017_NETWORK_GENERALIZATION_QUERIES")
    parser.add_argument("--expected-queries", type=int, default=0)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
