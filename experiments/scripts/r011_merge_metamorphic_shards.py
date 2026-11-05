#!/usr/bin/env python
"""Merge R011 B5 metamorphic symmetry shard outputs and aggregate features."""

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


META = load_script_module("r011_metamorphic_symmetry_features", SCRIPT_DIR / "r011_metamorphic_symmetry_features.py")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge_query_rows(shard_paths: list[pathlib.Path]) -> list[dict]:
    merged = []
    seen = set()
    duplicates = []
    for path in shard_paths:
        for row in META.read_jsonl(path):
            query_id = row.get("query_id")
            if query_id in seen:
                duplicates.append(query_id)
                continue
            seen.add(query_id)
            merged.append(row)
    if duplicates:
        sample = ", ".join(str(value) for value in duplicates[:10])
        raise ValueError(f"duplicate query_id values while merging shards: {sample}")
    return merged


def run(args: argparse.Namespace) -> int:
    shards_root = pathlib.Path(args.shards_root)
    output_root = pathlib.Path(args.output_root)
    shard_paths = [shards_root / f"shard_{index:02d}" / "symmetry_query_features.jsonl" for index in range(args.shard_count)]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard query files: {missing}")
    symmetry_rows = merge_query_rows(shard_paths)
    write_jsonl(output_root / "symmetry_query_features.jsonl", symmetry_rows)
    base_rows = META.read_jsonl(pathlib.Path(args.base_query_features))
    aggregated = META.aggregate_metamorphic_features(base_rows, symmetry_rows, board_size=args.board_size)
    write_jsonl(output_root / "position_metamorphic_features.jsonl", aggregated)
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    symmetries = [value for value in args.symmetries.split(",") if value.strip()]
    records, skipped = META.R011_SEARCH.load_engine_position_records()
    expected_query_count = len(records) * len(symmetries) * len(visits_values)
    expected_position_count = len(records)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "merged KataGo metamorphic symmetry-consistency feature extraction",
        "status": "done" if len(symmetry_rows) == expected_query_count and len(aggregated) == expected_position_count else "running_or_incomplete",
        "query_compatible_position_count": len(records),
        "skipped_position_count": len(skipped),
        "shard_count": args.shard_count,
        "symmetries": symmetries,
        "visits": visits_values,
        "expected_query_count": expected_query_count,
        "completed_query_count": len(symmetry_rows),
        "expected_position_feature_count": expected_position_count,
        "position_feature_count": len(aggregated),
        "base_query_feature_path": args.base_query_features,
        "shards_root": args.shards_root,
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
    parser.add_argument("--shards-root", default="experiments/results/R011_METAMORPHIC_SYMMETRY_FEATURES_SHARDS")
    parser.add_argument("--output-root", default="experiments/results/R011_METAMORPHIC_SYMMETRY_FEATURES")
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--visits", default="64,256")
    parser.add_argument("--symmetries", default=",".join(META.SYMMETRIES))
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--base-query-features", default="experiments/results/R011_KATAGO_SEARCH_FEATURES/query_search_features.jsonl")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
