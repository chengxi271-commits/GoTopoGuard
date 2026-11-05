#!/usr/bin/env python
"""R017 network-generalization queries on the R016 D3/D4/D5 subset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import time


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011_SEARCH = load_script_module("r011_katago_search_features", SCRIPT_DIR / "r011_katago_search_features.py")
R016Q = load_script_module("r016_budget_generalization_queries", SCRIPT_DIR / "r016_budget_generalization_queries.py")


def parse_networks(value: str) -> list[tuple[str, str]]:
    networks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"network entry must be alias=path: {item}")
        alias, path = item.split("=", 1)
        alias = alias.strip()
        path = path.strip()
        if not alias or not path:
            raise ValueError(f"network entry must be alias=path: {item}")
        networks.append((alias, path))
    return networks


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return R016Q.read_jsonl(path)


def append_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    return R016Q.append_jsonl(path, rows)


def write_json(path: pathlib.Path, payload) -> None:
    return R016Q.write_json(path, payload)


def existing_query_ids(path: pathlib.Path) -> set[str]:
    return {row["query_id"] for row in read_jsonl(path) if row.get("query_id")}


def build_network_query(record: dict, network_alias: str, visits: int) -> tuple[dict, dict]:
    query, metadata = R011_SEARCH.build_query_from_board_record(record, board_key="board", visits=visits)
    query["id"] = f"{record['position_id']}__r017_{network_alias}_v{visits}"
    metadata["query_id"] = query["id"]
    metadata["network_alias"] = network_alias
    metadata["run_context"] = "R017_network_generalization"
    return query, metadata


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
                raise RuntimeError("KataGo analysis ended before returning all R017 responses")
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


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    query_feature_path = output_root / "query_network_features.jsonl"
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    selected_records, skipped = R016Q.load_selected_records(R016Q.DEFAULT_PATHS, args.d4_limit, args.d5_limit)
    shard_selected_records = R016Q.shard_records(selected_records, args.shard_index, args.shard_count)
    existing = existing_query_ids(query_feature_path)
    query_metadata = []
    for record in shard_selected_records:
        for visits in visits_values:
            query, metadata = build_network_query(record, args.network_alias, visits)
            if metadata["query_id"] in existing:
                continue
            query_metadata.append((query, metadata))
    if args.prepare_only:
        R016Q.write_jsonl(output_root / "pending_network_queries.jsonl", [query for query, _metadata in query_metadata])
        completed = 0
    else:
        completed = run_katago_queries(
            pathlib.Path(args.katago_exe),
            pathlib.Path(args.config),
            pathlib.Path(args.model),
            query_metadata,
            query_feature_path,
            output_root / "katago_network_stderr.log",
        )
    query_rows = read_jsonl(query_feature_path)
    expected_query_count = len(shard_selected_records) * len(visits_values)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R017 network-generalization KataGo query extraction",
        "status": "done" if len(query_rows) == expected_query_count else "running_or_incomplete",
        "network_alias": args.network_alias,
        "model": args.model,
        "selected_position_count": len(selected_records),
        "shard_position_count": len(shard_selected_records),
        "dataset_counts": {dataset: sum(1 for row in selected_records if row["dataset"] == dataset) for dataset in ["D3", "D4", "D5"]},
        "shard_dataset_counts": {dataset: sum(1 for row in shard_selected_records if row["dataset"] == dataset) for dataset in ["D3", "D4", "D5"]},
        "d4_limit": args.d4_limit,
        "d5_limit": args.d5_limit,
        "visits": visits_values,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "expected_query_count": expected_query_count,
        "completed_query_count": len(query_rows),
        "newly_completed_query_count": completed,
        "skipped_position_count": len(skipped),
    }
    write_json(output_root / "r017_network_generalization_query_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R017_NETWORK_QUERIES")
    parser.add_argument("--network-alias", required=True)
    parser.add_argument("--katago-exe", default="external/engines/katago/v1.16.5-opencl/katago.exe")
    parser.add_argument("--config", default="external/engines/katago/v1.16.5-opencl/analysis_example.cfg")
    parser.add_argument("--model", required=True)
    parser.add_argument("--visits", default="1024")
    parser.add_argument("--d4-limit", type=int, default=1000)
    parser.add_argument("--d5-limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-root", default="experiments/results/R017_NETWORK_GENERALIZATION_QUERIES")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
