#!/usr/bin/env python
"""Run R014 high-budget KataGo queries for D3/D5 guard pilot positions."""

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


R011_SEARCH = load_script_module("r011_katago_search_features", SCRIPT_DIR / "r011_katago_search_features.py")

DEFAULT_PATHS = {
    "D3": "experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl",
    "D5": "experiments/results/R010_TOPOLOGY_STRESS_SPLIT/positions.jsonl",
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


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_guard_position_records(paths: dict[str, str]) -> list[dict]:
    records = []
    for row in read_jsonl(pathlib.Path(paths["D3"])):
        records.append(
            {
                **row,
                "position_id": row["position_id"],
                "dataset": "D3",
                "family": f"go_attack_{row['failure_window_bucket']}",
                "board": row["board"],
            }
        )
    for row in read_jsonl(pathlib.Path(paths["D5"])):
        records.append(
            {
                **row,
                "position_id": row["position_id"],
                "dataset": "D5",
                "family": row["family"],
                "board": row["board"],
            }
        )
    return records


def existing_query_ids(path: pathlib.Path) -> set[str]:
    return {row["query_id"] for row in read_jsonl(path) if row.get("query_id")}


def build_budget_query(record: dict, *, visits: int) -> tuple[dict, dict]:
    query, metadata = R011_SEARCH.build_query_from_board_record(record, board_key="board", visits=visits)
    query["id"] = f"{record['position_id']}__r014_v{visits}"
    metadata["query_id"] = query["id"]
    metadata["run_context"] = "R014_guard_budget_pilot"
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
                raise RuntimeError("KataGo analysis ended before returning all R014 responses")
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
    by_position = defaultdict(list)
    for row in query_rows:
        by_position[row["position_id"]].append(row)
    aggregated = []
    for position_id, rows in by_position.items():
        out = {"position_id": position_id}
        for row in sorted(rows, key=lambda item: int(item["visits"])):
            prefix = f"v{int(row['visits'])}"
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
        aggregated.append(out)
    return sorted(aggregated, key=lambda row: row["position_id"])


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    query_feature_path = output_root / "query_budget_features.jsonl"
    position_feature_path = output_root / "position_budget_features.jsonl"
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    records = load_guard_position_records(DEFAULT_PATHS)
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
        append_jsonl(output_root / "pending_budget_queries.jsonl", [query for query, _metadata in query_metadata])
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
        "evidence_class": "R014 high-budget KataGo query extraction for guard equal-compute pilot",
        "status": "done" if len(query_rows) == expected_query_count and len(aggregated) == len(records) else "running_or_incomplete",
        "position_count": len(records),
        "dataset_counts": {
            "D3": sum(1 for row in records if row["dataset"] == "D3"),
            "D5": sum(1 for row in records if row["dataset"] == "D5"),
        },
        "visits": visits_values,
        "expected_query_count": expected_query_count,
        "completed_query_count": len(query_rows),
        "newly_completed_query_count": completed,
        "position_feature_count": len(aggregated),
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth",
    }
    write_json(output_root / "r014_guard_budget_query_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R014_BUDGET_QUERIES")
    parser.add_argument("--katago-exe", default="external/engines/katago/v1.16.5-opencl/katago.exe")
    parser.add_argument("--config", default="external/engines/katago/v1.16.5-opencl/analysis_example.cfg")
    parser.add_argument("--model", default="external/engines/downloads/networks/katago/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz")
    parser.add_argument("--visits", default="1024,4096")
    parser.add_argument("--output-root", default="experiments/results/R014_GUARD_BUDGET_QUERIES")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
