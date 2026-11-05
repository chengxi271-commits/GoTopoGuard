#!/usr/bin/env python
"""R016 budget-generalization KataGo queries for D3/D4/D5 subset."""

from __future__ import annotations

import argparse
import hashlib
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
    "D4": "experiments/results/R009_PERTURBATION_SPLIT/perturbation_pairs.jsonl",
    "D5": "experiments/results/R010_TOPOLOGY_STRESS_SPLIT/positions.jsonl",
}


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
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


def stable_sample(rows: list[dict], limit: int, key_name: str) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (stable_hash(str(row[key_name])), str(row[key_name])))
    return ordered if limit <= 0 else ordered[:limit]


def shard_records(records: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    if shard_count <= 1:
        return records
    return [row for row in records if stable_hash(row["position_id"]) % shard_count == shard_index]


def load_selected_records(paths: dict[str, str], d4_limit: int, d5_limit: int) -> tuple[list[dict], list[dict]]:
    records = []
    skipped = []
    for row in read_jsonl(pathlib.Path(paths["D3"])):
        record = {
            **row,
            "position_id": row["position_id"],
            "dataset": "D3",
            "family": f"go_attack_{row['failure_window_bucket']}",
            "board": row["board"],
        }
        if not R011_SEARCH.is_query_compatible_komi(record.get("komi")):
            skipped.append({"position_id": record["position_id"], "dataset": "D3", "reason": "katago_incompatible_komi", "komi": record.get("komi")})
            continue
        records.append(record)
    d4_rows = []
    for row in read_jsonl(pathlib.Path(paths["D4"])):
        record = {
            **row,
            "position_id": row["pair_id"],
            "dataset": "D4",
            "family": "perturbation_low_adjacency",
            "board": row["perturbed_board"],
        }
        if not R011_SEARCH.is_query_compatible_komi(record.get("komi")):
            skipped.append({"position_id": record["position_id"], "dataset": "D4", "reason": "katago_incompatible_komi", "komi": record.get("komi")})
            continue
        d4_rows.append(record)
    records.extend(stable_sample(d4_rows, d4_limit, "position_id"))
    d5_rows = []
    for row in read_jsonl(pathlib.Path(paths["D5"])):
        record = {
            **row,
            "position_id": row["position_id"],
            "dataset": "D5",
            "family": row["family"],
            "board": row["board"],
        }
        if not R011_SEARCH.is_query_compatible_komi(record.get("komi")):
            skipped.append({"position_id": record["position_id"], "dataset": "D5", "reason": "katago_incompatible_komi", "komi": record.get("komi")})
            continue
        d5_rows.append(record)
    records.extend(stable_sample(d5_rows, d5_limit, "position_id"))
    return records, skipped


def existing_query_ids(path: pathlib.Path) -> set[str]:
    return {row["query_id"] for row in read_jsonl(path) if row.get("query_id")}


def build_budget_query(record: dict, visits: int) -> tuple[dict, dict]:
    query, metadata = R011_SEARCH.build_query_from_board_record(record, board_key="board", visits=visits)
    query["id"] = f"{record['position_id']}__r016_v{visits}"
    metadata["query_id"] = query["id"]
    metadata["run_context"] = "R016_budget_generalization"
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
                raise RuntimeError("KataGo analysis ended before returning all R016 responses")
            response = json.loads(line)
            features = R011_SEARCH.extract_query_features(response, visits=metadata["visits"])
            rows.append({**metadata, **features, "schema_status": "valid"})
            if len(rows) >= 32:
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
        for low, high in [(1024, 8192)]:
            if low in by_visit and high in by_visit:
                low_row = by_visit[low]
                high_row = by_visit[high]
                out[f"budget_top_move_disagreement_{low}_{high}"] = int(low_row.get("top_move") != high_row.get("top_move"))
                out[f"budget_winrate_abs_delta_{low}_{high}"] = abs(float(low_row.get("winrate") or 0.0) - float(high_row.get("winrate") or 0.0))
                out[f"budget_score_abs_delta_{low}_{high}"] = abs(float(low_row.get("scoreLead") or 0.0) - float(high_row.get("scoreLead") or 0.0))
        aggregated.append(out)
    return sorted(aggregated, key=lambda row: row["position_id"])


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    query_feature_path = output_root / "query_budget_features.jsonl"
    position_feature_path = output_root / "position_budget_features.jsonl"
    visits_values = [int(value) for value in args.visits.split(",") if value.strip()]
    selected_records, skipped = load_selected_records(DEFAULT_PATHS, args.d4_limit, args.d5_limit)
    shard_selected_records = shard_records(selected_records, args.shard_index, args.shard_count)
    existing = existing_query_ids(query_feature_path)
    query_metadata = []
    for record in shard_selected_records:
        for visits in visits_values:
            query, metadata = build_budget_query(record, visits)
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
    expected_query_count = len(shard_selected_records) * len(visits_values)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R016 budget-generalization KataGo query extraction",
        "status": "done" if len(query_rows) == expected_query_count and len(aggregated) == len(shard_selected_records) else "running_or_incomplete",
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
        "position_feature_count": len(aggregated),
        "skipped_position_count": len(skipped),
        "oracle_boundary": "8192 visits is a stronger same-engine relative oracle only, not ground truth.",
    }
    write_json(output_root / "r016_budget_generalization_query_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R016_BUDGET_QUERIES")
    parser.add_argument("--katago-exe", default="external/engines/katago/v1.16.5-opencl/katago.exe")
    parser.add_argument("--config", default="external/engines/katago/v1.16.5-opencl/analysis_example.cfg")
    parser.add_argument("--model", default="external/engines/downloads/networks/katago/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz")
    parser.add_argument("--visits", default="1024,8192")
    parser.add_argument("--d4-limit", type=int, default=1000)
    parser.add_argument("--d5-limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-root", default="experiments/results/R016_BUDGET_GENERALIZATION_QUERIES")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
