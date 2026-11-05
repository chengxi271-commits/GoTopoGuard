#!/usr/bin/env python
"""Analyze R017 alternate-network agreement against b28 1024-visit output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from collections import defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R016Q = load_script_module("r016_budget_generalization_queries", SCRIPT_DIR / "r016_budget_generalization_queries.py")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return R016Q.read_jsonl(path)


def write_json(path: pathlib.Path, payload) -> None:
    return R016Q.write_json(path, payload)


def write_jsonl(path: pathlib.Path, rows) -> None:
    return R016Q.write_jsonl(path, rows)


def as_float(value) -> float:
    return float(value if value is not None else 0.0)


def network_stability_record(base: dict, alt: dict) -> dict:
    return {
        "position_id": alt["position_id"],
        "dataset": alt["dataset"],
        "family": alt["family"],
        "network_alias": alt["network_alias"],
        "b28_1024_top_move": base.get("v1024_top_move"),
        "network_top_move": alt.get("top_move"),
        "top_move_agrees_with_b28_1024": int(base.get("v1024_top_move") == alt.get("top_move")),
        "score_abs_delta_vs_b28_1024": abs(as_float(base.get("v1024_scoreLead")) - as_float(alt.get("scoreLead"))),
        "winrate_abs_delta_vs_b28_1024": abs(as_float(base.get("v1024_winrate")) - as_float(alt.get("winrate"))),
    }


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_one(name: str, group: str, rows: list[dict]) -> dict:
    return {
        "group": group,
        "name": name,
        "position_count": len(rows),
        "top_move_agreement_rate_vs_b28_1024": mean([float(row["top_move_agrees_with_b28_1024"]) for row in rows]),
        "top_move_disagreement_rate_vs_b28_1024": mean([1.0 - float(row["top_move_agrees_with_b28_1024"]) for row in rows]),
        "mean_score_abs_delta_vs_b28_1024": mean([float(row["score_abs_delta_vs_b28_1024"]) for row in rows]),
        "mean_winrate_abs_delta_vs_b28_1024": mean([float(row["winrate_abs_delta_vs_b28_1024"]) for row in rows]),
    }


def summarize_records(records: list[dict]) -> list[dict]:
    summaries = []
    by_network = defaultdict(list)
    by_network_dataset = defaultdict(list)
    by_network_family = defaultdict(list)
    for row in records:
        by_network[row["network_alias"]].append(row)
        by_network_dataset[f"{row['network_alias']}/{row['dataset']}"].append(row)
        by_network_family[f"{row['network_alias']}/{row['dataset']}/{row['family']}"].append(row)
    for name, rows in sorted(by_network.items()):
        summaries.append(summarize_one(name, "network", rows))
    for name, rows in sorted(by_network_dataset.items()):
        summaries.append(summarize_one(name, "network_dataset", rows))
    for name, rows in sorted(by_network_family.items()):
        summaries.append(summarize_one(name, "network_family", rows))
    return summaries


def run(args: argparse.Namespace) -> int:
    base_rows = {row["position_id"]: row for row in read_jsonl(pathlib.Path(args.base_budget_features))}
    alt_rows = read_jsonl(pathlib.Path(args.network_features))
    records = []
    for alt in alt_rows:
        base = base_rows.get(alt["position_id"])
        if not base or base.get("v1024_top_move") is None:
            continue
        records.append(network_stability_record(base, alt))
    summaries = summarize_records(records)
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "network_stability_records.jsonl", records)
    write_jsonl(output_root / "network_stability_summary.jsonl", summaries)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R017 network-generalization analysis against b28 1024-visit reference",
        "status": "done" if records and (not args.expected_records or len(records) == args.expected_records) else "incomplete",
        "record_count": len(records),
        "expected_record_count": args.expected_records,
        "network_counts": {},
        "reference_boundary": "b28 1024 visits is a same-engine/network comparison anchor, not ground truth.",
    }
    for row in records:
        report["network_counts"][row["network_alias"]] = report["network_counts"].get(row["network_alias"], 0) + 1
    write_json(output_root / "r017_network_generalization_analysis_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R017_NETWORK_ANALYSIS")
    parser.add_argument("--base-budget-features", default="experiments/results/R016_BUDGET_GENERALIZATION_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--network-features", default="experiments/results/R017_NETWORK_GENERALIZATION_QUERIES/query_network_features.jsonl")
    parser.add_argument("--output-root", default="experiments/results/R017_NETWORK_GENERALIZATION_ANALYSIS")
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
