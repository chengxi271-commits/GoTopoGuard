#!/usr/bin/env python
"""Analyze R016 budget-generalization curves against 8192 visits."""

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

BUDGETS = [64, 256, 1024, 8192]
REFERENCE = 8192


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return R016Q.read_jsonl(path)


def write_json(path: pathlib.Path, payload) -> None:
    return R016Q.write_json(path, payload)


def write_jsonl(path: pathlib.Path, rows) -> None:
    return R016Q.write_jsonl(path, rows)


def feature_map(rows: list[dict]) -> dict[str, dict]:
    return {row["position_id"]: row for row in rows}


def merge_budget_rows(feature_rows: list[dict], low_budget_rows: list[dict], high_budget_rows: list[dict]) -> list[dict]:
    features = feature_map(feature_rows)
    low = feature_map(low_budget_rows)
    high = feature_map(high_budget_rows)
    rows = []
    for position_id, high_row in sorted(high.items()):
        low_row = low.get(position_id)
        feature_row = features.get(position_id)
        if not low_row or not feature_row:
            continue
        merged = {**feature_row}
        for source in [low_row, high_row]:
            for key, value in source.items():
                if key != "position_id":
                    merged[key] = value
        rows.append(merged)
    return rows


def required_budget_fields_available(row: dict) -> bool:
    for visits in BUDGETS:
        for key in ["top_move", "scoreLead", "winrate"]:
            if row.get(f"v{visits}_{key}") is None:
                return False
    return True


def as_float(value) -> float:
    return float(value if value is not None else 0.0)


def budget_stability_record(row: dict) -> dict:
    out = {
        "position_id": row["position_id"],
        "dataset": row["dataset"],
        "family": row["family"],
    }
    for visits in [64, 256, 1024]:
        out[f"agree_{visits}_{REFERENCE}"] = int(row.get(f"v{visits}_top_move") == row.get(f"v{REFERENCE}_top_move"))
        out[f"disagree_{visits}_{REFERENCE}"] = 1 - out[f"agree_{visits}_{REFERENCE}"]
        out[f"score_abs_delta_{visits}_{REFERENCE}"] = abs(as_float(row.get(f"v{visits}_scoreLead")) - as_float(row.get(f"v{REFERENCE}_scoreLead")))
        out[f"winrate_abs_delta_{visits}_{REFERENCE}"] = abs(as_float(row.get(f"v{visits}_winrate")) - as_float(row.get(f"v{REFERENCE}_winrate")))
    for low, high in [(64, 256), (256, 1024), (1024, 8192)]:
        out[f"agree_{low}_{high}"] = int(row.get(f"v{low}_top_move") == row.get(f"v{high}_top_move"))
        out[f"disagree_{low}_{high}"] = 1 - out[f"agree_{low}_{high}"]
        out[f"score_abs_delta_{low}_{high}"] = abs(as_float(row.get(f"v{low}_scoreLead")) - as_float(row.get(f"v{high}_scoreLead")))
        out[f"winrate_abs_delta_{low}_{high}"] = abs(as_float(row.get(f"v{low}_winrate")) - as_float(row.get(f"v{high}_winrate")))
    for key in [
        "black_group_count",
        "white_group_count",
        "empty_region_count",
        "enclosed_empty_region_count",
        "black_articulation_count",
        "white_articulation_count",
        "black_bounded_empty_region_count",
        "white_bounded_empty_region_count",
    ]:
        if key in row:
            out[key] = row[key]
    return out


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_one(name: str, group: str, rows: list[dict]) -> dict:
    out = {"group": group, "name": name, "position_count": len(rows)}
    for low, high in [(64, 256), (256, 1024), (1024, 8192), (64, 8192), (256, 8192)]:
        agree_key = f"agree_{low}_{high}"
        disagree_key = f"disagree_{low}_{high}"
        score_key = f"score_abs_delta_{low}_{high}"
        winrate_key = f"winrate_abs_delta_{low}_{high}"
        if agree_key in rows[0]:
            out[f"agreement_rate_{low}_{high}"] = mean([float(row[agree_key]) for row in rows])
        if disagree_key in rows[0]:
            out[f"disagreement_rate_{low}_{high}"] = mean([float(row[disagree_key]) for row in rows])
        if score_key in rows[0]:
            out[f"mean_score_abs_delta_{low}_{high}"] = mean([float(row[score_key]) for row in rows])
        if winrate_key in rows[0]:
            out[f"mean_winrate_abs_delta_{low}_{high}"] = mean([float(row[winrate_key]) for row in rows])
    return out


def summarize_records(records: list[dict]) -> list[dict]:
    summaries = []
    by_dataset = defaultdict(list)
    by_family = defaultdict(list)
    for row in records:
        by_dataset[row["dataset"]].append(row)
        by_family[f"{row['dataset']}/{row['family']}"].append(row)
    for dataset, rows in sorted(by_dataset.items()):
        summaries.append(summarize_one(dataset, "dataset", rows))
    for family, rows in sorted(by_family.items()):
        summaries.append(summarize_one(family, "family", rows))
    if records:
        summaries.append(summarize_one("ALL", "all", records))
    return summaries


def run(args: argparse.Namespace) -> int:
    feature_rows = read_jsonl(pathlib.Path(args.feature_rows))
    low_budget_rows = read_jsonl(pathlib.Path(args.low_budget_features))
    high_budget_rows = read_jsonl(pathlib.Path(args.high_budget_features))
    merged = merge_budget_rows(feature_rows, low_budget_rows, high_budget_rows)
    complete_rows = [row for row in merged if required_budget_fields_available(row)]
    records = [budget_stability_record(row) for row in complete_rows]
    summaries = summarize_records(records)
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "budget_stability_records.jsonl", records)
    write_jsonl(output_root / "budget_stability_summary.jsonl", summaries)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R016 budget-generalization analysis using 8192-visit same-engine relative oracle",
        "status": "done" if records and (not args.expected_positions or len(records) == args.expected_positions) else "incomplete",
        "position_count": len(records),
        "expected_position_count": args.expected_positions,
        "dataset_counts": {dataset: sum(1 for row in records if row["dataset"] == dataset) for dataset in ["D3", "D4", "D5"]},
        "budgets": BUDGETS,
        "reference_budget": REFERENCE,
        "oracle_boundary": "8192 visits is a stronger same-engine relative oracle only, not ground truth.",
    }
    write_json(output_root / "r016_budget_generalization_analysis_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R016_BUDGET_ANALYSIS")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--low-budget-features", default="experiments/results/R011_KATAGO_SEARCH_FEATURES/position_search_features.jsonl")
    parser.add_argument("--high-budget-features", default="experiments/results/R016_BUDGET_GENERALIZATION_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--output-root", default="experiments/results/R016_BUDGET_GENERALIZATION_ANALYSIS")
    parser.add_argument("--expected-positions", type=int, default=0)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
