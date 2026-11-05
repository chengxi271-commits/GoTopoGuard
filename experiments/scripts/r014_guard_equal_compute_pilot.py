#!/usr/bin/env python
"""Evaluate R014 guard equal-compute pilot on D3/D5 positions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import time
from collections import defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R013_OP = load_script_module("r013_operating_point_tradeoff", SCRIPT_DIR / "r013_operating_point_tradeoff.py")


GUARD_DETECTORS = {
    "G2_confidence": "B1_engine_confidence_logreg",
    "G3_topology": "B3_topology_gbdt",
    "G4_topology_search_metamorphic": "B5_topology_plus_search_plus_metamorphic_gbdt",
}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def guard_top_move(row: dict, *, triggered: bool) -> str | None:
    return row.get("v1024_top_move") if triggered else row.get("v256_top_move")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_method(rows: list[dict]) -> dict:
    g0_errors = [row for row in rows if row["g0_top_move"] != row["oracle_top_move"]]
    g0_correct = [row for row in rows if row["g0_top_move"] == row["oracle_top_move"]]
    corrections = [row for row in g0_errors if row["method_top_move"] == row["oracle_top_move"]]
    regressions = [row for row in g0_correct if row["method_top_move"] != row["oracle_top_move"]]
    agreements = [row for row in rows if row["method_top_move"] == row["oracle_top_move"]]
    triggers = [row for row in rows if row.get("triggered")]
    visits = [float(row["method_visits"]) for row in rows]
    return {
        "position_count": len(rows),
        "oracle_agreement_count": len(agreements),
        "oracle_agreement_rate": len(agreements) / len(rows) if rows else None,
        "g0_error_count": len(g0_errors),
        "correction_count": len(corrections),
        "correction_rate_on_g0_errors": len(corrections) / len(g0_errors) if g0_errors else None,
        "regression_count": len(regressions),
        "regression_rate_on_g0_correct": len(regressions) / len(g0_correct) if g0_correct else None,
        "trigger_count": len(triggers),
        "trigger_rate": len(triggers) / len(rows) if rows else None,
        "mean_visits": mean(visits),
        "total_visits": sum(visits),
    }


def attach_budget_features(rows: list[dict], budget_rows: list[dict]) -> list[dict]:
    budget_map = {row["position_id"]: row for row in budget_rows}
    merged = []
    for row in rows:
        if row["position_id"] not in budget_map:
            continue
        merged.append({**row, **budget_map[row["position_id"]]})
    return merged


def load_guard_rows(feature_rows_path: pathlib.Path, budget_features_path: pathlib.Path, d3_positions_path: pathlib.Path) -> list[dict]:
    feature_rows = read_jsonl(feature_rows_path)
    budget_rows = read_jsonl(budget_features_path)
    d3_metadata = R012.load_d3_metadata(d3_positions_path)
    feature_rows = R012.add_d3_metadata(feature_rows, d3_metadata)
    rows = [row for row in feature_rows if row["dataset"] in {"D3", "D5"}]
    rows = attach_budget_features(rows, budget_rows)
    for row in rows:
        row["oracle_top_move"] = row.get("v4096_top_move")
        row["g0_top_move"] = row.get("v256_top_move")
    return rows


def train_group_trigger_scores(rows: list[dict], detector: str, fpr_cap: float, target_rows: list[dict]) -> dict[str, dict]:
    feature_names = R012.feature_names_for_baseline(detector)
    trigger_info = {}
    d3_game_indices = sorted({int(row["game_index"]) for row in target_rows if row["dataset"] == "D3"})
    d5_families = sorted({row["family"] for row in target_rows if row["dataset"] == "D5"})
    groups = [("D3", game_index) for game_index in d3_game_indices] + [("D5", family) for family in d5_families]
    for group_type, group_value in groups:
        train = []
        test = []
        for row in rows:
            if row["dataset"] == "D3" and group_type == "D3" and int(row.get("game_index", -1)) == int(group_value):
                test.append(row)
            elif row["dataset"] == "D5" and group_type == "D5" and row["family"] == group_value:
                test.append(row)
            else:
                train.append(row)
        model, filtered_train, _y_train, _train_scores = R012.fit_model(train, detector, feature_names)
        train_natural = [row for row in filtered_train if row["label"] == 0 and row["dataset"] in {"D1", "D2"}]
        scored_train_natural = R012.score_rows(model, train_natural, feature_names)
        threshold = R013_OP.threshold_at_fpr_cap([row["score"] for row in scored_train_natural], fpr_cap)
        scored_test = R012.score_rows(model, test, feature_names)
        for row in scored_test:
            trigger_info[row["position_id"]] = {
                "score": row["score"],
                "threshold": threshold,
                "triggered": row["score"] >= threshold,
                "trigger_group_type": group_type,
                "trigger_group_value": group_value,
                "detector_baseline": detector,
                "fpr_cap": fpr_cap,
            }
    return trigger_info


def build_method_rows(target_rows: list[dict], *, method: str, fpr_cap: float | None = None, trigger_info: dict[str, dict] | None = None) -> list[dict]:
    out = []
    for row in target_rows:
        if method == "G0_unguarded_256":
            method_top = row["v256_top_move"]
            triggered = False
            visits = 256
        elif method == "G1_uniform_1024":
            method_top = row["v1024_top_move"]
            triggered = True
            visits = 1024
        else:
            info = trigger_info.get(row["position_id"], {}) if trigger_info else {}
            triggered = bool(info.get("triggered"))
            method_top = guard_top_move(row, triggered=triggered)
            visits = 1024 if triggered else 256
        out.append(
            {
                "position_id": row["position_id"],
                "dataset": row["dataset"],
                "family": row["family"],
                "method": method,
                "fpr_cap": fpr_cap,
                "oracle_top_move": row["oracle_top_move"],
                "g0_top_move": row["g0_top_move"],
                "method_top_move": method_top,
                "method_visits": visits,
                "triggered": triggered,
            }
        )
    return out


def run(args: argparse.Namespace) -> int:
    d3_metadata = R012.load_d3_metadata(pathlib.Path(args.d3_positions))
    rows = R012.add_d3_metadata(read_jsonl(pathlib.Path(args.feature_rows)), d3_metadata)
    guard_rows = load_guard_rows(
        pathlib.Path(args.feature_rows),
        pathlib.Path(args.budget_features),
        pathlib.Path(args.d3_positions),
    )
    fpr_caps = [float(value) for value in args.fpr_caps.split(",") if value.strip()]
    target_datasets = {value for value in args.datasets.split(",") if value.strip()}
    target_rows = [row for row in guard_rows if row["dataset"] in target_datasets]
    method_records = []
    method_records.extend(build_method_rows(target_rows, method="G0_unguarded_256"))
    method_records.extend(build_method_rows(target_rows, method="G1_uniform_1024"))
    for fpr_cap in fpr_caps:
        for method, detector in GUARD_DETECTORS.items():
            trigger_info = train_group_trigger_scores(rows, detector, fpr_cap, target_rows)
            method_records.extend(build_method_rows(target_rows, method=method, fpr_cap=fpr_cap, trigger_info=trigger_info))
    summaries = []
    for key in sorted({(row["method"], row.get("fpr_cap"), row["dataset"]) for row in method_records}, key=lambda item: (item[0], str(item[1]), item[2])):
        method, fpr_cap, dataset = key
        subset = [row for row in method_records if row["method"] == method and row.get("fpr_cap") == fpr_cap and row["dataset"] == dataset]
        summaries.append({"method": method, "fpr_cap": fpr_cap, "dataset": dataset, **summarize_method(subset)})
    for key in sorted({(row["method"], row.get("fpr_cap")) for row in method_records}, key=lambda item: (item[0], str(item[1]))):
        method, fpr_cap = key
        subset = [row for row in method_records if row["method"] == method and row.get("fpr_cap") == fpr_cap]
        summaries.append({"method": method, "fpr_cap": fpr_cap, "dataset": "D3+D5", **summarize_method(subset)})
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R014 guard equal-compute pilot using 4096-visit same-engine relative oracle",
        "status": "done" if method_records else "no_records",
        "position_count": len(target_rows),
        "dataset_counts": {
            "D3": sum(1 for row in target_rows if row["dataset"] == "D3"),
            "D5": sum(1 for row in target_rows if row["dataset"] == "D5"),
        },
        "fpr_caps": fpr_caps,
        "guard_detectors": GUARD_DETECTORS,
        "baseline_boundary": "G1_uniform_1024 is a stronger uniform-extra-visits baseline and may spend more visits than sparse guards; total visits are reported for every method.",
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth.",
        "summary_count": len(summaries),
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "guard_method_records.jsonl", method_records)
    write_jsonl(output_root / "guard_summary_records.jsonl", summaries)
    write_json(output_root / "r014_guard_equal_compute_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R014")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--budget-features", default="experiments/results/R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument("--fpr-caps", default="0.01,0.05,0.1")
    parser.add_argument("--datasets", default="D3,D5")
    parser.add_argument("--output-root", default="experiments/results/R014_GUARD_EQUAL_COMPUTE_PILOT")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
