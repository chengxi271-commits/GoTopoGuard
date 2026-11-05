#!/usr/bin/env python
"""Evaluate normal-position guard regressions on R015 D1/D2 budget data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import time
from collections import Counter, defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")
R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R013_OP = load_script_module("r013_operating_point_tradeoff", SCRIPT_DIR / "r013_operating_point_tradeoff.py")
R014 = load_script_module("r014_guard_equal_compute_pilot", SCRIPT_DIR / "r014_guard_equal_compute_pilot.py")
R014B = load_script_module("r014b_utility_guard_redesign", SCRIPT_DIR / "r014b_utility_guard_redesign.py")
R014C = load_script_module("r014c_utility_guard_model_sweep", SCRIPT_DIR / "r014c_utility_guard_model_sweep.py")


RISK_GUARDS = {
    "G3_topology_fpr05": ("B3_topology_gbdt", 0.05),
    "G4_topology_search_metamorphic_fpr05": ("B5_topology_plus_search_plus_metamorphic_gbdt", 0.05),
}

S09_CONFIG = {
    "name": "S09_b3_extra_p1p5",
    "feature_profile": "B3",
    "model_kind": "extra",
    "penalty": 1.5,
    "max_trigger_rate": None,
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


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def method_top_move(row: dict, triggered: bool) -> str | None:
    return row.get("v1024_top_move") if triggered else row.get("v256_top_move")


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
        "normal_correction_count": len(corrections),
        "normal_correction_rate_on_g0_errors": len(corrections) / len(g0_errors) if g0_errors else None,
        "normal_regression_count": len(regressions),
        "normal_regression_rate_on_g0_correct": len(regressions) / len(g0_correct) if g0_correct else None,
        "net_normal_gain": len(corrections) - len(regressions),
        "trigger_count": len(triggers),
        "trigger_rate": len(triggers) / len(rows) if rows else None,
        "mean_visits": mean(visits),
        "total_visits": sum(visits),
    }


def attach_normal_budget_features(feature_rows: list[dict], budget_rows: list[dict]) -> list[dict]:
    budget_map = {row["position_id"]: row for row in budget_rows}
    rows = []
    for row in feature_rows:
        if row.get("dataset") not in {"D1", "D2"}:
            continue
        budget = budget_map.get(row["position_id"])
        if not budget:
            continue
        merged = {**row, **budget}
        merged["oracle_top_move"] = merged.get("v4096_top_move")
        merged["g0_top_move"] = merged.get("v256_top_move")
        if merged["oracle_top_move"] is not None and merged["g0_top_move"] is not None:
            rows.append(merged)
    return rows


def build_baseline_records(rows: list[dict], method: str) -> list[dict]:
    records = []
    for row in rows:
        if method == "G0_unguarded_256":
            triggered = False
            visits = 256
        elif method == "G1_uniform_1024":
            triggered = True
            visits = 1024
        else:
            raise ValueError(method)
        records.append(
            {
                "position_id": row["position_id"],
                "dataset": row["dataset"],
                "family": row["family"],
                "method": method,
                "oracle_top_move": row["oracle_top_move"],
                "g0_top_move": row["g0_top_move"],
                "method_top_move": method_top_move(row, triggered),
                "method_visits": visits,
                "triggered": triggered,
            }
        )
    return records


def stable_natural_fold(row: dict, fold_count: int = 5) -> int:
    signature = row.get("position_signature")
    if signature:
        token = str(signature)[:8]
    else:
        token = hashlib.sha256(str(row["position_id"]).encode("utf-8")).hexdigest()[:8]
    return int(token, 16) % fold_count


def risk_trigger_records(all_feature_rows: list[dict], normal_rows: list[dict], method: str, detector: str, fpr_cap: float) -> list[dict]:
    feature_names = R012.feature_names_for_baseline(detector)
    records = []
    fold_count = 5
    for fold in range(fold_count):
        test_ids = {row["position_id"] for row in normal_rows if stable_natural_fold(row, fold_count) == fold}
        if not test_ids:
            continue
        train = []
        test = []
        for row in all_feature_rows:
            if row.get("dataset") in {"D1", "D2"} and stable_natural_fold(row, fold_count) == fold:
                if row["position_id"] in test_ids:
                    test.append(row)
            else:
                train.append(row)
        model, filtered_train, _y_train, _train_scores = R012.fit_model(train, detector, feature_names)
        train_natural = [row for row in filtered_train if row["label"] == 0 and row["dataset"] in {"D1", "D2"}]
        scored_train_natural = R012.score_rows(model, train_natural, feature_names)
        threshold = R013_OP.threshold_at_fpr_cap([row["score"] for row in scored_train_natural], fpr_cap)
        scored_test = {row["position_id"]: row for row in R012.score_rows(model, test, feature_names)}
        normal_by_id = {row["position_id"]: row for row in normal_rows if row["position_id"] in test_ids}
        for position_id, row in normal_by_id.items():
            score = scored_test.get(position_id, {}).get("score")
            triggered = bool(score is not None and score >= threshold)
            records.append(
                {
                    "position_id": row["position_id"],
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "method": method,
                    "oracle_top_move": row["oracle_top_move"],
                    "g0_top_move": row["g0_top_move"],
                    "method_top_move": method_top_move(row, triggered),
                    "method_visits": 1024 if triggered else 256,
                    "triggered": triggered,
                    "score": score,
                    "threshold": threshold,
                    "fold": fold,
                    "detector_baseline": detector,
                    "fpr_cap": fpr_cap,
                }
            )
    return records


def train_s09_on_calibration(feature_rows: list[dict], r014_budget_rows: list[dict], d3_positions: pathlib.Path):
    calibration_rows = R014B.attach_budget_features(feature_rows, r014_budget_rows, d3_positions)
    calibration_rows = [row for row in calibration_rows if row["effect"] != "missing"]
    feature_names = R014C.feature_names_for_profile(S09_CONFIG["feature_profile"])
    corr_model, usable_train = R014C.fit_binary_model(
        calibration_rows,
        feature_names,
        "correction",
        S09_CONFIG["model_kind"],
        R014C.stable_seed("R015", S09_CONFIG["name"], "correction"),
    )
    reg_model, _usable_reg_train = R014C.fit_binary_model(
        calibration_rows,
        feature_names,
        "regression",
        S09_CONFIG["model_kind"],
        R014C.stable_seed("R015", S09_CONFIG["name"], "regression"),
    )
    scored_train = R014C.score_rows(corr_model, reg_model, usable_train, feature_names, S09_CONFIG["penalty"])
    selected = R014B.select_threshold_by_training_utility(scored_train, S09_CONFIG["max_trigger_rate"])
    return corr_model, reg_model, feature_names, selected, len(calibration_rows)


def s09_records(feature_rows: list[dict], normal_rows: list[dict], r014_budget_rows: list[dict], d3_positions: pathlib.Path) -> tuple[list[dict], dict]:
    corr_model, reg_model, feature_names, selected, calibration_count = train_s09_on_calibration(feature_rows, r014_budget_rows, d3_positions)
    normal_by_id = {row["position_id"]: row for row in normal_rows}
    scored = R014C.score_rows(corr_model, reg_model, list(normal_by_id.values()), feature_names, S09_CONFIG["penalty"])
    records = []
    for scored_row in scored:
        row = normal_by_id[scored_row["position_id"]]
        triggered = scored_row["score"] >= selected["threshold"]
        records.append(
            {
                "position_id": row["position_id"],
                "dataset": row["dataset"],
                "family": row["family"],
                "method": "S09_utility_b3_extra_p1p5",
                "oracle_top_move": row["oracle_top_move"],
                "g0_top_move": row["g0_top_move"],
                "method_top_move": method_top_move(row, triggered),
                "method_visits": 1024 if triggered else 256,
                "triggered": triggered,
                "score": scored_row["score"],
                "p_correction": scored_row["p_correction"],
                "p_regression": scored_row["p_regression"],
                "threshold": selected["threshold"],
            }
        )
    meta = {
        "s09_calibration_count": calibration_count,
        "s09_training_oracle_agreement_rate": selected["oracle_agreement_rate"],
        "s09_training_trigger_rate": selected["trigger_rate"],
        "s09_threshold": selected["threshold"],
    }
    return records, meta


def summarize_records(records: list[dict]) -> list[dict]:
    summaries = []
    keys = sorted({(row["method"], row["dataset"]) for row in records}, key=lambda item: (item[0], item[1]))
    for method, dataset in keys:
        subset = [row for row in records if row["method"] == method and row["dataset"] == dataset]
        summaries.append({"method": method, "dataset": dataset, **summarize_method(subset)})
    for method in sorted({row["method"] for row in records}):
        subset = [row for row in records if row["method"] == method]
        summaries.append({"method": method, "dataset": "D1+D2", **summarize_method(subset)})
    return summaries


def run(args: argparse.Namespace) -> int:
    feature_rows = read_jsonl(pathlib.Path(args.feature_rows))
    normal_budget_rows = read_jsonl(pathlib.Path(args.normal_budget_features))
    r014_budget_rows = read_jsonl(pathlib.Path(args.r014_budget_features))
    normal_rows = attach_normal_budget_features(feature_rows, normal_budget_rows)
    records = []
    records.extend(build_baseline_records(normal_rows, "G0_unguarded_256"))
    records.extend(build_baseline_records(normal_rows, "G1_uniform_1024"))
    for method, (detector, fpr_cap) in RISK_GUARDS.items():
        records.extend(risk_trigger_records(feature_rows, normal_rows, method, detector, fpr_cap))
    s09_method_records, s09_meta = s09_records(feature_rows, normal_rows, r014_budget_rows, pathlib.Path(args.d3_positions))
    records.extend(s09_method_records)
    summaries = summarize_records(records)
    expected_positions = int(args.expected_positions) if args.expected_positions else None
    status = "done"
    if expected_positions is not None and len(normal_rows) != expected_positions:
        status = "running_or_incomplete"
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R015 normal-position guard regression evaluation using 4096-visit same-engine relative oracle",
        "status": status if records else "no_records",
        "position_count": len(normal_rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in normal_rows)),
        "methods": sorted({row["method"] for row in records}),
        "summary_count": len(summaries),
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth.",
        **s09_meta,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "normal_guard_method_records.jsonl", records)
    write_jsonl(output_root / "normal_guard_summary_records.jsonl", summaries)
    write_json(output_root / "r015_normal_guard_regression_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R015")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--normal-budget-features", default="experiments/results/R015_NORMAL_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--r014-budget-features", default="experiments/results/R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument("--output-root", default="experiments/results/R015_NORMAL_GUARD_REGRESSION")
    parser.add_argument("--expected-positions", type=int, default=6781)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
