#!/usr/bin/env python
"""R014B utility-calibrated guard redesign.

This run reuses R014 1024/4096-visit budget features and asks a sharper
question than R014: can a guard learn when extra search is useful, rather than
triggering on generic risk?  The 4096-visit move remains a same-engine relative
oracle only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import time
from collections import Counter, defaultdict

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")
R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R014 = load_script_module("r014_guard_equal_compute_pilot", SCRIPT_DIR / "r014_guard_equal_compute_pilot.py")


UTILITY_GUARDS = {
    "U1_utility_b3_correction": {
        "feature_baseline": "B3_topology_gbdt",
        "mode": "correction_only",
        "penalty": 0.0,
        "max_trigger_rate": None,
    },
    "U2_utility_b5_correction": {
        "feature_baseline": "B5_topology_plus_search_plus_metamorphic_gbdt",
        "mode": "correction_only",
        "penalty": 0.0,
        "max_trigger_rate": None,
    },
    "U3_utility_b3_corr_minus_reg": {
        "feature_baseline": "B3_topology_gbdt",
        "mode": "correction_minus_regression",
        "penalty": 1.0,
        "max_trigger_rate": None,
    },
    "U4_utility_b5_corr_minus_reg": {
        "feature_baseline": "B5_topology_plus_search_plus_metamorphic_gbdt",
        "mode": "correction_minus_regression",
        "penalty": 1.0,
        "max_trigger_rate": None,
    },
    "U5_utility_b5_regression_averse": {
        "feature_baseline": "B5_topology_plus_search_plus_metamorphic_gbdt",
        "mode": "correction_minus_regression",
        "penalty": 2.0,
        "max_trigger_rate": None,
    },
    "U6_utility_b5_cap75": {
        "feature_baseline": "B5_topology_plus_search_plus_metamorphic_gbdt",
        "mode": "correction_minus_regression",
        "penalty": 1.0,
        "max_trigger_rate": 0.75,
    },
    "U7_utility_b5_cap50": {
        "feature_baseline": "B5_topology_plus_search_plus_metamorphic_gbdt",
        "mode": "correction_minus_regression",
        "penalty": 1.0,
        "max_trigger_rate": 0.50,
    },
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


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def extra_search_effect(row: dict) -> str:
    low = row.get("v256_top_move")
    high = row.get("v1024_top_move")
    oracle = row.get("v4096_top_move")
    if low is None or high is None or oracle is None:
        return "missing"
    if low != oracle and high == oracle:
        return "correction"
    if low == oracle and high != oracle:
        return "regression"
    if low == oracle and high == oracle:
        return "preserved_correct"
    return "unresolved_error"


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
        "net_correction_gain": len(corrections) - len(regressions),
        "trigger_count": len(triggers),
        "trigger_rate": len(triggers) / len(rows) if rows else None,
        "mean_visits": mean(visits),
        "total_visits": sum(visits),
    }


def agreement_if_triggered(row: dict, triggered: bool) -> int:
    if "v256_top_move" not in row or "v1024_top_move" not in row or "v4096_top_move" not in row:
        effect = row.get("effect")
        if triggered:
            return int(effect in {"correction", "preserved_correct"})
        return int(effect in {"regression", "preserved_correct"})
    chosen = row["v1024_top_move"] if triggered else row["v256_top_move"]
    return int(chosen == row["v4096_top_move"])


def select_threshold_by_training_utility(scored_rows: list[dict], max_trigger_rate: float | None) -> dict:
    if not scored_rows:
        return {
            "threshold": math.inf,
            "oracle_agreement_rate": None,
            "correction_count": 0,
            "regression_count": 0,
            "trigger_rate": 0.0,
        }
    thresholds = sorted({float(row["score"]) for row in scored_rows}, reverse=True)
    thresholds.append(math.inf)
    best = None
    for threshold in thresholds:
        triggered_rows = [row for row in scored_rows if float(row["score"]) >= threshold]
        trigger_rate = len(triggered_rows) / len(scored_rows)
        if max_trigger_rate is not None and trigger_rate > max_trigger_rate:
            continue
        agreements = sum(agreement_if_triggered(row, float(row["score"]) >= threshold) for row in scored_rows)
        corrections = sum(1 for row in triggered_rows if row["effect"] == "correction")
        regressions = sum(1 for row in triggered_rows if row["effect"] == "regression")
        candidate = {
            "threshold": float(threshold),
            "oracle_agreement_rate": agreements / len(scored_rows),
            "correction_count": corrections,
            "regression_count": regressions,
            "net_correction_gain": corrections - regressions,
            "trigger_rate": trigger_rate,
        }
        rank = (
            candidate["oracle_agreement_rate"],
            candidate["net_correction_gain"],
            -candidate["trigger_rate"],
            -candidate["regression_count"],
            candidate["correction_count"],
        )
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return best[1] if best else select_threshold_by_training_utility(scored_rows, None)


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x):
        n = len(x)
        p = self.probability
        return np.array([[1.0 - p, p] for _ in range(n)], dtype=float)


def fit_binary_model(rows: list[dict], feature_names: list[str], positive_label: str):
    usable = [row for row in rows if R011.features_available(row, feature_names)]
    y = [1 if row["effect"] == positive_label else 0 for row in usable]
    if not usable:
        return ConstantProbabilityModel(0.0), []
    if len(set(y)) < 2:
        return ConstantProbabilityModel(sum(y) / len(y)), usable
    x, _unused = R011.matrix([{**row, "label": label} for row, label in zip(usable, y)], feature_names)
    model = R011.GradientBoostingClassifier(random_state=0)
    model.fit(x, np.array(y, dtype=int))
    return model, usable


def score_utility_rows(corr_model, reg_model, rows: list[dict], feature_names: list[str], mode: str, penalty: float) -> list[dict]:
    usable = [row for row in rows if R011.features_available(row, feature_names)]
    if not usable:
        return []
    x, _unused = R011.matrix([{**row, "label": 0} for row in usable], feature_names)
    corr_p = corr_model.predict_proba(x)[:, 1]
    if mode == "correction_only":
        reg_p = np.zeros_like(corr_p)
        scores = corr_p
    elif mode == "correction_minus_regression":
        reg_p = reg_model.predict_proba(x)[:, 1]
        scores = corr_p - penalty * reg_p
    else:
        raise ValueError(f"unsupported utility scoring mode: {mode}")
    return [
        {
            **row,
            "p_correction": float(cp),
            "p_regression": float(rp),
            "score": float(score),
        }
        for row, cp, rp, score in zip(usable, corr_p, reg_p, scores)
    ]


def attach_budget_features(feature_rows: list[dict], budget_rows: list[dict], d3_positions: pathlib.Path) -> list[dict]:
    budget_map = {row["position_id"]: row for row in budget_rows}
    d3_metadata = R012.load_d3_metadata(d3_positions)
    rows = R012.add_d3_metadata(feature_rows, d3_metadata)
    out = []
    for row in rows:
        if row["dataset"] not in {"D3", "D5"}:
            continue
        budget = budget_map.get(row["position_id"])
        if not budget:
            continue
        merged = {**row, **budget}
        merged["effect"] = extra_search_effect(merged)
        merged["g0_top_move"] = merged.get("v256_top_move")
        merged["oracle_top_move"] = merged.get("v4096_top_move")
        out.append(merged)
    return out


def heldout_groups(rows: list[dict]) -> list[tuple[str, int | str]]:
    d3_games = sorted({int(row["game_index"]) for row in rows if row["dataset"] == "D3"})
    d5_families = sorted({row["family"] for row in rows if row["dataset"] == "D5"})
    return [("D3", game_index) for game_index in d3_games] + [("D5", family) for family in d5_families]


def split_for_group(rows: list[dict], group_type: str, group_value: int | str) -> tuple[list[dict], list[dict]]:
    train = []
    test = []
    for row in rows:
        if row["dataset"] == "D3" and group_type == "D3" and int(row["game_index"]) == int(group_value):
            test.append(row)
        elif row["dataset"] == "D5" and group_type == "D5" and row["family"] == group_value:
            test.append(row)
        else:
            train.append(row)
    return train, test


def build_baseline_records(rows: list[dict], method: str) -> list[dict]:
    records = []
    for row in rows:
        if method == "G0_unguarded_256":
            method_top_move = row["v256_top_move"]
            triggered = False
            visits = 256
        elif method == "G1_uniform_1024":
            method_top_move = row["v1024_top_move"]
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
                "oracle_top_move": row["v4096_top_move"],
                "g0_top_move": row["v256_top_move"],
                "method_top_move": method_top_move,
                "method_visits": visits,
                "triggered": triggered,
                "effect": row["effect"],
            }
        )
    return records


def build_utility_records(rows: list[dict], method: str, config: dict) -> list[dict]:
    feature_names = R012.feature_names_for_baseline(config["feature_baseline"])
    records = []
    for group_type, group_value in heldout_groups(rows):
        train, test = split_for_group(rows, group_type, group_value)
        corr_model, usable_train = fit_binary_model(train, feature_names, "correction")
        reg_model, _usable_reg_train = fit_binary_model(train, feature_names, "regression")
        scored_train = score_utility_rows(corr_model, reg_model, usable_train, feature_names, config["mode"], config["penalty"])
        selected = select_threshold_by_training_utility(scored_train, config.get("max_trigger_rate"))
        scored_test = score_utility_rows(corr_model, reg_model, test, feature_names, config["mode"], config["penalty"])
        threshold = selected["threshold"]
        for row in scored_test:
            triggered = float(row["score"]) >= threshold
            records.append(
                {
                    "position_id": row["position_id"],
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "method": method,
                    "oracle_top_move": row["v4096_top_move"],
                    "g0_top_move": row["v256_top_move"],
                    "method_top_move": row["v1024_top_move"] if triggered else row["v256_top_move"],
                    "method_visits": 1024 if triggered else 256,
                    "triggered": triggered,
                    "effect": row["effect"],
                    "score": row["score"],
                    "p_correction": row["p_correction"],
                    "p_regression": row["p_regression"],
                    "threshold": threshold,
                    "train_oracle_agreement_rate": selected["oracle_agreement_rate"],
                    "train_trigger_rate": selected["trigger_rate"],
                    "heldout_group_type": group_type,
                    "heldout_group_value": group_value,
                    "feature_baseline": config["feature_baseline"],
                    "utility_mode": config["mode"],
                    "regression_penalty": config["penalty"],
                    "max_trigger_rate": config.get("max_trigger_rate"),
                }
            )
    return records


def summarize_records(records: list[dict]) -> list[dict]:
    summaries = []
    keys = sorted({(row["method"], row["dataset"]) for row in records}, key=lambda item: (item[0], item[1]))
    for method, dataset in keys:
        subset = [row for row in records if row["method"] == method and row["dataset"] == dataset]
        summaries.append({"method": method, "dataset": dataset, **summarize_method(subset)})
    for method in sorted({row["method"] for row in records}):
        subset = [row for row in records if row["method"] == method]
        summaries.append({"method": method, "dataset": "D3+D5", **summarize_method(subset)})
    return summaries


def analyze_against_uniform(summaries: list[dict]) -> dict:
    combined = {row["method"]: row for row in summaries if row["dataset"] == "D3+D5"}
    uniform = combined.get("G1_uniform_1024")
    best = None
    for method, row in combined.items():
        if not method.startswith("U"):
            continue
        rank = (
            row["oracle_agreement_rate"],
            row["net_correction_gain"],
            -row["mean_visits"],
            -row["trigger_rate"],
        )
        if best is None or rank > best[0]:
            best = (rank, method, row)
    if not uniform or not best:
        return {"uniform_available": bool(uniform), "best_utility_method": None}
    _, method, row = best
    return {
        "uniform_available": True,
        "uniform_oracle_agreement_rate": uniform["oracle_agreement_rate"],
        "uniform_mean_visits": uniform["mean_visits"],
        "best_utility_method": method,
        "best_utility_oracle_agreement_rate": row["oracle_agreement_rate"],
        "best_utility_mean_visits": row["mean_visits"],
        "best_utility_trigger_rate": row["trigger_rate"],
        "best_utility_net_correction_gain": row["net_correction_gain"],
        "beats_uniform_agreement": row["oracle_agreement_rate"] > uniform["oracle_agreement_rate"],
        "agreement_delta_vs_uniform": row["oracle_agreement_rate"] - uniform["oracle_agreement_rate"],
        "mean_visit_delta_vs_uniform": row["mean_visits"] - uniform["mean_visits"],
    }


def run(args: argparse.Namespace) -> int:
    feature_rows = read_jsonl(pathlib.Path(args.feature_rows))
    budget_rows = read_jsonl(pathlib.Path(args.budget_features))
    rows = attach_budget_features(feature_rows, budget_rows, pathlib.Path(args.d3_positions))
    rows = [row for row in rows if row["effect"] != "missing"]
    records = []
    records.extend(build_baseline_records(rows, "G0_unguarded_256"))
    records.extend(build_baseline_records(rows, "G1_uniform_1024"))
    selected_methods = [name for name in args.methods.split(",") if name.strip()]
    for method in selected_methods:
        if method not in UTILITY_GUARDS:
            raise ValueError(f"unknown utility guard method: {method}")
        records.extend(build_utility_records(rows, method, UTILITY_GUARDS[method]))
    summaries = summarize_records(records)
    effect_counts = dict(Counter(row["effect"] for row in rows))
    verdict = analyze_against_uniform(summaries)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R014B utility-calibrated guard redesign using 4096-visit same-engine relative oracle labels",
        "status": "done" if records else "no_records",
        "position_count": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "effect_counts": effect_counts,
        "utility_methods": {method: UTILITY_GUARDS[method] for method in selected_methods},
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth.",
        "calibration_boundary": "Utility triggers are trained from held-out-group calibration labels derived from v256/v1024/v4096 top-move relations; this is a redesigned guard experiment, not a deployment-free claim.",
        **verdict,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "utility_guard_method_records.jsonl", records)
    write_jsonl(output_root / "utility_guard_summary_records.jsonl", summaries)
    write_json(output_root / "r014b_utility_guard_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    if args.require_beat_uniform and not report.get("beats_uniform_agreement"):
        return 3
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R014B")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--budget-features", default="experiments/results/R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument(
        "--methods",
        default=",".join(UTILITY_GUARDS),
        help="Comma-separated utility guard methods to evaluate.",
    )
    parser.add_argument("--output-root", default="experiments/results/R014B_UTILITY_GUARD_REDESIGN")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--require-beat-uniform", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
