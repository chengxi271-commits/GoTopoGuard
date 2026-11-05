#!/usr/bin/env python
"""R015B normal-aware utility guard over D3/D5 and D1/D2 budget labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import time
from collections import Counter

from joblib import Parallel, delayed


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")
R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R014B = load_script_module("r014b_utility_guard_redesign", SCRIPT_DIR / "r014b_utility_guard_redesign.py")
R014C = load_script_module("r014c_utility_guard_model_sweep", SCRIPT_DIR / "r014c_utility_guard_model_sweep.py")
R015 = load_script_module("r015_normal_guard_regression_eval", SCRIPT_DIR / "r015_normal_guard_regression_eval.py")


CONFIGS = [
    {"name": "NA01_b3_extra_p1p5_cost0", "feature_profile": "B3", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.0, "max_trigger_rate": None},
    {"name": "NA02_b3_extra_p1p5_cost01", "feature_profile": "B3", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.01, "max_trigger_rate": None},
    {"name": "NA03_b3_extra_p1p5_cost02", "feature_profile": "B3", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.02, "max_trigger_rate": None},
    {"name": "NA04_b3_extra_p1p5_cost05", "feature_profile": "B3", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.05, "max_trigger_rate": None},
    {"name": "NA05_b3_extra_p1p5_cost10", "feature_profile": "B3", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.10, "max_trigger_rate": None},
    {"name": "NA06_b3_rf_p1p5_cost02", "feature_profile": "B3", "model_kind": "rf", "penalty": 1.5, "visit_penalty": 0.02, "max_trigger_rate": None},
    {"name": "NA07_b3_gbdt_p1p5_cost02", "feature_profile": "B3", "model_kind": "gbdt", "penalty": 1.5, "visit_penalty": 0.02, "max_trigger_rate": None},
    {"name": "NA08_b4_extra_p1p5_cost02", "feature_profile": "B4", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.02, "max_trigger_rate": None},
    {"name": "NA09_b5_extra_p1p5_cost02", "feature_profile": "B5", "model_kind": "extra", "penalty": 1.5, "visit_penalty": 0.02, "max_trigger_rate": None},
]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return R014B.read_jsonl(path)


def write_json(path: pathlib.Path, payload) -> None:
    return R014B.write_json(path, payload)


def write_jsonl(path: pathlib.Path, rows) -> None:
    return R014B.write_jsonl(path, rows)


def agreement_if_triggered(row: dict, triggered: bool) -> int:
    return R014B.agreement_if_triggered(row, triggered)


def select_threshold_by_objective(scored_rows: list[dict], visit_penalty: float, max_trigger_rate: float | None) -> dict:
    if not scored_rows:
        return {"threshold": float("inf"), "objective": None, "agreement_rate": None, "trigger_rate": 0.0, "correction_count": 0, "regression_count": 0}
    thresholds = sorted({float(row["score"]) for row in scored_rows}, reverse=True)
    thresholds.append(float("inf"))
    best = None
    for threshold in thresholds:
        triggered = [row for row in scored_rows if row["score"] >= threshold]
        trigger_rate = len(triggered) / len(scored_rows)
        if max_trigger_rate is not None and trigger_rate > max_trigger_rate:
            continue
        agreements = sum(agreement_if_triggered(row, row["score"] >= threshold) for row in scored_rows)
        corrections = sum(1 for row in triggered if row["effect"] == "correction")
        regressions = sum(1 for row in triggered if row["effect"] == "regression")
        agreement_rate = agreements / len(scored_rows)
        objective = agreement_rate - visit_penalty * trigger_rate
        candidate = {
            "threshold": threshold,
            "objective": objective,
            "agreement_rate": agreement_rate,
            "trigger_rate": trigger_rate,
            "correction_count": corrections,
            "regression_count": regressions,
            "net_gain": corrections - regressions,
        }
        rank = (objective, agreement_rate, corrections - regressions, -trigger_rate, -regressions)
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return best[1] if best else select_threshold_by_objective(scored_rows, visit_penalty, None)


def group_parallel_jobs(default: int = 1) -> int:
    raw = os.environ.get("GOTOPOGUARD_GROUP_N_JOBS")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value != 0 else default


def group_key(row: dict, normal_fold_count: int = 5) -> tuple[str, int | str]:
    if row["dataset"] == "D3":
        return ("D3", int(row["game_index"]))
    if row["dataset"] == "D5":
        return ("D5", row["family"])
    return ("N", R015.stable_natural_fold(row, normal_fold_count))


def summarize_method(rows: list[dict]) -> dict:
    base = R014B.summarize_method(rows)
    return {
        "position_count": base["position_count"],
        "oracle_agreement_count": base["oracle_agreement_count"],
        "oracle_agreement_rate": base["oracle_agreement_rate"],
        "g0_error_count": base["g0_error_count"],
        "correction_count": base["correction_count"],
        "correction_rate_on_g0_errors": base["correction_rate_on_g0_errors"],
        "regression_count": base["regression_count"],
        "regression_rate_on_g0_correct": base["regression_rate_on_g0_correct"],
        "net_correction_gain": base["net_correction_gain"],
        "trigger_count": base["trigger_count"],
        "trigger_rate": base["trigger_rate"],
        "mean_visits": base["mean_visits"],
        "total_visits": base["total_visits"],
    }


def attach_all_budget_features(feature_rows: list[dict], r014_budget_rows: list[dict], r015_budget_rows: list[dict], d3_positions: pathlib.Path) -> list[dict]:
    d3d5_rows = R014B.attach_budget_features(feature_rows, r014_budget_rows, d3_positions)
    d3d5_rows = [row for row in d3d5_rows if row["effect"] != "missing"]
    normal_rows = R015.attach_normal_budget_features(feature_rows, r015_budget_rows)
    for row in normal_rows:
        row["effect"] = R014B.extra_search_effect(row)
    normal_rows = [row for row in normal_rows if row["effect"] != "missing"]
    return d3d5_rows + normal_rows


def method_top_move(row: dict, triggered: bool) -> str | None:
    return row.get("v1024_top_move") if triggered else row.get("v256_top_move")


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
                "oracle_top_move": row["v4096_top_move"],
                "g0_top_move": row["v256_top_move"],
                "method_top_move": method_top_move(row, triggered),
                "method_visits": visits,
                "triggered": triggered,
                "effect": row["effect"],
            }
        )
    return records


def build_config_group_records(rows: list[dict], config: dict, feature_names: list[str], holdout_key: tuple[str, int | str]) -> list[dict]:
    records = []
    train = [row for row in rows if group_key(row) != holdout_key]
    test = [row for row in rows if group_key(row) == holdout_key]
    seed = R014C.stable_seed("R015B", config["name"], holdout_key[0], holdout_key[1])
    corr_model, usable_train = R014C.fit_binary_model(train, feature_names, "correction", config["model_kind"], seed)
    reg_model, _usable_reg_train = R014C.fit_binary_model(train, feature_names, "regression", config["model_kind"], seed + 17)
    scored_train = R014C.score_rows(corr_model, reg_model, usable_train, feature_names, config["penalty"])
    selected = select_threshold_by_objective(scored_train, config["visit_penalty"], config.get("max_trigger_rate"))
    scored_test = R014C.score_rows(corr_model, reg_model, test, feature_names, config["penalty"])
    threshold = selected["threshold"]
    for row in scored_test:
        triggered = row["score"] >= threshold
        records.append(
            {
                "position_id": row["position_id"],
                "dataset": row["dataset"],
                "family": row["family"],
                "method": config["name"],
                "oracle_top_move": row["v4096_top_move"],
                "g0_top_move": row["v256_top_move"],
                "method_top_move": method_top_move(row, triggered),
                "method_visits": 1024 if triggered else 256,
                "triggered": triggered,
                "effect": row["effect"],
                "score": row["score"],
                "p_correction": row["p_correction"],
                "p_regression": row["p_regression"],
                "threshold": threshold,
                "holdout_group_type": holdout_key[0],
                "holdout_group_value": holdout_key[1],
                "train_objective": selected["objective"],
                "train_agreement_rate": selected["agreement_rate"],
                "train_trigger_rate": selected["trigger_rate"],
                "feature_profile": config["feature_profile"],
                "model_kind": config["model_kind"],
                "regression_penalty": config["penalty"],
                "visit_penalty": config["visit_penalty"],
                "max_trigger_rate": config.get("max_trigger_rate"),
            }
        )
    return records


def build_config_records(rows: list[dict], config: dict) -> list[dict]:
    feature_names = R014C.feature_names_for_profile(config["feature_profile"])
    groups = sorted({group_key(row) for row in rows}, key=lambda item: (item[0], str(item[1])))
    jobs = group_parallel_jobs()
    if jobs == 1:
        chunks = [build_config_group_records(rows, config, feature_names, holdout_key) for holdout_key in groups]
    else:
        chunks = Parallel(n_jobs=jobs)(
            delayed(build_config_group_records)(rows, config, feature_names, holdout_key) for holdout_key in groups
        )
    records = []
    for chunk in chunks:
        records.extend(chunk)
    return records


def summarize_records(records: list[dict]) -> list[dict]:
    summaries = []
    for method in sorted({row["method"] for row in records}):
        for dataset in sorted({row["dataset"] for row in records if row["method"] == method}):
            subset = [row for row in records if row["method"] == method and row["dataset"] == dataset]
            summaries.append({"method": method, "dataset": dataset, **summarize_method(subset)})
        for group_name, dataset_set in [
            ("D1+D2", {"D1", "D2"}),
            ("D3+D5", {"D3", "D5"}),
            ("ALL", {"D1", "D2", "D3", "D5"}),
        ]:
            subset = [row for row in records if row["method"] == method and row["dataset"] in dataset_set]
            summaries.append({"method": method, "dataset": group_name, **summarize_method(subset)})
    return summaries


def best_config_summary(summaries: list[dict], dataset: str) -> dict:
    uniform = next(row for row in summaries if row["method"] == "G1_uniform_1024" and row["dataset"] == dataset)
    candidates = [row for row in summaries if row["dataset"] == dataset and row["method"].startswith("NA")]
    best = sorted(
        candidates,
        key=lambda row: (
            row["oracle_agreement_rate"],
            row["net_correction_gain"],
            -row["mean_visits"],
            -row["trigger_rate"],
        ),
        reverse=True,
    )[0]
    return {
        f"{dataset}_uniform_agreement": uniform["oracle_agreement_rate"],
        f"{dataset}_uniform_mean_visits": uniform["mean_visits"],
        f"{dataset}_best_method": best["method"],
        f"{dataset}_best_agreement": best["oracle_agreement_rate"],
        f"{dataset}_best_mean_visits": best["mean_visits"],
        f"{dataset}_best_trigger_rate": best["trigger_rate"],
        f"{dataset}_agreement_delta_vs_uniform": best["oracle_agreement_rate"] - uniform["oracle_agreement_rate"],
        f"{dataset}_mean_visit_delta_vs_uniform": best["mean_visits"] - uniform["mean_visits"],
    }


def run(args: argparse.Namespace) -> int:
    feature_rows = read_jsonl(pathlib.Path(args.feature_rows))
    r014_budget_rows = read_jsonl(pathlib.Path(args.r014_budget_features))
    r015_budget_rows = read_jsonl(pathlib.Path(args.r015_budget_features))
    rows = attach_all_budget_features(feature_rows, r014_budget_rows, r015_budget_rows, pathlib.Path(args.d3_positions))
    configs = CONFIGS
    if args.configs != "all":
        requested = {name.strip() for name in args.configs.split(",") if name.strip()}
        configs = [config for config in CONFIGS if config["name"] in requested]
    records = []
    records.extend(build_baseline_records(rows, "G0_unguarded_256"))
    records.extend(build_baseline_records(rows, "G1_uniform_1024"))
    for config in configs:
        print(json.dumps({"event": "config_start", "method": config["name"], "generated_at_unix": int(time.time())}), flush=True)
        records.extend(build_config_records(rows, config))
        print(json.dumps({"event": "config_done", "method": config["name"], "generated_at_unix": int(time.time())}), flush=True)
    summaries = summarize_records(records)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R015B normal-aware utility guard using out-of-group D1/D2/D3/D5 budget labels",
        "status": "done" if records else "no_records",
        "position_count": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "effect_counts": dict(Counter(row["effect"] for row in rows)),
        "config_count": len(configs),
        "configs": configs,
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth.",
        **best_config_summary(summaries, "D1+D2"),
        **best_config_summary(summaries, "D3+D5"),
        **best_config_summary(summaries, "ALL"),
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "normal_aware_method_records.jsonl", records)
    write_jsonl(output_root / "normal_aware_summary_records.jsonl", summaries)
    write_json(output_root / "r015b_normal_aware_utility_guard_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R015B")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--r014-budget-features", default="experiments/results/R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--r015-budget-features", default="experiments/results/R015_NORMAL_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument("--configs", default="all")
    parser.add_argument("--output-root", default="experiments/results/R015B_NORMAL_AWARE_UTILITY_GUARD")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
