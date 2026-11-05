#!/usr/bin/env python
"""R014C model/feature sweep for utility-calibrated guards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")
R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R014B = load_script_module("r014b_utility_guard_redesign", SCRIPT_DIR / "r014b_utility_guard_redesign.py")


FEATURE_PROFILES = {
    "B3": "B3_topology_gbdt",
    "B4": "B4_topology_plus_search_gbdt",
    "B5": "B5_topology_plus_search_plus_metamorphic_gbdt",
}


def penalty_token(value: float) -> str:
    return str(value).replace(".", "p").rstrip("0").rstrip("p")


def config_name(prefix: str, config: dict) -> str:
    name = f"{prefix}_{config['feature_profile'].lower()}_{config['model_kind']}_p{penalty_token(config['penalty'])}"
    cap = config.get("max_trigger_rate")
    if cap is not None:
        name += f"_cap{int(round(float(cap) * 100))}"
    return name


def build_sweep_configs() -> list[dict]:
    raw = [
        {"feature_profile": "B3", "model_kind": "gbdt", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "gbdt", "penalty": 0.5, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "gbdt", "penalty": 1.5, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "gbdt", "penalty": 2.0, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "logreg", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "rf", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "rf", "penalty": 1.5, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "extra", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B3", "model_kind": "extra", "penalty": 1.5, "max_trigger_rate": None},
        {"feature_profile": "B4", "model_kind": "gbdt", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B4", "model_kind": "gbdt", "penalty": 1.5, "max_trigger_rate": None},
        {"feature_profile": "B5", "model_kind": "gbdt", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B5", "model_kind": "gbdt", "penalty": 2.0, "max_trigger_rate": None},
        {"feature_profile": "B5", "model_kind": "extra", "penalty": 1.0, "max_trigger_rate": None},
        {"feature_profile": "B5", "model_kind": "gbdt", "penalty": 1.0, "max_trigger_rate": 0.75},
        {"feature_profile": "B5", "model_kind": "gbdt", "penalty": 1.0, "max_trigger_rate": 0.50},
    ]
    configs = []
    for idx, config in enumerate(raw, start=1):
        out = dict(config)
        out["name"] = config_name(f"S{idx:02d}", out)
        configs.append(out)
    return configs


def make_model(kind: str, random_state: int):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        )
    if kind == "gbdt":
        return GradientBoostingClassifier(random_state=random_state)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=model_n_jobs(),
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=160,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=model_n_jobs(),
        )
    raise ValueError(f"unsupported model kind: {kind}")


def stable_seed(*parts: object) -> int:
    text = "\t".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x):
        n = len(x)
        p = self.probability
        return np.array([[1.0 - p, p] for _ in range(n)], dtype=float)


def model_n_jobs(default: int = 1) -> int:
    raw = os.environ.get("GOTOPOGUARD_MODEL_N_JOBS")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value != 0 else default


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return R014B.read_jsonl(path)


def write_json(path: pathlib.Path, payload) -> None:
    return R014B.write_json(path, payload)


def write_jsonl(path: pathlib.Path, rows) -> None:
    return R014B.write_jsonl(path, rows)


def feature_names_for_profile(profile: str) -> list[str]:
    return R012.feature_names_for_baseline(FEATURE_PROFILES[profile])


def fit_binary_model(rows: list[dict], feature_names: list[str], positive_label: str, model_kind: str, random_state: int):
    usable = [row for row in rows if R011.features_available(row, feature_names)]
    y = [1 if row["effect"] == positive_label else 0 for row in usable]
    if not usable:
        return ConstantProbabilityModel(0.0), []
    if len(set(y)) < 2:
        return ConstantProbabilityModel(sum(y) / len(y)), usable
    x, _unused = R011.matrix([{**row, "label": label} for row, label in zip(usable, y)], feature_names)
    model = make_model(model_kind, random_state=random_state)
    model.fit(x, np.array(y, dtype=int))
    return model, usable


def score_rows(corr_model, reg_model, rows: list[dict], feature_names: list[str], penalty: float) -> list[dict]:
    usable = [row for row in rows if R011.features_available(row, feature_names)]
    if not usable:
        return []
    x, _unused = R011.matrix([{**row, "label": 0} for row in usable], feature_names)
    corr_p = corr_model.predict_proba(x)[:, 1]
    reg_p = reg_model.predict_proba(x)[:, 1]
    scores = corr_p - penalty * reg_p
    return [
        {
            **row,
            "p_correction": float(cp),
            "p_regression": float(rp),
            "score": float(score),
        }
        for row, cp, rp, score in zip(usable, corr_p, reg_p, scores)
    ]


def build_sweep_records(rows: list[dict], config: dict) -> list[dict]:
    feature_names = feature_names_for_profile(config["feature_profile"])
    records = []
    for group_type, group_value in R014B.heldout_groups(rows):
        train, test = R014B.split_for_group(rows, group_type, group_value)
        seed = stable_seed(config["name"], group_type, group_value)
        corr_model, usable_train = fit_binary_model(train, feature_names, "correction", config["model_kind"], seed)
        reg_model, _usable_reg_train = fit_binary_model(train, feature_names, "regression", config["model_kind"], seed + 17)
        scored_train = score_rows(corr_model, reg_model, usable_train, feature_names, config["penalty"])
        selected = R014B.select_threshold_by_training_utility(scored_train, config.get("max_trigger_rate"))
        scored_test = score_rows(corr_model, reg_model, test, feature_names, config["penalty"])
        threshold = selected["threshold"]
        for row in scored_test:
            triggered = float(row["score"]) >= threshold
            records.append(
                {
                    "position_id": row["position_id"],
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "method": config["name"],
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
                    "feature_profile": config["feature_profile"],
                    "model_kind": config["model_kind"],
                    "regression_penalty": config["penalty"],
                    "max_trigger_rate": config.get("max_trigger_rate"),
                }
            )
    return records


def analyze_against_uniform(summaries: list[dict]) -> dict:
    combined = {row["method"]: row for row in summaries if row["dataset"] == "D3+D5"}
    uniform = combined.get("G1_uniform_1024")
    best = None
    for method, row in combined.items():
        if not method.startswith("S"):
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
        return {"uniform_available": bool(uniform), "best_sweep_method": None}
    _, method, row = best
    return {
        "uniform_available": True,
        "uniform_oracle_agreement_rate": uniform["oracle_agreement_rate"],
        "uniform_mean_visits": uniform["mean_visits"],
        "best_sweep_method": method,
        "best_sweep_oracle_agreement_rate": row["oracle_agreement_rate"],
        "best_sweep_mean_visits": row["mean_visits"],
        "best_sweep_trigger_rate": row["trigger_rate"],
        "best_sweep_net_correction_gain": row["net_correction_gain"],
        "beats_uniform_agreement": row["oracle_agreement_rate"] > uniform["oracle_agreement_rate"],
        "agreement_delta_vs_uniform": row["oracle_agreement_rate"] - uniform["oracle_agreement_rate"],
        "mean_visit_delta_vs_uniform": row["mean_visits"] - uniform["mean_visits"],
    }


def run(args: argparse.Namespace) -> int:
    feature_rows = read_jsonl(pathlib.Path(args.feature_rows))
    budget_rows = read_jsonl(pathlib.Path(args.budget_features))
    rows = R014B.attach_budget_features(feature_rows, budget_rows, pathlib.Path(args.d3_positions))
    rows = [row for row in rows if row["effect"] != "missing"]
    configs = build_sweep_configs()
    if args.configs != "all":
        requested = {name.strip() for name in args.configs.split(",") if name.strip()}
        configs = [config for config in configs if config["name"] in requested]
    records = []
    records.extend(R014B.build_baseline_records(rows, "G0_unguarded_256"))
    records.extend(R014B.build_baseline_records(rows, "G1_uniform_1024"))
    for config in configs:
        records.extend(build_sweep_records(rows, config))
    summaries = R014B.summarize_records(records)
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R014C utility guard model/feature sweep using 4096-visit same-engine relative oracle labels",
        "status": "done" if records else "no_records",
        "position_count": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "effect_counts": dict(Counter(row["effect"] for row in rows)),
        "config_count": len(configs),
        "configs": configs,
        "oracle_boundary": "4096 visits is a same-engine relative oracle only, not ground truth.",
        "calibration_boundary": "Utility triggers are trained from held-out-group calibration labels derived from v256/v1024/v4096 top-move relations.",
        **analyze_against_uniform(summaries),
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "utility_sweep_method_records.jsonl", records)
    write_jsonl(output_root / "utility_sweep_summary_records.jsonl", summaries)
    write_json(output_root / "r014c_utility_guard_model_sweep_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    if args.require_beat_uniform and not report.get("beats_uniform_agreement"):
        return 3
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R014C")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--budget-features", default="experiments/results/R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument("--configs", default="all")
    parser.add_argument("--output-root", default="experiments/results/R014C_UTILITY_GUARD_MODEL_SWEEP")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--require-beat-uniform", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
