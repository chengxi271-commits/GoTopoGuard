#!/usr/bin/env python
"""Run available R011 detection baselines over D1-D5 feature rows.

This script does not fabricate missing KataGo search features. It runs B0 and
topology/board-proxy baselines from currently available board features, and
records planned engine/search baselines as missing prerequisites.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPO = load_module("topology_features", ROOT_DIR / "src" / "gotopobench" / "topology_features.py")

DEFAULT_PATHS = {
    "D1": "experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl",
    "D2": "experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl",
    "D3": "experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl",
    "D4": "experiments/results/R009_PERTURBATION_SPLIT/perturbation_pairs.jsonl",
    "D5": "experiments/results/R010_TOPOLOGY_STRESS_SPLIT/positions.jsonl",
}

BOARD_FEATURES = [
    "black_count",
    "white_count",
    "empty_count",
    "stone_count",
    "stone_balance",
    "edge_stone_count",
]

TOPOLOGY_FEATURES = [
    "black_group_count",
    "white_group_count",
    "black_max_group_size",
    "white_max_group_size",
    "black_max_liberty_count",
    "white_max_liberty_count",
    "empty_region_count",
    "enclosed_empty_region_count",
    "black_bounded_empty_region_count",
    "white_bounded_empty_region_count",
    "black_articulation_count",
    "white_articulation_count",
]

ENGINE_CONFIDENCE_FEATURES = [
    "v256_winrate",
    "v256_scoreLead",
    "v256_scoreStdev",
    "v256_top_visit_gap",
    "v256_top_prior",
    "v256_move_info_count",
    "v256_visit_entropy",
    "v256_prior_entropy",
    "v256_policy_entropy",
]

SEARCH_BRITTLENESS_FEATURES = [
    "budget_top_move_disagreement_64_256",
    "budget_winrate_abs_delta_64_256",
    "budget_score_abs_delta_64_256",
]

METAMORPHIC_FEATURES = [
    "sym_v64_top_move_disagreement_rate",
    "sym_v64_top_move_unique_count",
    "sym_v64_winrate_abs_delta_mean",
    "sym_v64_winrate_abs_delta_max",
    "sym_v64_score_abs_delta_mean",
    "sym_v64_score_abs_delta_max",
    "sym_v256_top_move_disagreement_rate",
    "sym_v256_top_move_unique_count",
    "sym_v256_winrate_abs_delta_mean",
    "sym_v256_winrate_abs_delta_max",
    "sym_v256_score_abs_delta_mean",
    "sym_v256_score_abs_delta_max",
]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def board_counts(board_rows: list[str]) -> dict:
    black = sum(row.count("B") for row in board_rows)
    white = sum(row.count("W") for row in board_rows)
    empty = sum(row.count(".") for row in board_rows)
    size = len(board_rows)
    edge = 0
    for row_index, row in enumerate(board_rows):
        for col_index, stone in enumerate(row):
            if stone in ("B", "W") and (row_index in (0, size - 1) or col_index in (0, size - 1)):
                edge += 1
    return {
        "black_count": black,
        "white_count": white,
        "empty_count": empty,
        "stone_count": black + white,
        "stone_balance": black - white,
        "edge_stone_count": edge,
    }


def feature_row_from_board(*, position_id: str, board_rows: list[str], label: int, dataset: str, family: str) -> dict:
    board = TOPO.parse_board(board_rows)
    features = TOPO.extract_topology_features(board)
    counts = board_counts(board_rows)
    return {
        "position_id": position_id,
        "label": int(label),
        "dataset": dataset,
        "family": family,
        **counts,
        **features,
    }


def load_feature_rows(paths: dict[str, str]) -> list[dict]:
    rows = []
    for record in read_jsonl(pathlib.Path(paths["D1"])):
        rows.append(feature_row_from_board(position_id=record["position_id"], board_rows=record["board"], label=0, dataset="D1", family="natural_human"))
    for record in read_jsonl(pathlib.Path(paths["D2"])):
        rows.append(feature_row_from_board(position_id=record["position_id"], board_rows=record["board"], label=0, dataset="D2", family="natural_engine"))
    for record in read_jsonl(pathlib.Path(paths["D3"])):
        family = f"go_attack_{record['failure_window_bucket']}"
        rows.append(feature_row_from_board(position_id=record["position_id"], board_rows=record["board"], label=1, dataset="D3", family=family))
    for record in read_jsonl(pathlib.Path(paths["D4"])):
        rows.append(feature_row_from_board(position_id=record["pair_id"], board_rows=record["perturbed_board"], label=1, dataset="D4", family="perturbation_low_adjacency"))
    for record in read_jsonl(pathlib.Path(paths["D5"])):
        rows.append(feature_row_from_board(position_id=record["position_id"], board_rows=record["board"], label=1, dataset="D5", family=record["family"]))
    return rows


def load_search_feature_map(path: pathlib.Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["position_id"]: row for row in read_jsonl(path)}


def load_metamorphic_feature_map(path: pathlib.Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["position_id"]: row for row in read_jsonl(path)}


def attach_search_features(rows: list[dict], search_features: dict[str, dict]) -> tuple[list[dict], int]:
    merged = []
    attached = 0
    for row in rows:
        feature_row = dict(row)
        extras = search_features.get(row["position_id"])
        if extras:
            attached += 1
            for key, value in extras.items():
                if key != "position_id":
                    feature_row[key] = value
        merged.append(feature_row)
    return merged, attached


def attach_metamorphic_features(rows: list[dict], metamorphic_features: dict[str, dict]) -> tuple[list[dict], int]:
    merged = []
    attached = 0
    for row in rows:
        feature_row = dict(row)
        extras = metamorphic_features.get(row["position_id"])
        if extras:
            attached += 1
            for key, value in extras.items():
                if key != "position_id":
                    feature_row[key] = value
        merged.append(feature_row)
    return merged, attached


def family_heldout_split(rows: list[dict], heldout_family: str) -> tuple[list[dict], list[dict]]:
    train = []
    test = []
    for row in rows:
        if row["label"] == 1:
            if row["family"] == heldout_family:
                test.append(row)
            else:
                train.append(row)
            continue
        if stable_bucket(row["position_id"], 5) == 0:
            test.append(row)
        else:
            train.append(row)
    return train, test


def matrix(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return x, y


def features_available(row: dict, feature_names: list[str]) -> bool:
    for name in feature_names:
        value = row.get(name)
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if math.isnan(numeric) or math.isinf(numeric):
            return False
    return True


def fpr_at_recall(y_true, scores, target_recall: float) -> float | None:
    paired = sorted(zip(scores, y_true), reverse=True)
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    tp = 0
    fp = 0
    index = 0
    while index < len(paired):
        score = paired[index][0]
        while index < len(paired) and paired[index][0] == score:
            label = paired[index][1]
            if label:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / positives
        if recall >= target_recall:
            return fp / negatives
    return None


def metric_summary(y_true, scores) -> dict:
    labels = set(int(value) for value in y_true)
    if labels != {0, 1}:
        return {
            "auroc": None,
            "auprc": None,
            "fpr_at_recall_80": None,
            "fpr_at_recall_90": None,
            "status": "single_class_test",
        }
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "fpr_at_recall_80": fpr_at_recall(y_true, scores, 0.8),
        "fpr_at_recall_90": fpr_at_recall(y_true, scores, 0.9),
        "status": "done",
    }


def fit_predict_baseline(train: list[dict], test: list[dict], baseline: str, feature_names: list[str]) -> dict:
    if baseline != "B0_prior":
        train = [row for row in train if features_available(row, feature_names)]
        test = [row for row in test if features_available(row, feature_names)]
    x_train, y_train = matrix(train, feature_names)
    x_test, y_test = matrix(test, feature_names)
    if len(set(y_train.tolist())) < 2 or len(test) == 0:
        return {
            "baseline": baseline,
            "feature_count": len(feature_names),
            "train_count": len(train),
            "test_count": len(test),
            "test_positive_count": int(np.sum(y_test)) if len(test) else 0,
            "test_negative_count": int(len(y_test) - np.sum(y_test)) if len(test) else 0,
            "auroc": None,
            "auprc": None,
            "fpr_at_recall_80": None,
            "fpr_at_recall_90": None,
            "status": "insufficient_filtered_classes",
        }
    if baseline == "B0_prior":
        prior = float(np.mean(y_train)) if len(y_train) else 0.0
        scores = np.full(len(y_test), prior, dtype=float)
    elif baseline.endswith("logreg"):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
        )
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
    elif baseline.endswith("gbdt"):
        model = GradientBoostingClassifier(random_state=0)
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
    else:
        raise ValueError(f"unknown baseline {baseline}")
    metrics = metric_summary(y_test.tolist(), scores.tolist())
    metrics.update(
        {
            "baseline": baseline,
            "feature_count": len(feature_names),
            "train_count": len(train),
            "test_count": len(test),
            "test_positive_count": int(np.sum(y_test)),
            "test_negative_count": int(len(y_test) - np.sum(y_test)),
        }
    )
    return metrics


def run_family_heldout(rows: list[dict], *, include_search_baselines: bool, include_metamorphic_baselines: bool = False) -> list[dict]:
    positive_families = sorted({row["family"] for row in rows if row["label"] == 1})
    baselines = {
        "B0_prior": [],
        "B1_board_proxy_logreg": BOARD_FEATURES,
        "B3_topology_logreg": BOARD_FEATURES + TOPOLOGY_FEATURES,
        "B3_topology_gbdt": BOARD_FEATURES + TOPOLOGY_FEATURES,
    }
    if include_search_baselines:
        baselines.update(
            {
                "B1_engine_confidence_logreg": ENGINE_CONFIDENCE_FEATURES,
                "B2_search_brittleness_logreg": SEARCH_BRITTLENESS_FEATURES,
                "B4_topology_plus_search_logreg": BOARD_FEATURES + TOPOLOGY_FEATURES + ENGINE_CONFIDENCE_FEATURES + SEARCH_BRITTLENESS_FEATURES,
                "B4_topology_plus_search_gbdt": BOARD_FEATURES + TOPOLOGY_FEATURES + ENGINE_CONFIDENCE_FEATURES + SEARCH_BRITTLENESS_FEATURES,
            }
        )
    if include_search_baselines and include_metamorphic_baselines:
        baselines.update(
            {
                "B5_topology_plus_search_plus_metamorphic_logreg": BOARD_FEATURES
                + TOPOLOGY_FEATURES
                + ENGINE_CONFIDENCE_FEATURES
                + SEARCH_BRITTLENESS_FEATURES
                + METAMORPHIC_FEATURES,
                "B5_topology_plus_search_plus_metamorphic_gbdt": BOARD_FEATURES
                + TOPOLOGY_FEATURES
                + ENGINE_CONFIDENCE_FEATURES
                + SEARCH_BRITTLENESS_FEATURES
                + METAMORPHIC_FEATURES,
            }
        )
    metrics = []
    for heldout in positive_families:
        train, test = family_heldout_split(rows, heldout)
        for baseline, features in baselines.items():
            names = BOARD_FEATURES if baseline == "B0_prior" else features
            result = fit_predict_baseline(train, test, baseline, names)
            result["heldout_family"] = heldout
            metrics.append(result)
    return metrics


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    rows = load_feature_rows(DEFAULT_PATHS)
    search_map = load_search_feature_map(pathlib.Path(args.search_features))
    rows, attached_search_count = attach_search_features(rows, search_map)
    metamorphic_map = load_metamorphic_feature_map(pathlib.Path(args.metamorphic_features))
    rows, attached_metamorphic_count = attach_metamorphic_features(rows, metamorphic_map)
    include_search = attached_search_count > 0
    include_metamorphic = attached_metamorphic_count > 0
    metrics = run_family_heldout(rows, include_search_baselines=include_search, include_metamorphic_baselines=include_metamorphic)
    dataset_counts = Counter(row["dataset"] for row in rows)
    family_counts = Counter(row["family"] for row in rows if row["label"] == 1)
    if include_search and include_metamorphic:
        status = "done"
        reason_not_done = None
        planned_not_run = []
    elif include_search:
        status = "partial_missing_metamorphic_features"
        reason_not_done = "B5 planned metamorphic features are not complete."
        planned_not_run = ["B5_topology_plus_search_plus_metamorphic"]
    else:
        status = "partial_missing_engine_features"
        reason_not_done = "B1/B2/B4/B5 planned engine/search/metamorphic baselines require full KataGo search-feature extraction over D1-D5."
        planned_not_run = [
            "B1_engine_confidence",
            "B2_search_brittleness",
            "B4_topology_plus_search",
            "B5_topology_plus_search_plus_metamorphic",
        ]
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "family-held-out R011 detection baseline sweep",
        "status": status,
        "reason_not_done": reason_not_done,
        "dataset_counts": dict(dataset_counts),
        "positive_family_counts": dict(family_counts),
        "row_count": len(rows),
        "metric_row_count": len(metrics),
        "search_feature_rows_attached": attached_search_count,
        "search_feature_path": args.search_features,
        "metamorphic_feature_rows_attached": attached_metamorphic_count,
        "metamorphic_feature_path": args.metamorphic_features,
        "available_baselines_run": sorted({row["baseline"] for row in metrics}),
        "planned_baselines_not_run": planned_not_run,
        "feature_groups": {
            "board_proxy": BOARD_FEATURES,
            "topology": TOPOLOGY_FEATURES,
            "engine_confidence": ENGINE_CONFIDENCE_FEATURES,
            "search_brittleness": SEARCH_BRITTLENESS_FEATURES,
            "metamorphic_symmetry": METAMORPHIC_FEATURES,
        },
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "feature_rows.jsonl", rows)
    write_jsonl(output_root / "family_heldout_metrics.jsonl", metrics)
    write_json(output_root / "r011_detection_baseline_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_full and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R011")
    parser.add_argument("--output-root", default="experiments/results/R011_DETECTION_BASELINE_SWEEP")
    parser.add_argument("--search-features", default="experiments/results/R011_KATAGO_SEARCH_FEATURES/position_search_features.jsonl")
    parser.add_argument("--metamorphic-features", default="experiments/results/R011_METAMORPHIC_SYMMETRY_FEATURES/position_metamorphic_features.jsonl")
    parser.add_argument("--fail-on-full", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
