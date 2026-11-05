#!/usr/bin/env python
"""Run R012 leave-one-adversarial-game-out lead-time evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def select_best_detector(metrics: list[dict]) -> tuple[str, dict[str, dict]]:
    by_baseline = defaultdict(list)
    for row in metrics:
        if row.get("status") != "done":
            continue
        if row.get("auroc") is None or row.get("auprc") is None:
            continue
        by_baseline[row["baseline"]].append(row)
    summaries = {}
    for baseline, rows in by_baseline.items():
        summaries[baseline] = {
            "metric_count": len(rows),
            "mean_auroc": mean([float(row["auroc"]) for row in rows]),
            "mean_auprc": mean([float(row["auprc"]) for row in rows]),
            "mean_fpr_at_recall_90": mean([float(row["fpr_at_recall_90"]) for row in rows if row.get("fpr_at_recall_90") is not None]),
        }
    if not summaries:
        raise ValueError("no completed detector metrics available for auto selection")
    selected = sorted(
        summaries,
        key=lambda name: (
            summaries[name]["mean_auroc"] or -1.0,
            summaries[name]["mean_auprc"] or -1.0,
            -(summaries[name]["mean_fpr_at_recall_90"] if summaries[name]["mean_fpr_at_recall_90"] is not None else 1.0),
        ),
        reverse=True,
    )[0]
    return selected, summaries


def feature_names_for_baseline(baseline: str) -> list[str]:
    if baseline == "B1_board_proxy_logreg":
        return R011.BOARD_FEATURES
    if baseline == "B1_engine_confidence_logreg":
        return R011.ENGINE_CONFIDENCE_FEATURES
    if baseline == "B2_search_brittleness_logreg":
        return R011.SEARCH_BRITTLENESS_FEATURES
    if baseline in {"B3_topology_logreg", "B3_topology_gbdt"}:
        return R011.BOARD_FEATURES + R011.TOPOLOGY_FEATURES
    if baseline in {"B4_topology_plus_search_logreg", "B4_topology_plus_search_gbdt"}:
        return R011.BOARD_FEATURES + R011.TOPOLOGY_FEATURES + R011.ENGINE_CONFIDENCE_FEATURES + R011.SEARCH_BRITTLENESS_FEATURES
    if baseline in {"B5_topology_plus_search_plus_metamorphic_logreg", "B5_topology_plus_search_plus_metamorphic_gbdt"}:
        return (
            R011.BOARD_FEATURES
            + R011.TOPOLOGY_FEATURES
            + R011.ENGINE_CONFIDENCE_FEATURES
            + R011.SEARCH_BRITTLENESS_FEATURES
            + R011.METAMORPHIC_FEATURES
        )
    raise ValueError(f"unsupported detector baseline for R012: {baseline}")


def fit_model(train_rows: list[dict], baseline: str, feature_names: list[str]):
    train_rows = [row for row in train_rows if R011.features_available(row, feature_names)]
    if len({int(row["label"]) for row in train_rows}) < 2:
        raise ValueError(f"training rows for {baseline} do not contain both classes")
    x_train, y_train = R011.matrix(train_rows, feature_names)
    if baseline.endswith("logreg"):
        model = R011.make_pipeline(
            R011.StandardScaler(),
            R011.LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
        )
    elif baseline.endswith("gbdt"):
        model = R011.GradientBoostingClassifier(random_state=0)
    else:
        raise ValueError(f"R012 needs a trained detector, got {baseline}")
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_train)[:, 1]
    return model, train_rows, y_train.tolist(), scores.tolist()


def score_rows(model, rows: list[dict], feature_names: list[str]) -> list[dict]:
    usable = [row for row in rows if R011.features_available(row, feature_names)]
    if not usable:
        return []
    x_test, _y_test = R011.matrix(usable, feature_names)
    scores = model.predict_proba(x_test)[:, 1]
    return [{**row, "score": float(score)} for row, score in zip(usable, scores)]


def threshold_at_recall(y_true: list[int], scores: list[float], target_recall: float) -> float:
    paired = sorted(zip(scores, y_true), reverse=True)
    positives = sum(y_true)
    if positives == 0:
        raise ValueError("cannot select recall threshold without positive training examples")
    tp = 0
    index = 0
    while index < len(paired):
        threshold = paired[index][0]
        while index < len(paired) and paired[index][0] == threshold:
            if paired[index][1]:
                tp += 1
            index += 1
        if tp / positives >= target_recall:
            return float(threshold)
    return float(paired[-1][0])


def summarize_game_warning(scored_positions: list[dict], threshold: float) -> dict:
    triggered = [row for row in scored_positions if row["score"] >= threshold]
    leads = [int(row["lead_moves_to_terminal"]) for row in triggered if row.get("lead_moves_to_terminal") is not None]
    earliest = max(leads) if leads else None
    terminal_triggered = any(int(row.get("lead_moves_to_terminal", -1)) == 0 for row in triggered)
    return {
        "warning_detected": earliest is not None,
        "earliest_warning_lead_moves": earliest,
        "terminal_triggered": terminal_triggered,
        "triggered_position_count": len(triggered),
        "evaluated_position_count": len(scored_positions),
        "max_score": max((row["score"] for row in scored_positions), default=None),
        "terminal_score": next((row["score"] for row in scored_positions if int(row.get("lead_moves_to_terminal", -1)) == 0), None),
    }


def load_d3_metadata(path: pathlib.Path) -> dict[str, dict]:
    return {row["position_id"]: row for row in read_jsonl(path)}


def add_d3_metadata(feature_rows: list[dict], d3_metadata: dict[str, dict]) -> list[dict]:
    enriched = []
    for row in feature_rows:
        out = dict(row)
        if row["position_id"] in d3_metadata:
            meta = d3_metadata[row["position_id"]]
            out["game_index"] = meta["game_index"]
            out["lead_moves_to_terminal"] = meta["lead_moves_to_terminal"]
            out["failure_window_bucket"] = meta["failure_window_bucket"]
            out["source_name"] = meta["source_name"]
        enriched.append(out)
    return enriched


def run(args: argparse.Namespace) -> int:
    metrics = read_jsonl(pathlib.Path(args.metrics))
    detector_summaries = {}
    if args.detector_baseline == "auto":
        detector, detector_summaries = select_best_detector(metrics)
    else:
        detector = args.detector_baseline
    feature_names = feature_names_for_baseline(detector)
    rows = read_jsonl(pathlib.Path(args.feature_rows))
    d3_metadata = load_d3_metadata(pathlib.Path(args.d3_positions))
    rows = add_d3_metadata(rows, d3_metadata)
    d3_rows = [row for row in rows if row["dataset"] == "D3" and row["position_id"] in d3_metadata]
    game_indices = sorted({int(row["game_index"]) for row in d3_rows})
    target_recalls = [float(value) for value in args.target_recalls.split(",") if value.strip()]
    lead_records = []
    for game_index in game_indices:
        train = []
        test = []
        for row in rows:
            if row["dataset"] == "D3":
                if int(row.get("game_index", -1)) == game_index:
                    test.append(row)
                else:
                    train.append(row)
            else:
                train.append(row)
        model, filtered_train, y_train, train_scores = fit_model(train, detector, feature_names)
        scored_test = score_rows(model, test, feature_names)
        for target_recall in target_recalls:
            threshold = threshold_at_recall(y_train, train_scores, target_recall)
            warning = summarize_game_warning(scored_test, threshold)
            lead_records.append(
                {
                    "run_id": args.run_id,
                    "detector_baseline": detector,
                    "target_recall": target_recall,
                    "threshold_basis": "training_positive_recall",
                    "threshold": threshold,
                    "heldout_game_index": game_index,
                    "train_count": len(filtered_train),
                    "test_count": len(scored_test),
                    **warning,
                }
            )
    by_recall = {}
    for target_recall in target_recalls:
        subset = [row for row in lead_records if row["target_recall"] == target_recall]
        detected_leads = [int(row["earliest_warning_lead_moves"]) for row in subset if row["earliest_warning_lead_moves"] is not None]
        by_recall[str(target_recall)] = {
            "heldout_game_count": len(subset),
            "warning_detected_count": sum(1 for row in subset if row["warning_detected"]),
            "no_warning_count": sum(1 for row in subset if not row["warning_detected"]),
            "mean_earliest_warning_lead_moves": mean(detected_leads),
            "median_earliest_warning_lead_moves": float(statistics.median(detected_leads)) if detected_leads else None,
            "min_earliest_warning_lead_moves": min(detected_leads) if detected_leads else None,
            "max_earliest_warning_lead_moves": max(detected_leads) if detected_leads else None,
        }
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R012 D3 leave-one-adversarial-game-out lead-time evaluation",
        "status": "done" if lead_records else "no_records",
        "detector_selection": "auto_mean_auroc_then_auprc_then_fpr90" if args.detector_baseline == "auto" else "manual",
        "detector_baseline": detector,
        "detector_summaries": detector_summaries,
        "feature_count": len(feature_names),
        "d3_position_count": len(d3_rows),
        "heldout_game_count": len(game_indices),
        "target_recalls": target_recalls,
        "summary_by_target_recall": by_recall,
        "input_feature_rows": args.feature_rows,
        "input_metrics": args.metrics,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "lead_time_records.jsonl", lead_records)
    write_json(output_root / "r012_lead_time_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R012")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--metrics", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/family_heldout_metrics.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument("--detector-baseline", default="auto")
    parser.add_argument("--target-recalls", default="0.8,0.9")
    parser.add_argument("--output-root", default="experiments/results/R012_LEAD_TIME_EVALUATION")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
