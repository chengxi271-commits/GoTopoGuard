#!/usr/bin/env python
"""Evaluate detector operating points under natural-FPR caps.

This supplements R012/R013 when training-recall thresholds create excessive
natural trigger rates. Thresholds are selected only from training natural
negatives, then evaluated on a held-out natural fold and a held-out D3 game.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import time


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R013 = load_script_module("r013_natural_false_positive_audit", SCRIPT_DIR / "r013_natural_false_positive_audit.py")


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


def threshold_at_fpr_cap(negative_scores: list[float], fpr_cap: float) -> float:
    if not negative_scores:
        raise ValueError("cannot select FPR threshold without negative scores")
    sorted_scores = sorted(negative_scores, reverse=True)
    total = len(sorted_scores)
    triggered = 0
    best_threshold = max(sorted_scores) + 1e-12
    index = 0
    while index < total:
        threshold = sorted_scores[index]
        group_count = 0
        while index < total and sorted_scores[index] == threshold:
            group_count += 1
            index += 1
        candidate_triggered = triggered + group_count
        if candidate_triggered / total > fpr_cap:
            break
        triggered = candidate_triggered
        best_threshold = threshold
    return float(best_threshold)


def trigger_rate(rows: list[dict], threshold: float) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row["score"] >= threshold) / len(rows)


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(pathlib.Path(args.feature_rows))
    d3_metadata = R012.load_d3_metadata(pathlib.Path(args.d3_positions))
    natural_metadata = R013.load_natural_metadata([pathlib.Path(args.d1_positions), pathlib.Path(args.d2_positions)])
    rows = R012.add_d3_metadata(rows, d3_metadata)
    rows = R013.add_natural_metadata(rows, natural_metadata, fold_count=args.fold_count)
    d3_rows = [row for row in rows if row["dataset"] == "D3" and row["position_id"] in d3_metadata]
    game_indices = sorted({int(row["game_index"]) for row in d3_rows})
    detectors = [value for value in args.detectors.split(",") if value.strip()]
    fpr_caps = [float(value) for value in args.fpr_caps.split(",") if value.strip()]
    records = []
    for detector in detectors:
        feature_names = R012.feature_names_for_baseline(detector)
        for fold in range(args.fold_count):
            for game_index in game_indices:
                train = []
                test_natural = []
                test_d3 = []
                for row in rows:
                    if row["label"] == 0 and row["dataset"] in {"D1", "D2"}:
                        if int(row["natural_fold"]) == fold:
                            test_natural.append(row)
                        else:
                            train.append(row)
                    elif row["dataset"] == "D3":
                        if int(row.get("game_index", -1)) == game_index:
                            test_d3.append(row)
                        else:
                            train.append(row)
                    else:
                        train.append(row)
                model, filtered_train, _y_train, _train_scores = R012.fit_model(train, detector, feature_names)
                train_natural = [row for row in filtered_train if row["label"] == 0 and row["dataset"] in {"D1", "D2"}]
                scored_train_natural = R012.score_rows(model, train_natural, feature_names)
                scored_test_natural = R012.score_rows(model, test_natural, feature_names)
                scored_test_d3 = R012.score_rows(model, test_d3, feature_names)
                negative_scores = [row["score"] for row in scored_train_natural]
                for fpr_cap in fpr_caps:
                    threshold = threshold_at_fpr_cap(negative_scores, fpr_cap)
                    warning = R012.summarize_game_warning(scored_test_d3, threshold)
                    records.append(
                        {
                            "run_id": args.run_id,
                            "detector_baseline": detector,
                            "threshold_basis": "training_natural_fpr_cap",
                            "fpr_cap": fpr_cap,
                            "threshold": threshold,
                            "fold": fold,
                            "heldout_game_index": game_index,
                            "train_count": len(filtered_train),
                            "test_natural_count": len(scored_test_natural),
                            "test_natural_trigger_rate": trigger_rate(scored_test_natural, threshold),
                            **warning,
                        }
                    )
    summary = {}
    for detector in detectors:
        summary[detector] = {}
        for fpr_cap in fpr_caps:
            subset = [row for row in records if row["detector_baseline"] == detector and row["fpr_cap"] == fpr_cap]
            leads = [int(row["earliest_warning_lead_moves"]) for row in subset if row["earliest_warning_lead_moves"] is not None]
            natural_rates = [row["test_natural_trigger_rate"] for row in subset if row["test_natural_trigger_rate"] is not None]
            summary[detector][str(fpr_cap)] = {
                "evaluation_count": len(subset),
                "warning_detected_count": sum(1 for row in subset if row["warning_detected"]),
                "warning_detected_rate": sum(1 for row in subset if row["warning_detected"]) / len(subset) if subset else None,
                "mean_earliest_warning_lead_moves": mean(leads),
                "median_earliest_warning_lead_moves": float(statistics.median(leads)) if leads else None,
                "mean_test_natural_trigger_rate": mean(natural_rates),
                "median_test_natural_trigger_rate": float(statistics.median(natural_rates)) if natural_rates else None,
            }
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R013 operating-point tradeoff under natural-FPR-capped thresholds",
        "status": "done" if records else "no_records",
        "detectors": detectors,
        "fpr_caps": fpr_caps,
        "fold_count": args.fold_count,
        "heldout_game_count": len(game_indices),
        "record_count": len(records),
        "summary": summary,
        "input_feature_rows": args.feature_rows,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "operating_point_records.jsonl", records)
    write_json(output_root / "r013_operating_point_tradeoff_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R013_OPERATING_POINT_TRADEOFF")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--d1-positions", default="experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl")
    parser.add_argument("--d2-positions", default="experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl")
    parser.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    parser.add_argument("--detectors", default="B3_topology_gbdt,B5_topology_plus_search_plus_metamorphic_gbdt")
    parser.add_argument("--fpr-caps", default="0.01,0.05,0.1")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--output-root", default="experiments/results/R013_OPERATING_POINT_TRADEOFF")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
