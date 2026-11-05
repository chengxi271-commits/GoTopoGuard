#!/usr/bin/env python
"""Run R013 held-out natural false-positive audit over D1/D2."""

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


R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")
R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")


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


def natural_fold(position_id: str, *, fold_count: int) -> int:
    return R011.stable_bucket(position_id, fold_count)


def summarize_false_positive_group(rows: list[dict], threshold: float) -> dict:
    triggers = [row for row in rows if row["score"] >= threshold]
    return {
        "position_count": len(rows),
        "trigger_count": len(triggers),
        "trigger_rate": len(triggers) / len(rows) if rows else None,
        "mean_score": sum(row["score"] for row in rows) / len(rows) if rows else None,
        "max_score": max((row["score"] for row in rows), default=None),
    }


def load_natural_metadata(paths: list[pathlib.Path]) -> dict[str, dict]:
    metadata = {}
    for path in paths:
        for row in read_jsonl(path):
            metadata[row["position_id"]] = {
                "window": row.get("window"),
                "source_id": row.get("source_id"),
                "source_name": row.get("source_name"),
                "move_prefix_len": row.get("move_prefix_len"),
                "komi": row.get("komi"),
            }
    return metadata


def add_natural_metadata(rows: list[dict], metadata: dict[str, dict], *, fold_count: int) -> list[dict]:
    enriched = []
    for row in rows:
        out = dict(row)
        if row["position_id"] in metadata:
            out.update(metadata[row["position_id"]])
            out["natural_fold"] = natural_fold(row["position_id"], fold_count=fold_count)
        enriched.append(out)
    return enriched


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def run(args: argparse.Namespace) -> int:
    metrics = read_jsonl(pathlib.Path(args.metrics))
    detector_summaries = {}
    if args.detector_baseline == "auto":
        detector, detector_summaries = R012.select_best_detector(metrics)
    else:
        detector = args.detector_baseline
    feature_names = R012.feature_names_for_baseline(detector)
    rows = read_jsonl(pathlib.Path(args.feature_rows))
    natural_metadata = load_natural_metadata([pathlib.Path(args.d1_positions), pathlib.Path(args.d2_positions)])
    rows = add_natural_metadata(rows, natural_metadata, fold_count=args.fold_count)
    target_recalls = [float(value) for value in args.target_recalls.split(",") if value.strip()]
    audit_records = []
    top_false_positive_candidates = []
    for fold in range(args.fold_count):
        train = []
        test = []
        for row in rows:
            if row["label"] == 0 and row["dataset"] in {"D1", "D2"}:
                if int(row["natural_fold"]) == fold:
                    test.append(row)
                else:
                    train.append(row)
            else:
                train.append(row)
        model, filtered_train, y_train, train_scores = R012.fit_model(train, detector, feature_names)
        scored_test = R012.score_rows(model, test, feature_names)
        for target_recall in target_recalls:
            threshold = R012.threshold_at_recall(y_train, train_scores, target_recall)
            for group_key, group_rows in group_natural_rows(scored_test).items():
                dataset, window = group_key
                summary = summarize_false_positive_group(group_rows, threshold)
                audit_records.append(
                    {
                        "run_id": args.run_id,
                        "detector_baseline": detector,
                        "target_recall": target_recall,
                        "threshold_basis": "training_positive_recall",
                        "threshold": threshold,
                        "fold": fold,
                        "dataset": dataset,
                        "window": window,
                        "train_count": len(filtered_train),
                        **summary,
                    }
                )
            triggered = [row for row in scored_test if row["score"] >= threshold]
            for row in sorted(triggered, key=lambda item: item["score"], reverse=True)[: args.top_k_per_fold]:
                top_false_positive_candidates.append(
                    {
                        "run_id": args.run_id,
                        "detector_baseline": detector,
                        "target_recall": target_recall,
                        "fold": fold,
                        "position_id": row["position_id"],
                        "dataset": row["dataset"],
                        "window": row.get("window"),
                        "score": row["score"],
                        "threshold": threshold,
                        "source_name": row.get("source_name"),
                        "move_prefix_len": row.get("move_prefix_len"),
                    }
                )
    summary_by_recall = {}
    for target_recall in target_recalls:
        subset = [row for row in audit_records if row["target_recall"] == target_recall]
        total_positions = sum(row["position_count"] for row in subset)
        total_triggers = sum(row["trigger_count"] for row in subset)
        rates = [row["trigger_rate"] for row in subset if row["trigger_rate"] is not None]
        summary_by_recall[str(target_recall)] = {
            "group_count": len(subset),
            "position_count": total_positions,
            "trigger_count": total_triggers,
            "micro_trigger_rate": total_triggers / total_positions if total_positions else None,
            "mean_group_trigger_rate": mean(rates),
            "median_group_trigger_rate": float(statistics.median(rates)) if rates else None,
            "max_group_trigger_rate": max(rates) if rates else None,
        }
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "R013 held-out natural false-positive audit",
        "status": "done" if audit_records else "no_records",
        "detector_selection": "auto_mean_auroc_then_auprc_then_fpr90" if args.detector_baseline == "auto" else "manual",
        "detector_baseline": detector,
        "detector_summaries": detector_summaries,
        "feature_count": len(feature_names),
        "fold_count": args.fold_count,
        "target_recalls": target_recalls,
        "summary_by_target_recall": summary_by_recall,
        "input_feature_rows": args.feature_rows,
        "input_metrics": args.metrics,
    }
    output_root = pathlib.Path(args.output_root)
    write_jsonl(output_root / "natural_false_positive_records.jsonl", audit_records)
    write_jsonl(output_root / "top_false_positive_candidates.jsonl", top_false_positive_candidates)
    write_json(output_root / "r013_natural_false_positive_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def group_natural_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row.get("window") or "unknown")].append(row)
    return dict(grouped)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R013")
    parser.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    parser.add_argument("--metrics", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/family_heldout_metrics.jsonl")
    parser.add_argument("--d1-positions", default="experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl")
    parser.add_argument("--d2-positions", default="experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl")
    parser.add_argument("--detector-baseline", default="auto")
    parser.add_argument("--target-recalls", default="0.8,0.9")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--top-k-per-fold", type=int, default=25)
    parser.add_argument("--output-root", default="experiments/results/R013_NATURAL_FALSE_POSITIVE_AUDIT")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
