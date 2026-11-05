#!/usr/bin/env python3
"""Post-analyze R020 cross-engine records with TopoRisk trigger scores."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from collections import defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]


def load_script_module(name: str, path: pathlib.Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R016 = load_script_module("r016_budget_generalization_queries", SCRIPT_DIR / "r016_budget_generalization_queries.py")
R018 = load_script_module("r018_guard_aware_hardcoded_attacks", SCRIPT_DIR / "guard_models.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=pathlib.Path, default=pathlib.Path.cwd())
    p.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("experiments/results/R020_CROSS_ENGINE_TOPO_ANALYSIS"))
    p.add_argument(
        "--engine-roots",
        nargs="+",
        default=[
            "experiments/results/R020_LEELA_FULL_CPU16_V512",
            "experiments/results/R020_ELF_FULL_GPU0_R1600_WAIT_R019",
        ],
    )
    p.add_argument("--feature-rows", default="experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    p.add_argument("--r014-budget-features", default="experiments/results/R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl")
    p.add_argument("--d3-positions", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl")
    p.add_argument("--toporisk-fpr-cap", type=float, default=0.05)
    p.add_argument("--utility-penalty", type=float, default=1.5)
    p.add_argument("--random-state", type=int, default=19019)
    p.add_argument("--d4-limit", type=int, default=1000)
    p.add_argument("--d5-limit", type=int, default=1008)
    p.add_argument("--require-complete", action="store_true")
    return p.parse_args()


def fill_defaults(args: argparse.Namespace) -> None:
    project = args.project_root
    args.output_root = args.output_root if args.output_root.is_absolute() else project / args.output_root
    args.engine_roots = [str(pathlib.Path(root) if pathlib.Path(root).is_absolute() else project / root) for root in args.engine_roots]
    for attr in ("feature_rows", "r014_budget_features", "d3_positions"):
        value = pathlib.Path(getattr(args, attr))
        if not value.is_absolute():
            setattr(args, attr, str(project / value))


def read_json(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def normalize_move(move: object) -> str:
    if move is None:
        return ""
    text = str(move).strip().lower()
    return "pass" if text in {"pass", "tt"} else text


def score_positions(args: argparse.Namespace, scorers) -> dict[str, dict]:
    selected, _skipped = R016.load_selected_records(R016.DEFAULT_PATHS, args.d4_limit, args.d5_limit)
    scored: dict[str, dict] = {}
    for record in selected:
        position_id = record["position_id"]
        row = R018.board_feature_row(
            position_id,
            record["board"],
            dataset=record.get("dataset", "unknown"),
            family=record.get("family", "unknown"),
            label=1,
        )
        s = scorers.score_row(row)
        scored[position_id] = {
            "toporisk_b3_fpr05_score": s["toporisk_b3_fpr05_score"],
            "toporisk_b3_fpr05_threshold": s["toporisk_b3_fpr05_threshold"],
            "toporisk_b3_fpr05_triggered": bool(s["toporisk_b3_fpr05_triggered"]),
            "utility_s09_like_score": s["utility_s09_like_score"],
            "utility_s09_like_threshold": s["utility_s09_like_threshold"],
            "utility_s09_like_triggered": bool(s["utility_s09_like_triggered"]),
            "utility_s09_like_p_correction": s["utility_s09_like_p_correction"],
            "utility_s09_like_p_regression": s["utility_s09_like_p_regression"],
            "scored_dataset": record.get("dataset"),
            "scored_family": record.get("family"),
        }
    return scored


def load_engine_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rows = []
    roots = []
    for root_text in args.engine_roots:
        root = pathlib.Path(root_text)
        records_path = root / "cross_engine_records.jsonl"
        report_path = root / "r020_cross_engine_transfer_report.json"
        report = read_json(report_path)
        engine_rows = read_jsonl(records_path)
        roots.append(
            {
                "root": str(root),
                "records_path": str(records_path),
                "report_path": str(report_path),
                "exists": records_path.exists(),
                "report_status": report.get("status"),
                "record_count": len(engine_rows),
                "num_selected": report.get("num_selected"),
            }
        )
        for row in engine_rows:
            row = dict(row)
            row["engine_root"] = str(root)
            rows.append(row)
    return rows, roots


def summarize_guard(rows: list[dict], *, group_name: str, group_value: str, guard: str) -> dict:
    valid = [row for row in rows if row.get("schema_status") == "valid"]
    trig_key = f"{guard}_triggered"
    score_key = f"{guard}_score"
    triggered = [row for row in valid if row.get(trig_key) is True]
    nontriggered = [row for row in valid if row.get(trig_key) is False]

    def disagree(records: list[dict], key: str) -> float | None:
        return mean([float(row[key]) for row in records if row.get(key) is not None])

    return {
        "group": group_name,
        "name": group_value,
        "guard": guard,
        "n": len(rows),
        "valid_n": len(valid),
        "failure_count": len(rows) - len(valid),
        "trigger_rate": mean([float(row[trig_key]) for row in valid if row.get(trig_key) is not None]),
        "mean_score": mean([float(row[score_key]) for row in valid if row.get(score_key) is not None]),
        "disagree_1024_all": disagree(valid, "engine_disagrees_katago_v1024"),
        "disagree_8192_all": disagree(valid, "engine_disagrees_katago_v8192"),
        "disagree_1024_triggered": disagree(triggered, "engine_disagrees_katago_v1024"),
        "disagree_1024_nontriggered": disagree(nontriggered, "engine_disagrees_katago_v1024"),
        "disagree_8192_triggered": disagree(triggered, "engine_disagrees_katago_v8192"),
        "disagree_8192_nontriggered": disagree(nontriggered, "engine_disagrees_katago_v8192"),
        "triggered_n": len(triggered),
        "nontriggered_n": len(nontriggered),
    }


def summarize_all(joined: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in joined:
        engine = str(row.get("engine_name", "unknown"))
        groups[("ALL", "ALL")].append(row)
        groups[("engine", engine)].append(row)
        groups[(f"engine_dataset:{engine}", str(row.get("dataset", "unknown")))].append(row)
        groups[(f"engine_family:{engine}", str(row.get("family", "unknown")))].append(row)
    out = []
    for (group, name), rows in groups.items():
        for guard in ("toporisk_b3_fpr05", "utility_s09_like"):
            out.append(summarize_guard(rows, group_name=group, group_value=name, guard=guard))
    return sorted(out, key=lambda row: (str(row["group"]), str(row["name"]), str(row["guard"])))


def pairwise_summary(joined: list[dict]) -> list[dict]:
    by_position: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        if row.get("schema_status") == "valid":
            by_position[str(row.get("position_id"))].append(row)
    pair_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rows in by_position.values():
        by_engine = {str(row.get("engine_name")): row for row in rows}
        for a, b in itertools.combinations(sorted(by_engine), 2):
            ra, rb = by_engine[a], by_engine[b]
            pair_rows[(a, b)].append(
                {
                    "top_moves_differ": int(normalize_move(ra.get("engine_top_move")) != normalize_move(rb.get("engine_top_move"))),
                    "toporisk_b3_fpr05_triggered": bool(ra.get("toporisk_b3_fpr05_triggered")),
                    "utility_s09_like_triggered": bool(ra.get("utility_s09_like_triggered")),
                }
            )
    out = []
    for (a, b), rows in sorted(pair_rows.items()):
        item = {"engine_pair": f"{a}__{b}", "shared_valid_n": len(rows), "top_move_disagreement": mean([float(r["top_moves_differ"]) for r in rows])}
        for guard in ("toporisk_b3_fpr05", "utility_s09_like"):
            trig = [r for r in rows if r[f"{guard}_triggered"]]
            non = [r for r in rows if not r[f"{guard}_triggered"]]
            item[f"{guard}_triggered_n"] = len(trig)
            item[f"{guard}_nontriggered_n"] = len(non)
            item[f"{guard}_disagree_triggered"] = mean([float(r["top_moves_differ"]) for r in trig])
            item[f"{guard}_disagree_nontriggered"] = mean([float(r["top_moves_differ"]) for r in non])
        out.append(item)
    return out


def main() -> int:
    args = parse_args()
    fill_defaults(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    scorers, scorer_metadata = R018.train_guard_scorers(args)
    score_map = score_positions(args, scorers)
    engine_rows, root_status = load_engine_rows(args)
    incomplete_roots = [root for root in root_status if root.get("report_status") != "done"]
    if args.require_complete and incomplete_roots:
        write_json(args.output_root / "r020_cross_engine_toporisk_analysis_report.json", {"status": "blocked_incomplete", "engine_roots": root_status})
        raise SystemExit(2)
    joined = []
    missing_scores = 0
    for row in engine_rows:
        score = score_map.get(str(row.get("position_id")))
        if not score:
            missing_scores += 1
            continue
        merged = dict(row)
        merged.update(score)
        joined.append(merged)
    summary = summarize_all(joined)
    pairs = pairwise_summary(joined)
    write_jsonl(args.output_root / "r020_toporisk_joined_records.jsonl", joined)
    write_jsonl(args.output_root / "r020_toporisk_transfer_summary.jsonl", summary)
    write_jsonl(args.output_root / "r020_toporisk_engine_pair_summary.jsonl", pairs)
    status = "done" if not incomplete_roots else "partial"
    report = {
        "status": status,
        "generated_at_unix": int(time.time()),
        "engine_roots": root_status,
        "joined_count": len(joined),
        "missing_score_count": missing_scores,
        "summary_rows": len(summary),
        "pair_rows": len(pairs),
        "scorer_metadata": scorer_metadata,
        "boundary": "R020 post-analysis: trigger/disagreement association, not engine correctness or Go ground truth.",
    }
    write_json(args.output_root / "r020_cross_engine_toporisk_analysis_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
