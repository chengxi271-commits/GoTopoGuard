"""Generate Strong-Accept revision analyses and paper assets.

This script is intentionally local and deterministic. It does not run new Go
engine experiments. It regenerates reviewer-facing analyses from existing
R011/R014/R016/R017 artifacts plus fetched R019G/R020 JSON/JSONL reports.
"""

from __future__ import annotations

import importlib.util

import json

import math

import pathlib

import random

import statistics

from collections import Counter, defaultdict

from typing import Iterable

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]

PAPER = ROOT / "paper"

TABLES = PAPER / "tables"

FIGURES = PAPER / "figures"

OUT = ROOT / "experiments" / "results" / "R022_STRONG_ACCEPT_AUDIT"

SCRIPT_DIR = ROOT / "experiments" / "scripts"

def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

R011 = load_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")

TOPO = load_module("topology_features", ROOT / "src" / "gotopobench" / "topology_features.py")

DERIVED_CONTROL_FEATURES = [
    "move_prefix_or_stones",
    "board_density",
    "edge_contact_rate",
    "black_liberty_1_groups",
    "black_liberty_2_groups",
    "black_liberty_3_groups",
    "black_liberty_4plus_groups",
    "white_liberty_1_groups",
    "white_liberty_2_groups",
    "white_liberty_3_groups",
    "white_liberty_4plus_groups",
]

STRICT_CONTROL_FEATURES = (
    R011.BOARD_FEATURES
    + DERIVED_CONTROL_FEATURES
    + R011.ENGINE_CONFIDENCE_FEATURES
    + R011.SEARCH_BRITTLENESS_FEATURES
)

RESIDUAL_TOPOLOGY_FEATURES = [
    "black_group_count",
    "white_group_count",
    "black_max_group_size",
    "white_max_group_size",
    "empty_region_count",
    "enclosed_empty_region_count",
    "black_bounded_empty_region_count",
    "white_bounded_empty_region_count",
    "black_articulation_count",
    "white_articulation_count",
]

STRICT_PLUS_TOPOLOGY_FEATURES = STRICT_CONTROL_FEATURES + RESIDUAL_TOPOLOGY_FEATURES

STRICT_PLUS_TOPOLOGY_PLUS_SYMMETRY_FEATURES = STRICT_PLUS_TOPOLOGY_FEATURES + R011.METAMORPHIC_FEATURES

def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return float(sum(data) / len(data)) if data else None

def metric_summary(y_true: list[int], scores: list[float]) -> dict:
    if set(y_true) != {0, 1}:
        return {"auroc": None, "auprc": None, "status": "single_class"}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "status": "done",
    }

def features_available(row: dict, features: list[str]) -> bool:
    for name in features:
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

def matrix(rows: list[dict], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return x, y

def load_board_metadata() -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for path in [
        ROOT / "experiments/results/R006_NATURAL_HUMAN_SPLIT/positions.jsonl",
        ROOT / "experiments/results/R007_NATURAL_ENGINE_SPLIT/positions.jsonl",
        ROOT / "experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT/positions.jsonl",
        ROOT / "experiments/results/R010_TOPOLOGY_STRESS_SPLIT/positions.jsonl",
    ]:
        for row in read_jsonl(path):
            metadata[row["position_id"]] = row
    for row in read_jsonl(ROOT / "experiments/results/R009_PERTURBATION_SPLIT/perturbation_pairs.jsonl"):
        metadata[row["pair_id"]] = {
            **row,
            "position_id": row["pair_id"],
            "board": row["perturbed_board"],
            "move_prefix_len": row.get("source_move_prefix_len"),
        }
    return metadata

def liberty_histogram(board_rows: list[str], color: str) -> dict[str, int]:
    board = TOPO.parse_board(board_rows)
    groups = TOPO.stone_groups(board, color)
    prefix = "black" if color == "B" else "white"
    out = {
        f"{prefix}_liberty_1_groups": 0,
        f"{prefix}_liberty_2_groups": 0,
        f"{prefix}_liberty_3_groups": 0,
        f"{prefix}_liberty_4plus_groups": 0,
    }
    for group in groups:
        count = int(group["liberty_count"])
        if count <= 1:
            out[f"{prefix}_liberty_1_groups"] += 1
        elif count == 2:
            out[f"{prefix}_liberty_2_groups"] += 1
        elif count == 3:
            out[f"{prefix}_liberty_3_groups"] += 1
        else:
            out[f"{prefix}_liberty_4plus_groups"] += 1
    return out

def enrich_feature_rows() -> list[dict]:
    rows = read_jsonl(ROOT / "experiments/results/R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl")
    metadata = load_board_metadata()
    enriched = []
    for row in rows:
        out = dict(row)
        meta = metadata.get(row["position_id"], {})
        board_rows = meta.get("board")
        if board_rows:
            out.update(liberty_histogram(board_rows, "B"))
            out.update(liberty_histogram(board_rows, "W"))
        else:
            for name in DERIVED_CONTROL_FEATURES:
                out.setdefault(name, 0)
        move_prefix = meta.get("move_prefix_len")
        out["move_prefix_or_stones"] = float(move_prefix if move_prefix is not None else row["stone_count"])
        out["board_density"] = float(row["stone_count"]) / 361.0
        out["edge_contact_rate"] = float(row["edge_stone_count"]) / max(1.0, float(row["stone_count"]))
        out["source_window"] = meta.get("window") or meta.get("failure_window_bucket") or meta.get("family")
        enriched.append(out)
    return enriched

def fit_scores(train: list[dict], test: list[dict], features: list[str]) -> tuple[list[dict], list[float]]:
    train = [row for row in train if features_available(row, features)]
    test = [row for row in test if features_available(row, features)]
    if len({int(row["label"]) for row in train}) < 2 or not test:
        return test, []
    x_train, y_train = matrix(train, features)
    x_test, _ = matrix(test, features)
    model = GradientBoostingClassifier(random_state=0)
    model.fit(x_train, y_train)
    return test, model.predict_proba(x_test)[:, 1].tolist()

def family_heldout_metrics(rows: list[dict]) -> tuple[list[dict], dict]:
    feature_sets = {
        "strict_controls": STRICT_CONTROL_FEATURES,
        "strict_controls_plus_topology": STRICT_PLUS_TOPOLOGY_FEATURES,
        "strict_controls_plus_topology_plus_symmetry": STRICT_PLUS_TOPOLOGY_PLUS_SYMMETRY_FEATURES,
    }
    records = []
    positive_families = sorted({row["family"] for row in rows if int(row["label"]) == 1})
    for heldout in positive_families:
        train, test = R011.family_heldout_split(rows, heldout)
        for name, features in feature_sets.items():
            usable_test, scores = fit_scores(train, test, features)
            if not scores:
                rec = {"heldout_family": heldout, "model": name, "status": "insufficient"}
            else:
                y_true = [int(row["label"]) for row in usable_test]
                rec = {
                    "heldout_family": heldout,
                    "model": name,
                    "feature_count": len(features),
                    "train_count": sum(1 for row in train if features_available(row, features)),
                    "test_count": len(usable_test),
                    "test_positive_count": sum(y_true),
                    **metric_summary(y_true, scores),
                }
            records.append(rec)
    summary = {}
    for name in feature_sets:
        done = [r for r in records if r.get("model") == name and r.get("status") == "done"]
        summary[name] = {
            "folds": len(done),
            "mean_auroc": mean(r["auroc"] for r in done),
            "mean_auprc": mean(r["auprc"] for r in done),
            "feature_count": len(feature_sets[name]),
        }
    base = summary["strict_controls"]
    for name, row in summary.items():
        if row["mean_auroc"] is not None and base["mean_auroc"] is not None:
            row["delta_auroc_vs_strict_controls"] = row["mean_auroc"] - base["mean_auroc"]
            row["delta_auprc_vs_strict_controls"] = row["mean_auprc"] - base["mean_auprc"]
    return records, summary

def matched_family_metrics(rows: list[dict]) -> tuple[list[dict], dict]:
    feature_sets = {
        "strict_controls": STRICT_CONTROL_FEATURES,
        "strict_controls_plus_topology": STRICT_PLUS_TOPOLOGY_FEATURES,
    }
    records = []
    positive_families = sorted({row["family"] for row in rows if int(row["label"]) == 1})
    for heldout in positive_families:
        train, test = R011.family_heldout_split(rows, heldout)
        control_features = STRICT_CONTROL_FEATURES
        test = [row for row in test if features_available(row, STRICT_PLUS_TOPOLOGY_FEATURES)]
        train = [row for row in train if features_available(row, STRICT_PLUS_TOPOLOGY_FEATURES)]
        pos = [row for row in test if int(row["label"]) == 1]
        neg = [row for row in test if int(row["label"]) == 0]
        if not pos or not neg or len({int(row["label"]) for row in train}) < 2:
            continue
        scaler = StandardScaler()
        train_control, _ = matrix(train, control_features)
        scaler.fit(train_control)
        neg_x = scaler.transform(matrix(neg, control_features)[0])
        pos_x = scaler.transform(matrix(pos, control_features)[0])
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(neg_x)
        distances, indices = nn.kneighbors(pos_x)
        matched = []
        for p_row, distance, index in zip(pos, distances[:, 0], indices[:, 0]):
            n_row = dict(neg[int(index)])
            n_row["matched_distance"] = float(distance)
            p_out = dict(p_row)
            p_out["matched_distance"] = float(distance)
            matched.extend([p_out, n_row])
        for name, features in feature_sets.items():
            usable_test, scores = fit_scores(train, matched, features)
            y_true = [int(row["label"]) for row in usable_test]
            records.append(
                {
                    "heldout_family": heldout,
                    "model": name,
                    "matched_pair_count": len(pos),
                    "matched_negative_reuse_allowed": True,
                    "mean_nearest_control_distance": mean(float(row["matched_distance"]) for row in usable_test),
                    **metric_summary(y_true, scores),
                }
            )
    summary = {}
    for name in feature_sets:
        done = [r for r in records if r["model"] == name and r["status"] == "done"]
        summary[name] = {
            "folds": len(done),
            "mean_auroc": mean(r["auroc"] for r in done),
            "mean_auprc": mean(r["auprc"] for r in done),
            "mean_matched_pairs": mean(r["matched_pair_count"] for r in done),
        }
    base = summary["strict_controls"]
    for name, row in summary.items():
        if row["mean_auroc"] is not None and base["mean_auroc"] is not None:
            row["delta_auroc_vs_strict_controls"] = row["mean_auroc"] - base["mean_auroc"]
            row["delta_auprc_vs_strict_controls"] = row["mean_auprc"] - base["mean_auprc"]
    return records, summary

def mechanism_holdout(rows: list[dict]) -> tuple[list[dict], dict]:
    feature_sets = {
        "board_proxy": R011.BOARD_FEATURES,
        "b3_board_plus_topology": R011.BOARD_FEATURES + R011.TOPOLOGY_FEATURES,
        "b5_full": R011.BOARD_FEATURES + R011.TOPOLOGY_FEATURES + R011.ENGINE_CONFIDENCE_FEATURES + R011.SEARCH_BRITTLENESS_FEATURES + R011.METAMORPHIC_FEATURES,
        "strict_plus_topology": STRICT_PLUS_TOPOLOGY_FEATURES,
    }
    tasks = {
        "holdout_d5_generator": {
            "train_pos": {"D3", "D4"},
            "test_pos": {"D5"},
            "purpose": "D5 generated topology mechanism held out from positive training.",
        },
        "holdout_d3_public_attack": {
            "train_pos": {"D4", "D5"},
            "test_pos": {"D3"},
            "purpose": "Public adversarial replay split held out from positive training.",
        },
    }
    records = []
    for task, spec in tasks.items():
        for model_name, features in feature_sets.items():
            train = []
            test = []
            for row in rows:
                if int(row["label"]) == 0:
                    if R011.stable_bucket(row["position_id"], 2) == 0:
                        test.append(row)
                    else:
                        train.append(row)
                elif row["dataset"] in spec["train_pos"]:
                    train.append(row)
                elif row["dataset"] in spec["test_pos"]:
                    test.append(row)
            usable_test, scores = fit_scores(train, test, features)
            if scores:
                y_true = [int(row["label"]) for row in usable_test]
                rec = {
                    "task": task,
                    "purpose": spec["purpose"],
                    "model": model_name,
                    "feature_count": len(features),
                    "train_count": sum(1 for row in train if features_available(row, features)),
                    "test_count": len(usable_test),
                    "test_positive_count": sum(y_true),
                    **metric_summary(y_true, scores),
                }
            else:
                rec = {"task": task, "model": model_name, "status": "insufficient"}
            records.append(rec)
    summary = {
        task: {row["model"]: row for row in records if row.get("task") == task and row.get("status") == "done"}
        for task in tasks
    }
    return records, summary

def normalize_move(move: object) -> str:
    text = str(move or "").strip().lower()
    return "pass" if text in {"pass", "tt"} else text

def exact_binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    prob = 0.0
    tail = min(k, n - k)
    for i in range(0, tail + 1):
        prob += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * prob)

def bootstrap_delta(a: list[int], b: list[int], *, rounds: int = 5000) -> dict:
    rng = random.Random(19019)
    n = len(a)
    deltas = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idx) / n)
    deltas.sort()
    return {
        "rounds": rounds,
        "ci95_low": float(deltas[int(0.025 * rounds)]),
        "ci95_high": float(deltas[int(0.975 * rounds)]),
    }

def guard_significance() -> dict:
    report = read_json(ROOT / "experiments/results/R014C_UTILITY_GUARD_MODEL_SWEEP/r014c_utility_guard_model_sweep_report.json")
    best = report["best_sweep_method"]
    records = read_jsonl(ROOT / "experiments/results/R014C_UTILITY_GUARD_MODEL_SWEEP/utility_sweep_method_records.jsonl")
    by_key: dict[tuple[str, str], dict] = {}
    for row in records:
        if row["method"] in {"G1_uniform_1024", best}:
            by_key[(row["position_id"], row["method"])] = row
    paired = []
    for position_id in sorted({key[0] for key in by_key}):
        uniform = by_key.get((position_id, "G1_uniform_1024"))
        method = by_key.get((position_id, best))
        if not uniform or not method:
            continue
        u_correct = int(normalize_move(uniform["method_top_move"]) == normalize_move(uniform["oracle_top_move"]))
        m_correct = int(normalize_move(method["method_top_move"]) == normalize_move(method["oracle_top_move"]))
        paired.append({"position_id": position_id, "uniform_correct": u_correct, "method_correct": m_correct})
    n10 = sum(1 for row in paired if row["method_correct"] == 1 and row["uniform_correct"] == 0)
    n01 = sum(1 for row in paired if row["method_correct"] == 0 and row["uniform_correct"] == 1)
    method = [row["method_correct"] for row in paired]
    uniform = [row["uniform_correct"] for row in paired]
    delta = (sum(method) - sum(uniform)) / len(paired)
    result = {
        "best_method": best,
        "paired_count": len(paired),
        "uniform_correct": sum(uniform),
        "method_correct": sum(method),
        "agreement_delta": float(delta),
        "mcnemar_method_only_correct": n10,
        "mcnemar_uniform_only_correct": n01,
        "mcnemar_exact_p": exact_binomial_two_sided(n10, n10 + n01),
        **bootstrap_delta(method, uniform),
    }
    write_jsonl(OUT / "guard_significance_pairs.jsonl", paired)
    return result

def summarize_score_sensitivity() -> dict:
    budget = read_jsonl(ROOT / "experiments/results/R016_BUDGET_GENERALIZATION_ANALYSIS/budget_stability_records.jsonl")
    network = read_jsonl(ROOT / "experiments/results/R017_NETWORK_GENERALIZATION_ANALYSIS/network_stability_records.jsonl")

    def summary(rows: list[dict], key: str, group_key: str) -> dict:
        out = {}
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups["ALL"].append(float(row[key]))
            groups[str(row.get(group_key, "unknown"))].append(float(row[key]))
        for name, values in groups.items():
            values = sorted(values)
            out[name] = {
                "n": len(values),
                "mean": mean(values),
                "median": float(statistics.median(values)) if values else None,
                "p90": float(values[int(0.9 * (len(values) - 1))]) if values else None,
            }
        return out

    return {
        "budget_score_abs_delta_1024_8192": summary(budget, "score_abs_delta_1024_8192", "dataset"),
        "budget_winrate_abs_delta_1024_8192": summary(budget, "winrate_abs_delta_1024_8192", "dataset"),
        "network_score_abs_delta_vs_b28_1024": summary(network, "score_abs_delta_vs_b28_1024", "network_alias"),
        "network_winrate_abs_delta_vs_b28_1024": summary(network, "winrate_abs_delta_vs_b28_1024", "network_alias"),
    }
