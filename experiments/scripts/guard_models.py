"""PRICAI scoring helpers extracted unchanged from the R018 utility module.

Used by the R020 post-analysis; no hardcoded-policy or live-game evaluation is
included in this module. See docs/RECOVERY.md for provenance.
"""
from __future__ import annotations
import hashlib
import json
import argparse
import pathlib
import numpy as np
import importlib.util
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
def load_script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
R011 = load_script_module("r011_detection_baseline_sweep", SCRIPT_DIR / "r011_detection_baseline_sweep.py")
R012 = load_script_module("r012_lead_time_evaluation", SCRIPT_DIR / "r012_lead_time_evaluation.py")
R013 = load_script_module("r013_operating_point_tradeoff", SCRIPT_DIR / "r013_operating_point_tradeoff.py")
R013OP = R013
R014B = load_script_module("r014b_utility_guard_redesign", SCRIPT_DIR / "r014b_utility_guard_redesign.py")
R014C = load_script_module("r014c_utility_guard_model_sweep", SCRIPT_DIR / "r014c_utility_guard_model_sweep.py")
TOPO = load_script_module("topology_features", ROOT_DIR / "src/gotopobench/topology_features.py")
TOPO_FEATURES = R011.BOARD_FEATURES + R011.TOPOLOGY_FEATURES

def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("\t".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None

def board_feature_row(position_id: str, board_rows: list[str], *, dataset: str, family: str, label: int) -> dict:
    return R011.feature_row_from_board(
        position_id=position_id,
        board_rows=board_rows,
        label=label,
        dataset=dataset,
        family=family,
    )

class GuardScorers:
    def __init__(
        self,
        *,
        toporisk_model: object,
        toporisk_threshold: float,
        utility_corr_model: object,
        utility_reg_model: object,
        utility_threshold: float,
        utility_penalty: float,
    ):
        self.toporisk_model = toporisk_model
        self.toporisk_threshold = toporisk_threshold
        self.utility_corr_model = utility_corr_model
        self.utility_reg_model = utility_reg_model
        self.utility_threshold = utility_threshold
        self.utility_penalty = utility_penalty

    def score_row(self, row: dict) -> dict:
        x = np.array([[float(row[name]) for name in TOPO_FEATURES]], dtype=float)
        toporisk_score = float(self.toporisk_model.predict_proba(x)[0, 1])
        corr_p = float(self.utility_corr_model.predict_proba(x)[0, 1])
        reg_p = float(self.utility_reg_model.predict_proba(x)[0, 1])
        utility_score = corr_p - self.utility_penalty * reg_p
        return {
            "toporisk_b3_fpr05_score": toporisk_score,
            "toporisk_b3_fpr05_threshold": self.toporisk_threshold,
            "toporisk_b3_fpr05_triggered": toporisk_score >= self.toporisk_threshold,
            "utility_s09_like_score": utility_score,
            "utility_s09_like_threshold": self.utility_threshold,
            "utility_s09_like_triggered": utility_score >= self.utility_threshold,
            "utility_s09_like_p_correction": corr_p,
            "utility_s09_like_p_regression": reg_p,
        }

def train_guard_scorers(args: argparse.Namespace) -> tuple[GuardScorers, dict]:
    feature_rows = read_jsonl(pathlib.Path(args.feature_rows))
    feature_rows = [row for row in feature_rows if R011.features_available(row, TOPO_FEATURES)]
    x_train, y_train = R011.matrix(feature_rows, TOPO_FEATURES)
    toporisk_model = R011.GradientBoostingClassifier(random_state=args.random_state)
    toporisk_model.fit(x_train, y_train)
    natural = [row for row in feature_rows if row["label"] == 0 and row["dataset"] in {"D1", "D2"}]
    x_nat, _y_nat = R011.matrix(natural, TOPO_FEATURES)
    nat_scores = [float(value) for value in toporisk_model.predict_proba(x_nat)[:, 1]]
    toporisk_threshold = R013OP.threshold_at_fpr_cap(nat_scores, args.toporisk_fpr_cap)

    budget_rows = read_jsonl(pathlib.Path(args.r014_budget_features))
    utility_rows = R014B.attach_budget_features(feature_rows, budget_rows, pathlib.Path(args.d3_positions))
    utility_rows = [row for row in utility_rows if row["effect"] != "missing" and R011.features_available(row, TOPO_FEATURES)]
    corr_model, usable_corr = R014C.fit_binary_model(utility_rows, TOPO_FEATURES, "correction", "extra", args.random_state + 101)
    reg_model, _usable_reg = R014C.fit_binary_model(utility_rows, TOPO_FEATURES, "regression", "extra", args.random_state + 151)
    scored_train = R014C.score_rows(corr_model, reg_model, usable_corr, TOPO_FEATURES, args.utility_penalty)
    selected = R014B.select_threshold_by_training_utility(scored_train, None)
    scorers = GuardScorers(
        toporisk_model=toporisk_model,
        toporisk_threshold=float(toporisk_threshold),
        utility_corr_model=corr_model,
        utility_reg_model=reg_model,
        utility_threshold=float(selected["threshold"]),
        utility_penalty=float(args.utility_penalty),
    )
    metadata = {
        "toporisk_training_count": len(feature_rows),
        "toporisk_natural_threshold_count": len(natural),
        "toporisk_fpr_cap": args.toporisk_fpr_cap,
        "toporisk_threshold": float(toporisk_threshold),
        "utility_training_count": len(utility_rows),
        "utility_threshold": float(selected["threshold"]),
        "utility_penalty": float(args.utility_penalty),
        "utility_train_oracle_agreement_rate": selected["oracle_agreement_rate"],
        "utility_train_trigger_rate": selected["trigger_rate"],
    }
    return scorers, metadata
