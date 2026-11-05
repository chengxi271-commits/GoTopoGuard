#!/usr/bin/env python
"""R003 KataGo one-position analysis schema smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time


DEFAULT_KATAGO_EXE = "external/engines/katago/v1.16.5-eigen/katago.exe"
DEFAULT_KATAGO_CONFIG = "external/engines/katago/v1.16.5-eigen/analysis_example.cfg"
DEFAULT_KATAGO_MODEL = "external/engines/downloads/networks/katago/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz"

REQUIRED_TOP_LEVEL_FIELDS = ["id", "moveInfos", "rootInfo", "policy", "ownership"]
REQUIRED_MOVE_INFO_FIELDS = ["move", "visits", "winrate", "scoreLead", "prior", "pv"]
REQUIRED_ROOT_INFO_FIELDS = ["visits", "winrate", "scoreLead"]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hand_checked_query(*, max_visits: int) -> dict:
    return {
        "id": f"R003_HAND_CHECKED_V{max_visits}",
        "boardXSize": 19,
        "boardYSize": 19,
        "initialPlayer": "B",
        "initialStones": [
            ["B", "D4"],
            ["W", "Q16"],
            ["B", "Q4"],
            ["W", "D16"],
            ["B", "K10"],
            ["W", "K16"],
        ],
        "moves": [],
        "rules": "Chinese",
        "komi": 7.5,
        "maxVisits": max_visits,
        "includePolicy": True,
        "includeOwnership": True,
    }


def validate_response_schema(response: dict, *, expected_id: str) -> dict:
    missing_top = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in response]
    move_infos = response.get("moveInfos")
    missing_move = []
    if not isinstance(move_infos, list) or not move_infos:
        missing_move = REQUIRED_MOVE_INFO_FIELDS[:]
    else:
        first_move = move_infos[0]
        missing_move = [field for field in REQUIRED_MOVE_INFO_FIELDS if field not in first_move]
    root_info = response.get("rootInfo")
    missing_root = []
    if not isinstance(root_info, dict):
        missing_root = REQUIRED_ROOT_INFO_FIELDS[:]
    else:
        missing_root = [field for field in REQUIRED_ROOT_INFO_FIELDS if field not in root_info]
    shape_issues = []
    if response.get("id") != expected_id:
        shape_issues.append({"field": "id", "expected": expected_id, "actual": response.get("id")})
    if "policy" in response and len(response["policy"]) != 19 * 19 + 1:
        shape_issues.append({"field": "policy", "expected_length": 362, "actual_length": len(response["policy"])})
    if "ownership" in response and len(response["ownership"]) != 19 * 19:
        shape_issues.append({"field": "ownership", "expected_length": 361, "actual_length": len(response["ownership"])})
    status = "valid" if not missing_top and not missing_move and not missing_root and not shape_issues else "invalid"
    return {
        "status": status,
        "expected_id": expected_id,
        "missing_top_level_fields": missing_top,
        "missing_move_info_fields": missing_move,
        "missing_root_info_fields": missing_root,
        "shape_issues": shape_issues,
        "move_info_count": len(move_infos) if isinstance(move_infos, list) else 0,
    }


def run_katago_query(katago_exe: pathlib.Path, config: pathlib.Path, model: pathlib.Path, query: dict, *, timeout_seconds: int) -> tuple[dict, str]:
    command = [str(katago_exe), "analysis", "-config", str(config), "-model", str(model)]
    proc = subprocess.Popen(
        command,
        cwd=str(katago_exe.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout, stderr = proc.communicate(json.dumps(query, separators=(",", ":")) + "\n", timeout=timeout_seconds)
    if proc.returncode not in (0, None):
        raise RuntimeError(f"KataGo exited with return code {proc.returncode}: {stderr[-2000:]}")
    response_lines = [line for line in stdout.splitlines() if line.strip().startswith("{")]
    if not response_lines:
        raise RuntimeError(f"KataGo produced no JSON response. stderr tail: {stderr[-2000:]}")
    return json.loads(response_lines[-1]), stderr


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    katago_exe = pathlib.Path(args.katago_exe).resolve()
    config = pathlib.Path(args.config).resolve()
    model = pathlib.Path(args.model).resolve()
    query = build_hand_checked_query(max_visits=args.max_visits)
    backend_manifest = {
        "run_id": "R003",
        "generated_at_unix": int(time.time()),
        "evidence_class": "engine schema smoke only",
        "katago_exe": str(katago_exe),
        "config": str(config),
        "model": str(model),
        "katago_exe_sha256": sha256_file(katago_exe),
        "config_sha256": sha256_file(config),
        "model_sha256": sha256_file(model),
        "max_visits": args.max_visits,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "r003_query.json", query)
    try:
        response, stderr = run_katago_query(katago_exe, config, model, query, timeout_seconds=args.timeout_seconds)
        schema = validate_response_schema(response, expected_id=query["id"])
        report = {
            "run_id": "R003",
            "generated_at_unix": int(time.time()),
            "evidence_class": "engine schema smoke only",
            "status": "done" if schema["status"] == "valid" else "schema_invalid",
            "schema_validation": schema,
            "backend_manifest_path": str(output_root / "backend_manifest.json"),
            "raw_response_path": str(output_root / "raw_katago_response.json"),
        }
        write_json(output_root / "raw_katago_response.json", response)
        (output_root / "raw_katago_stderr.log").write_text(stderr, encoding="utf-8")
    except Exception as exc:
        report = {
            "run_id": "R003",
            "generated_at_unix": int(time.time()),
            "evidence_class": "engine schema smoke only",
            "status": "blocked",
            "error": str(exc),
            "backend_manifest_path": str(output_root / "backend_manifest.json"),
        }
    write_json(output_root / "backend_manifest.json", backend_manifest)
    write_json(output_root / "r003_katago_schema_smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_issue and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--katago-exe", default=DEFAULT_KATAGO_EXE)
    parser.add_argument("--config", default=DEFAULT_KATAGO_CONFIG)
    parser.add_argument("--model", default=DEFAULT_KATAGO_MODEL)
    parser.add_argument("--output-root", default="experiments/results/R003_KATAGO_SCHEMA_SMOKE")
    parser.add_argument("--max-visits", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
