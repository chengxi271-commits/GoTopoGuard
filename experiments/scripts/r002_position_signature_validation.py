#!/usr/bin/env python
"""R002 position-signature validation for D0/R001 records.

This gate checks legality/replay status and semantic duplicate signatures.
It is provenance/integrity evidence only, not a paper performance result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import Counter, defaultdict


DEFAULT_REQUIRED_SOURCES = {
    "jgdb": 50,
    "katago_kata1_2020_12_12": 50,
    "go_attack_site_examples": 20,
}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def validate_signature_records(records_path: pathlib.Path | str, *, required_sources: dict[str, int]) -> dict:
    records_path = pathlib.Path(records_path)
    rows = read_jsonl(records_path)
    status_counts = Counter(row.get("status") for row in rows)
    valid_rows = [row for row in rows if row.get("status") == "valid" and row.get("position_signature")]
    by_source = {}
    for source_id in sorted({row.get("source_id", "unknown") for row in rows}):
        source_rows = [row for row in rows if row.get("source_id", "unknown") == source_id]
        source_valid = [row for row in source_rows if row.get("status") == "valid" and row.get("position_signature")]
        source_status_counts = Counter(row.get("status") for row in source_rows)
        by_source[source_id] = {
            "row_count": len(source_rows),
            "valid_count": len(source_valid),
            "unique_signature_count": len({row["position_signature"] for row in source_valid}),
            "status_counts": dict(source_status_counts),
        }
    signature_groups = defaultdict(list)
    for row in valid_rows:
        signature_groups[row["position_signature"]].append(
            {
                "source_id": row.get("source_id"),
                "name": row.get("name"),
                "line_number": row.get("_line_number"),
            }
        )
    duplicate_signatures = [
        {
            "position_signature": signature,
            "count": len(group),
            "records": group,
        }
        for signature, group in sorted(signature_groups.items())
        if len(group) > 1
    ]
    completion_failures = {}
    for source_id, required_count in required_sources.items():
        valid_count = by_source.get(source_id, {}).get("valid_count", 0)
        if valid_count < required_count:
            completion_failures[source_id] = {"required": required_count, "valid": valid_count}
    status = "done"
    if duplicate_signatures:
        status = "duplicate_signatures_found"
    if completion_failures:
        status = "completion_failures"
    return {
        "run_id": "R002",
        "generated_at_unix": int(time.time()),
        "evidence_class": "provenance/integrity only",
        "records_path": str(records_path),
        "row_count": len(rows),
        "valid_count": len(valid_rows),
        "unique_signature_count": len(signature_groups),
        "duplicate_signature_count": len(duplicate_signatures),
        "duplicate_record_count": sum(item["count"] for item in duplicate_signatures),
        "duplicate_signatures": duplicate_signatures,
        "status_counts": dict(status_counts),
        "by_source": by_source,
        "required_sources": dict(required_sources),
        "completion_failures": completion_failures,
        "status": status,
    }


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_required_sources(args: argparse.Namespace) -> dict[str, int]:
    return {
        "jgdb": args.required_jgdb,
        "katago_kata1_2020_12_12": args.required_katago,
        "go_attack_site_examples": args.required_go_attack,
    }


def run(args: argparse.Namespace) -> int:
    report = validate_signature_records(args.records, required_sources=build_required_sources(args))
    output_root = pathlib.Path(args.output_root)
    write_json(output_root / "r002_position_signature_validation_report.json", report)
    write_jsonl(output_root / "duplicate_position_signatures.jsonl", report["duplicate_signatures"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_issue and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="experiments/results/R001_SGF_REPLAY_SMOKE/sgf_replay_records.jsonl")
    parser.add_argument("--output-root", default="experiments/results/R002_POSITION_SIGNATURE_VALIDATION")
    parser.add_argument("--required-jgdb", type=int, default=DEFAULT_REQUIRED_SOURCES["jgdb"])
    parser.add_argument("--required-katago", type=int, default=DEFAULT_REQUIRED_SOURCES["katago_kata1_2020_12_12"])
    parser.add_argument("--required-go-attack", type=int, default=DEFAULT_REQUIRED_SOURCES["go_attack_site_examples"])
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
