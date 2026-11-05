#!/usr/bin/env python
"""Build R008 adversarial replay split from public Go Attack examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from collections import Counter


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R001 = load_script_module("r001_sgf_replay_smoke", SCRIPT_DIR / "r001_sgf_replay_smoke.py")
R004 = load_script_module("r004_katago_100_position_smoke", SCRIPT_DIR / "r004_katago_100_position_smoke.py")
R006 = load_script_module("r006_r007_natural_split_build", SCRIPT_DIR / "r006_r007_natural_split_build.py")

DEFAULT_LEAD_MOVES = [80, 40, 20, 10, 5, 1, 0]


def lead_time_prefixes(move_count: int, lead_moves: list[int]) -> list[tuple[int, int]]:
    prefixes = []
    for lead in sorted(set(lead_moves), reverse=True):
        prefix_len = move_count - lead
        if prefix_len < 0:
            continue
        prefixes.append((lead, prefix_len))
    return prefixes


def failure_window_bucket(lead_moves: int) -> str:
    if lead_moves == 0:
        return "terminal"
    if lead_moves <= 5:
        return "near_1_5"
    if lead_moves <= 20:
        return "mid_6_20"
    if lead_moves <= 40:
        return "early_21_40"
    return "early_41_80"


def build_adversarial_records(
    sgf_records: list[tuple[str, str]],
    *,
    lead_moves: list[int],
    run_id: str,
    split_id: str,
    required_board_size: int,
) -> tuple[list[dict], dict]:
    records = []
    seen_signatures = set()
    status_counts = Counter()
    lead_counts = Counter()
    bucket_counts = Counter()
    duplicate_skipped = 0
    unavailable_prefix_count = 0
    valid_game_count = 0
    requested_available_prefix_count = 0
    for game_index, (name, sgf_text) in enumerate(sgf_records, start=1):
        try:
            game = R001.parse_sgf_mainline(sgf_text)
            if game.board_size != required_board_size:
                status_counts["rejected_board_size"] += 1
                continue
            full_result = R001.replay_game(game)
            if not full_result.ok:
                status_counts["invalid_replay"] += 1
                continue
            valid_game_count += 1
            available_prefixes = lead_time_prefixes(len(game.moves), lead_moves)
            unavailable_prefix_count += len(set(lead_moves)) - len(available_prefixes)
            requested_available_prefix_count += len(available_prefixes)
            for lead, prefix_len in available_prefixes:
                signature, board_rows = R006.prefix_position_signature(game, prefix_len)
                if signature in seen_signatures:
                    duplicate_skipped += 1
                    continue
                seen_signatures.add(signature)
                bucket = failure_window_bucket(lead)
                lead_counts[str(lead)] += 1
                bucket_counts[bucket] += 1
                record_index = len(records) + 1
                records.append(
                    {
                        "position_id": f"{run_id}_go_attack_{record_index:04d}",
                        "run_id": run_id,
                        "split_id": split_id,
                        "source_id": "go_attack_site_examples",
                        "source_name": name,
                        "source_url": R001.DEFAULT_GO_ATTACK_SITE_BASE,
                        "board_size": game.board_size,
                        "komi": game.komi,
                        "rules": game.rules,
                        "game_index": game_index,
                        "game_move_count": len(game.moves),
                        "move_prefix_len": prefix_len,
                        "lead_moves_to_terminal": lead,
                        "failure_window_bucket": bucket,
                        "failure_endpoint_proxy": "sgf_terminal_position",
                        "label_basis": "public_go_attack_example_relative_to_terminal",
                        "position_signature": signature,
                        "board": board_rows,
                        "katago_query_compatible_komi": R004.is_katago_komi_compatible(game.komi),
                        "evidence_class": "adversarial replay split construction only",
                    }
                )
            status_counts["valid"] += 1
        except Exception:
            status_counts["parse_or_prefix_error"] += 1
    summary = {
        "valid_game_count": valid_game_count,
        "status_counts": dict(status_counts),
        "requested_lead_moves": sorted(set(lead_moves), reverse=True),
        "requested_available_prefix_count": requested_available_prefix_count,
        "unavailable_prefix_count": unavailable_prefix_count,
        "duplicate_signature_skipped": duplicate_skipped,
        "lead_counts": dict(lead_counts),
        "failure_window_bucket_counts": dict(bucket_counts),
    }
    return records, summary


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    output_root = pathlib.Path(args.output_root)
    lead_moves = [int(value) for value in args.lead_moves.split(",") if value.strip()]
    sgf_records, source_summary = R004.fetch_go_attack_site_records_with_retries(
        args.go_attack_count,
        attempts=args.source_fetch_retries,
    )
    records, build_summary = build_adversarial_records(
        sgf_records,
        lead_moves=lead_moves,
        run_id=args.run_id,
        split_id=args.split_id,
        required_board_size=args.required_board_size,
    )
    status = "done"
    if len(sgf_records) != args.go_attack_count or build_summary["valid_game_count"] != args.go_attack_count:
        status = "incomplete"
    if not records:
        status = "empty"
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "adversarial replay split construction only",
        "status": status,
        "split_id": args.split_id,
        "source_id": "go_attack_site_examples",
        "source_url": R001.DEFAULT_GO_ATTACK_SITE_BASE,
        "target_game_count": args.go_attack_count,
        "fetched_game_count": len(sgf_records),
        "position_count": len(records),
        "unique_signature_count": len({record["position_signature"] for record in records}),
        **build_summary,
    }
    source_manifest = {
        "run_id": args.run_id,
        "split_id": args.split_id,
        "source_id": "go_attack_site_examples",
        "source_url": R001.DEFAULT_GO_ATTACK_SITE_BASE,
        "site_js_url": R001.DEFAULT_GO_ATTACK_SITE_JS_URL,
        "provenance_state": "source-found-not-citable-yet",
        "target_game_count": args.go_attack_count,
        "fetched_game_count": len(sgf_records),
        "source_fetch_summary": source_summary,
        "evidence_class": "adversarial replay split construction only",
    }
    write_jsonl(output_root / "positions.jsonl", records)
    write_json(output_root / "split_report.json", report)
    write_json(output_root / "source_manifest.json", source_manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="R008")
    parser.add_argument("--split-id", default="D3_ADVERSARIAL_GO_ATTACK")
    parser.add_argument("--go-attack-count", type=int, default=20)
    parser.add_argument("--lead-moves", default="80,40,20,10,5,1,0")
    parser.add_argument("--output-root", default="experiments/results/R008_ADVERSARIAL_REPLAY_SPLIT")
    parser.add_argument("--source-fetch-retries", type=int, default=5)
    parser.add_argument("--required-board-size", type=int, default=19)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
