#!/usr/bin/env python
"""Build R006/R007 natural Go position splits from public SGF archives."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
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

WINDOWS = [
    ("early_40_80", 40, 80, 60),
    ("mid_80_140", 80, 140, 110),
    ("late_140_220", 140, 220, 180),
]

DEFAULT_KATAGO_URLS = [
    R001.DEFAULT_KATAGO_URL,
    "https://katagoarchive.org/kata1/traininggames/2021-01-15sgfs.tar.bz2",
    "https://katagoarchive.org/kata1/traininggames/2021-03-15sgfs.tar.bz2",
    "https://katagoarchive.org/kata1/traininggames/2021-05-30sgfs.tar.bz2",
    "https://katagoarchive.org/kata1/traininggames/2021-07-10sgfs.tar.bz2",
    "https://katagoarchive.org/kata1/traininggames/2021-09-12sgfs.tar.bz2",
    "https://katagoarchive.org/kata1/traininggames/2021-10-30sgfs.tar.bz2",
]


def window_target_counts(total_count: int, window_names: list[str]) -> dict[str, int]:
    base = total_count // len(window_names)
    remainder = total_count % len(window_names)
    return {name: base + (1 if index < remainder else 0) for index, name in enumerate(window_names)}


def candidate_prefixes_for_game(move_count: int) -> list[tuple[str, int]]:
    candidates = []
    for name, _start, _end, center in WINDOWS:
        if move_count >= center:
            candidates.append((name, center))
    return candidates


def prefix_position_signature(game, prefix_len: int) -> tuple[str, list[str]]:
    prefix_game = R001.Game(
        board_size=game.board_size,
        komi=game.komi,
        rules=game.rules,
        setup_stones=game.setup_stones,
        moves=game.moves[:prefix_len],
    )
    result = R001.replay_game(prefix_game)
    if not result.ok:
        raise ValueError("; ".join(result.errors))
    board_rows = ["".join(row) for row in result.final_board]
    return R001.position_signature(result.final_board), board_rows


def katago_source_urls(urls: list[str] | None) -> list[str]:
    if urls:
        return urls
    return list(DEFAULT_KATAGO_URLS)


def katago_source_id(url: str) -> str:
    match = re.search(r"/(\d{4}-\d{2}-\d{2})sgfs\.tar\.bz2$", url)
    if not match:
        return "katago_kata1_unknown"
    return f"katago_kata1_{match.group(1).replace('-', '_')}"


def source_iterator(source: str, args: argparse.Namespace):
    if source == "jgdb":
        for name, sgf in R001.iter_tar_sgfs(args.jgdb_url, "r|gz", max_scan=args.max_scan):
            yield "jgdb", args.jgdb_url, name, sgf
        return
    if source == "katago":
        for url in katago_source_urls(args.katago_urls):
            for name, sgf in R001.iter_tar_sgfs(url, "r|bz2", max_scan=args.max_scan):
                yield katago_source_id(url), url, name, sgf
        return
    raise ValueError(f"unknown source {source}")


def source_id_for(source: str) -> str:
    if source == "jgdb":
        return "jgdb"
    if source == "katago":
        return "katago_kata1_multi"
    raise ValueError(f"unknown source {source}")


def build_split(args: argparse.Namespace) -> tuple[list[dict], dict]:
    source_id = source_id_for(args.source)
    window_names = [name for name, _start, _end, _center in WINDOWS]
    targets = window_target_counts(args.target_count, window_names)
    counts = Counter()
    records = []
    source_status_counts = Counter()
    scanned = 0
    duplicate_signatures = Counter()
    per_source_counts = Counter()
    for item_source_id, item_source_url, name, sgf in source_iterator(args.source, args):
        if sum(counts.values()) >= args.target_count:
            break
        scanned += 1
        per_source_counts[item_source_id] += 1
        try:
            game = R001.parse_sgf_mainline(sgf)
            if game.board_size != args.required_board_size:
                source_status_counts["rejected_board_size"] += 1
                continue
            full_result = R001.replay_game(game)
            if not full_result.ok:
                source_status_counts["invalid_replay"] += 1
                continue
            for window_name, prefix_len in candidate_prefixes_for_game(len(game.moves)):
                if counts[window_name] >= targets[window_name]:
                    continue
                signature, board_rows = prefix_position_signature(game, prefix_len)
                if duplicate_signatures[signature]:
                    source_status_counts["duplicate_prefix_signature_skipped"] += 1
                    continue
                duplicate_signatures[signature] += 1
                counts[window_name] += 1
                record_index = len(records) + 1
                records.append(
                    {
                        "position_id": f"{args.run_id}_{source_id}_{record_index:05d}",
                        "run_id": args.run_id,
                        "split_id": args.split_id,
                        "source_id": item_source_id,
                        "source_name": name,
                        "source_url": item_source_url,
                        "board_size": game.board_size,
                        "komi": game.komi,
                        "rules": game.rules,
                        "game_move_count": len(game.moves),
                        "move_prefix_len": prefix_len,
                        "window": window_name,
                        "position_signature": signature,
                        "board": board_rows,
                        "katago_query_compatible_komi": R004.is_katago_komi_compatible(game.komi),
                        "evidence_class": "split construction only",
                    }
                )
                if sum(counts.values()) >= args.target_count:
                    break
            source_status_counts["accepted_games_with_positions"] += 1
        except Exception:
            source_status_counts["parse_or_prefix_error"] += 1
    report = {
        "run_id": args.run_id,
        "generated_at_unix": int(time.time()),
        "evidence_class": "split construction only",
        "status": "done" if len(records) == args.target_count else "incomplete",
        "split_id": args.split_id,
        "source_id": source_id,
        "source_url": args.jgdb_url if args.source == "jgdb" else None,
        "source_urls": [args.jgdb_url] if args.source == "jgdb" else katago_source_urls(args.katago_urls),
        "target_count": args.target_count,
        "position_count": len(records),
        "window_target_counts": targets,
        "window_counts": dict(counts),
        "scanned_sgf_count": scanned,
        "source_status_counts": dict(source_status_counts),
        "per_source_scanned_sgf_count": dict(per_source_counts),
        "unique_signature_count": len({record["position_signature"] for record in records}),
        "katago_query_compatible_count": sum(1 for record in records if record["katago_query_compatible_komi"]),
    }
    return records, report


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
    records, report = build_split(args)
    write_jsonl(output_root / "positions.jsonl", records)
    write_json(output_root / "split_report.json", report)
    write_json(
        output_root / "source_manifest.json",
        {
            "run_id": args.run_id,
            "split_id": args.split_id,
            "source_id": report["source_id"],
            "source_url": report["source_url"],
            "source_urls": report["source_urls"],
            "provenance_state": "source-found-not-citable-yet",
            "target_count": args.target_count,
            "position_count": len(records),
            "evidence_class": "split construction only",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete and report["status"] != "done":
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--source", choices=["jgdb", "katago"], required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jgdb-url", default=R001.DEFAULT_JGDB_URL)
    parser.add_argument("--katago-url", dest="katago_urls", action="append", default=None)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--required-board-size", type=int, default=19)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
