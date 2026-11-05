#!/usr/bin/env python
"""R001 SGF replay smoke.

This is a provenance/replay smoke tool, not final paper evidence.
It streams public SGF archives when possible, replays a bounded sample, and
records legality, metadata, and duplicate final-position signatures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import re
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter


DEFAULT_JGDB_URL = "https://data.pjreddie.com/files/jgdb.tar.gz"
DEFAULT_KATAGO_URL = "https://katagoarchive.org/kata1/traininggames/2020-12-12sgfs.tar.bz2"
DEFAULT_GO_ATTACK_TREE_URL = "https://api.github.com/repos/AlignmentResearch/go_attack/git/trees/main?recursive=1"
DEFAULT_GO_ATTACK_SITE_BASE = "https://goattack.far.ai"
DEFAULT_GO_ATTACK_SITE_JS_URL = f"{DEFAULT_GO_ATTACK_SITE_BASE}/assets/index.fd610701.js"
USER_AGENT = "Mozilla/5.0 (GoTopoBench R001 SGF replay smoke)"


class Move:
    def __init__(self, color: str, coord: tuple[int, int] | None):
        self.color = color
        self.coord = coord


class Game:
    def __init__(
        self,
        *,
        board_size: int,
        komi: str | None,
        rules: str | None,
        setup_stones: list[tuple[str, tuple[int, int]]],
        moves: list[Move],
    ):
        self.board_size = board_size
        self.komi = komi
        self.rules = rules
        self.setup_stones = setup_stones
        self.moves = moves


class ReplayResult:
    def __init__(
        self,
        *,
        ok: bool,
        errors: list[str],
        final_board: list[list[str]],
        captures: dict[str, int],
        moves_played: int,
        pass_moves: int,
    ):
        self.ok = ok
        self.errors = errors
        self.final_board = final_board
        self.captures = captures
        self.moves_played = moves_played
        self.pass_moves = pass_moves


def decode_coord(raw: str | None, board_size: int) -> tuple[int, int] | None:
    if raw is None or raw == "" or raw.lower() == "tt":
        return None
    if len(raw) < 2:
        raise ValueError(f"invalid coordinate {raw!r}")
    col = ord(raw[0]) - ord("a")
    row = ord(raw[1]) - ord("a")
    if col < 0 or row < 0 or col >= board_size or row >= board_size:
        raise ValueError(f"coordinate out of bounds {raw!r} for board size {board_size}")
    return (col, row)


def _mainline_nodes(sgf: str) -> list[str]:
    nodes: list[str] = []
    depth = 0
    in_value = False
    escaped = False
    current: list[str] | None = None
    for ch in sgf:
        if in_value:
            if current is not None:
                current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "]":
                in_value = False
            continue
        if ch == "[":
            in_value = True
            if current is not None:
                current.append(ch)
            continue
        if ch == "(":
            depth += 1
            if current is not None:
                current.append(ch)
            continue
        if ch == ")":
            if current is not None:
                nodes.append("".join(current).strip())
                current = None
            depth -= 1
            continue
        if ch == ";" and depth == 1:
            if current is not None:
                nodes.append("".join(current).strip())
            current = []
            continue
        if current is not None:
            current.append(ch)
    if current is not None:
        nodes.append("".join(current).strip())
    return [node for node in nodes if node]


def _parse_properties(node: str) -> dict[str, list[str]]:
    props: dict[str, list[str]] = {}
    i = 0
    while i < len(node):
        while i < len(node) and not node[i].isalpha():
            i += 1
        start = i
        while i < len(node) and node[i].isalpha():
            i += 1
        if start == i:
            break
        prop = node[start:i]
        values: list[str] = []
        while i < len(node) and node[i] == "[":
            i += 1
            value: list[str] = []
            escaped = False
            while i < len(node):
                ch = node[i]
                i += 1
                if escaped:
                    value.append(ch)
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "]":
                    break
                else:
                    value.append(ch)
            values.append("".join(value))
        props.setdefault(prop, []).extend(values)
    return props


def parse_sgf_mainline(sgf: str) -> Game:
    nodes = _mainline_nodes(sgf)
    if not nodes:
        raise ValueError("no SGF mainline nodes found")
    root = _parse_properties(nodes[0])
    if root.get("GM", ["1"])[0] not in ("1", ""):
        raise ValueError(f"unsupported SGF game type {root.get('GM')}")
    board_size = int(root.get("SZ", ["19"])[0].split(":")[0])
    komi = root.get("KM", [None])[0]
    rules = root.get("RU", [None])[0]
    setup_stones: list[tuple[str, tuple[int, int]]] = []
    moves: list[Move] = []
    for node in nodes:
        props = _parse_properties(node)
        for prop, color in (("AB", "B"), ("AW", "W")):
            for raw in props.get(prop, []):
                coord = decode_coord(raw, board_size)
                if coord is not None:
                    setup_stones.append((color, coord))
        for color in ("B", "W"):
            if color in props:
                moves.append(Move(color, decode_coord(props[color][0], board_size)))
    return Game(board_size=board_size, komi=komi, rules=rules, setup_stones=setup_stones, moves=moves)


def _neighbors(col: int, row: int, size: int):
    if col > 0:
        yield col - 1, row
    if col + 1 < size:
        yield col + 1, row
    if row > 0:
        yield col, row - 1
    if row + 1 < size:
        yield col, row + 1


def _group_and_liberties(board: list[list[str]], col: int, row: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    color = board[row][col]
    group = {(col, row)}
    liberties: set[tuple[int, int]] = set()
    stack = [(col, row)]
    while stack:
        c, r = stack.pop()
        for nc, nr in _neighbors(c, r, len(board)):
            stone = board[nr][nc]
            if stone == ".":
                liberties.add((nc, nr))
            elif stone == color and (nc, nr) not in group:
                group.add((nc, nr))
                stack.append((nc, nr))
    return group, liberties


def replay_game(game: Game) -> ReplayResult:
    board = [["." for _ in range(game.board_size)] for _ in range(game.board_size)]
    errors: list[str] = []
    captures = {"B": 0, "W": 0}
    suicide_legal = "sui1" in (game.rules or "")
    for color, coord in game.setup_stones:
        col, row = coord
        if board[row][col] != ".":
            errors.append(f"setup occupied point {coord}")
            return ReplayResult(ok=False, errors=errors, final_board=board, captures=captures, moves_played=0, pass_moves=0)
        board[row][col] = color
    moves_played = 0
    pass_moves = 0
    for index, move in enumerate(game.moves, start=1):
        if move.coord is None:
            pass_moves += 1
            continue
        col, row = move.coord
        if board[row][col] != ".":
            errors.append(f"move {index}: occupied point {move.coord}")
            return ReplayResult(ok=False, errors=errors, final_board=board, captures=captures, moves_played=moves_played, pass_moves=pass_moves)
        board[row][col] = move.color
        moves_played += 1
        opponent = "W" if move.color == "B" else "B"
        captured: set[tuple[int, int]] = set()
        for nc, nr in _neighbors(col, row, game.board_size):
            if board[nr][nc] != opponent:
                continue
            group, liberties = _group_and_liberties(board, nc, nr)
            if not liberties:
                captured.update(group)
        for cc, rr in captured:
            board[rr][cc] = "."
        captures[move.color] += len(captured)
        own_group, own_liberties = _group_and_liberties(board, col, row)
        if not own_liberties:
            if not suicide_legal:
                errors.append(f"move {index}: suicide at {move.coord}")
                return ReplayResult(ok=False, errors=errors, final_board=board, captures=captures, moves_played=moves_played, pass_moves=pass_moves)
            for cc, rr in own_group:
                board[rr][cc] = "."
            captures[opponent] += len(own_group)
    return ReplayResult(ok=True, errors=errors, final_board=board, captures=captures, moves_played=moves_played, pass_moves=pass_moves)


def position_signature(board: list[list[str]]) -> str:
    payload = f"{len(board)}|" + "/".join("".join(row) for row in board)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _urlopen(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def iter_tar_sgfs(url: str, mode: str, *, max_scan: int):
    scanned = 0
    with _urlopen(url, timeout=120) as response:
        with tarfile.open(fileobj=response, mode=mode) as archive:
            for member in archive:
                if scanned >= max_scan:
                    break
                if not member.isfile() or not member.name.lower().endswith(".sgf"):
                    continue
                scanned += 1
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                raw = extracted.read()
                yield member.name, raw.decode("utf-8", "replace")


def replay_source_records(source_id: str, records, *, required_board_size: int, limit: int):
    rows = []
    scanned = 0
    accepted = 0
    for name, sgf in records:
        scanned += 1
        try:
            game = parse_sgf_mainline(sgf)
            if game.board_size != required_board_size:
                rows.append({"source_id": source_id, "name": name, "status": "rejected_board_size", "board_size": game.board_size})
                continue
            result = replay_game(game)
            row = {
                "source_id": source_id,
                "name": name,
                "status": "valid" if result.ok else "invalid_replay",
                "board_size": game.board_size,
                "komi": game.komi,
                "rules": game.rules,
                "move_count": len(game.moves),
                "moves_played": result.moves_played,
                "pass_moves": result.pass_moves,
                "captures": result.captures,
                "position_signature": position_signature(result.final_board) if result.ok else None,
                "errors": result.errors,
            }
            rows.append(row)
            if result.ok:
                accepted += 1
            if accepted >= limit:
                break
        except Exception as exc:  # keep source-level failures visible.
            rows.append({"source_id": source_id, "name": name, "status": "parse_error", "errors": [str(exc)]})
    return rows, {"source_id": source_id, "scanned_sgf_count": scanned, "accepted_valid_count": accepted}


def replay_tar_source_with_retries(
    source_id: str,
    url: str,
    mode: str,
    *,
    max_scan: int,
    required_board_size: int,
    limit: int,
    attempts: int,
):
    errors = []
    for attempt in range(1, max(1, attempts) + 1):
        try:
            rows, summary = replay_source_records(
                source_id,
                iter_tar_sgfs(url, mode, max_scan=max_scan),
                required_board_size=required_board_size,
                limit=limit,
            )
            summary["attempts_used"] = attempt
            if errors:
                summary["previous_errors"] = errors
            return rows, summary
        except Exception as exc:
            errors.append({"attempt": attempt, "error": str(exc)})
    return (
        [{"source_id": source_id, "status": "source_fetch_failed", "errors": [errors[-1]["error"] if errors else "unknown fetch failure"]}],
        {"source_id": source_id, "status": "source_fetch_failed", "attempts_used": max(1, attempts), "errors": errors},
    )


def fetch_go_attack_sgf_records(limit: int):
    try:
        with _urlopen(DEFAULT_GO_ATTACK_TREE_URL, timeout=30) as response:
            tree = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:
        return [], {"status": "probe_failed", "error": str(exc), "tree_url": DEFAULT_GO_ATTACK_TREE_URL}
    paths = [entry["path"] for entry in tree.get("tree", []) if entry.get("type") == "blob" and entry.get("path", "").lower().endswith(".sgf")]
    records = []
    for path in paths[:limit]:
        raw_url = f"https://raw.githubusercontent.com/AlignmentResearch/go_attack/main/{path}"
        try:
            with _urlopen(raw_url, timeout=30) as response:
                records.append((path, response.read().decode("utf-8", "replace")))
        except urllib.error.HTTPError:
            continue
    status = "ok" if records else "no_sgf_files_found_in_repo_tree"
    return records, {"status": status, "tree_url": DEFAULT_GO_ATTACK_TREE_URL, "sgf_path_count": len(paths), "fetched_count": len(records)}


def extract_go_attack_site_dir_names(script_text: str) -> list[str]:
    seen = set()
    dir_names = []
    for match in re.finditer(r'dir_name:"([^"]+)"', script_text):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            dir_names.append(name)
    return dir_names


def go_attack_site_sgf_url(dir_name: str, sgf_path: str) -> str:
    basename = sgf_path.replace("\\", "/").split("/")[-1]
    return f"{DEFAULT_GO_ATTACK_SITE_BASE}/sgfs/{dir_name}/{basename}"


def _select_sgf_text(blob: str, sgf_line: str | None) -> str:
    stripped = blob.strip()
    if stripped.startswith("(;"):
        return stripped
    lines = [line.strip() for line in blob.splitlines() if line.strip()]
    if not lines:
        return stripped
    try:
        index = int(sgf_line or "1") - 1
    except ValueError:
        index = 0
    if 0 <= index < len(lines):
        return lines[index]
    return lines[0]


def fetch_go_attack_site_sgf_records(limit: int):
    summary = {
        "status": "started",
        "site_base": DEFAULT_GO_ATTACK_SITE_BASE,
        "site_js_url": DEFAULT_GO_ATTACK_SITE_JS_URL,
        "dir_count": 0,
        "csv_probe_count": 0,
        "csv_success_count": 0,
        "candidate_count": 0,
        "fetched_count": 0,
        "errors": [],
    }
    try:
        with _urlopen(DEFAULT_GO_ATTACK_SITE_JS_URL, timeout=60) as response:
            script_text = response.read().decode("utf-8", "replace")
    except Exception as exc:
        summary.update({"status": "site_js_fetch_failed", "errors": [str(exc)]})
        return [], summary
    dir_names = extract_go_attack_site_dir_names(script_text)
    summary["dir_count"] = len(dir_names)
    records = []
    seen_urls = set()
    for dir_name in dir_names:
        if len(records) >= limit:
            break
        csv_url = f"{DEFAULT_GO_ATTACK_SITE_BASE}/sgfs/{dir_name}/game_infos.csv"
        summary["csv_probe_count"] += 1
        try:
            with _urlopen(csv_url, timeout=60) as response:
                csv_text = response.read().decode("utf-8", "replace")
        except Exception as exc:
            summary["errors"].append({"dir_name": dir_name, "csv_url": csv_url, "error": str(exc)})
            continue
        summary["csv_success_count"] += 1
        try:
            rows = list(csv.DictReader(io.StringIO(csv_text)))
        except Exception as exc:
            summary["errors"].append({"dir_name": dir_name, "csv_url": csv_url, "error": f"csv_parse_failed: {exc}"})
            continue
        for row_index, row in enumerate(rows, start=1):
            if len(records) >= limit:
                break
            sgf_path = row.get("sgf_path") or ""
            if not sgf_path:
                continue
            summary["candidate_count"] += 1
            sgf_url = go_attack_site_sgf_url(dir_name, sgf_path)
            if sgf_url in seen_urls:
                continue
            seen_urls.add(sgf_url)
            try:
                with _urlopen(sgf_url, timeout=60) as response:
                    sgf_blob = response.read().decode("utf-8", "replace")
            except Exception as exc:
                summary["errors"].append({"dir_name": dir_name, "sgf_url": sgf_url, "error": str(exc)})
                continue
            sgf_text = _select_sgf_text(sgf_blob, row.get("sgf_line"))
            if not sgf_text.lstrip().startswith("(;"):
                summary["errors"].append({"dir_name": dir_name, "sgf_url": sgf_url, "error": "downloaded content is not SGF"})
                continue
            name = f"{dir_name}/row{row_index}/{sgf_url.split('/')[-1]}"
            records.append((name, sgf_text))
            summary["fetched_count"] = len(records)
    summary["status"] = "ok" if records else "no_public_site_sgfs_fetched"
    return records, summary


def summarize(rows_by_source: dict[str, list[dict]], source_summaries: list[dict]) -> dict:
    rows = [row for source_rows in rows_by_source.values() for row in source_rows]
    valid = [row for row in rows if row.get("status") == "valid"]
    signature_counts = Counter(row["position_signature"] for row in valid)
    duplicates = {sig: count for sig, count in signature_counts.items() if count > 1}
    by_source = {}
    for source_id, source_rows in rows_by_source.items():
        statuses = Counter(row.get("status") for row in source_rows)
        by_source[source_id] = {
            "row_count": len(source_rows),
            "valid_count": statuses.get("valid", 0),
            "status_counts": dict(statuses),
            "rules_values": sorted({row.get("rules") for row in source_rows if row.get("rules")}),
            "komi_values_sample": sorted({row.get("komi") for row in source_rows if row.get("komi")})[:20],
        }
    return {
        "run_id": "R001",
        "generated_at_unix": int(time.time()),
        "evidence_class": "smoke/provenance only",
        "total_rows": len(rows),
        "total_valid": len(valid),
        "duplicate_final_position_signature_count": len(duplicates),
        "duplicate_final_position_signatures": duplicates,
        "by_source": by_source,
        "source_summaries": source_summaries,
    }


def completion_failures(report: dict, args: argparse.Namespace) -> dict:
    by_source = report.get("by_source", {})
    required = {
        "jgdb": args.jgdb_limit,
        "katago_kata1_2020_12_12": args.katago_limit,
    }
    failures = {}
    for source_id, required_count in required.items():
        valid_count = by_source.get(source_id, {}).get("valid_count", 0)
        if valid_count < required_count:
            failures[source_id] = {"required": required_count, "valid": valid_count}
    adversarial_valid = (
        by_source.get("go_attack_site_examples", {}).get("valid_count", 0)
        + by_source.get("go_attack_repo_probe", {}).get("valid_count", 0)
    )
    if adversarial_valid < args.go_attack_limit:
        failures["go_attack_examples"] = {"required": args.go_attack_limit, "valid": adversarial_valid}
    return failures


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
    rows_by_source: dict[str, list[dict]] = {}
    source_summaries: list[dict] = [
        {
            "source_id": "jgdb",
            "url": args.jgdb_url,
            "provenance_state": "source-found-not-citable-yet",
            "target_count": args.jgdb_limit,
        },
        {
            "source_id": "katago_kata1_2020_12_12",
            "url": args.katago_url,
            "provenance_state": "source-found-not-citable-yet",
            "target_count": args.katago_limit,
        },
        {
            "source_id": "go_attack_site_examples",
            "url": DEFAULT_GO_ATTACK_SITE_BASE,
            "provenance_state": "source-found-not-citable-yet",
            "target_count": args.go_attack_limit,
        },
    ]
    rows, summary = replay_tar_source_with_retries(
        "jgdb",
        args.jgdb_url,
        "r|gz",
        max_scan=args.jgdb_max_scan,
        required_board_size=args.required_board_size,
        limit=args.jgdb_limit,
        attempts=args.source_fetch_retries,
    )
    rows_by_source["jgdb"] = rows
    source_summaries.append(summary)
    rows, summary = replay_tar_source_with_retries(
        "katago_kata1_2020_12_12",
        args.katago_url,
        "r|bz2",
        max_scan=args.katago_max_scan,
        required_board_size=args.required_board_size,
        limit=args.katago_limit,
        attempts=args.source_fetch_retries,
    )
    rows_by_source["katago_kata1_2020_12_12"] = rows
    source_summaries.append(summary)
    go_attack_records, go_attack_summary = fetch_go_attack_sgf_records(args.go_attack_limit)
    source_summaries.append({"source_id": "go_attack_repo_probe", **go_attack_summary})
    site_go_attack_records, site_go_attack_summary = fetch_go_attack_site_sgf_records(args.go_attack_limit)
    source_summaries.append({"source_id": "go_attack_site_examples", **site_go_attack_summary})
    if site_go_attack_records:
        rows, summary = replay_source_records(
            "go_attack_site_examples",
            site_go_attack_records,
            required_board_size=args.required_board_size,
            limit=args.go_attack_limit,
        )
        rows_by_source["go_attack_site_examples"] = rows
        source_summaries.append(summary)
    elif go_attack_records:
        rows, summary = replay_source_records(
            "go_attack_repo_probe",
            go_attack_records,
            required_board_size=args.required_board_size,
            limit=args.go_attack_limit,
        )
        rows_by_source["go_attack_repo_probe"] = rows
        source_summaries.append(summary)
    else:
        rows_by_source["go_attack_site_examples"] = [
            {
                "source_id": "go_attack_site_examples",
                "status": "blocked_no_public_sgf_examples_found",
                "errors": ["GitHub tree and goattack.far.ai site probes found no replayable public SGF examples."],
            }
        ]
    all_rows = [row for source_rows in rows_by_source.values() for row in source_rows]
    report = summarize(rows_by_source, source_summaries)
    write_json(output_root / "r001_sgf_replay_report.json", report)
    write_jsonl(output_root / "sgf_replay_records.jsonl", all_rows)
    write_json(output_root / "source_manifest.json", source_summaries)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_incomplete:
        failures = completion_failures(report, args)
        if failures:
            print(json.dumps({"completion_failures": failures}, ensure_ascii=False, indent=2))
            return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="experiments/results/R001_SGF_REPLAY_SMOKE")
    parser.add_argument("--jgdb-url", default=DEFAULT_JGDB_URL)
    parser.add_argument("--katago-url", default=DEFAULT_KATAGO_URL)
    parser.add_argument("--jgdb-limit", type=int, default=50)
    parser.add_argument("--katago-limit", type=int, default=50)
    parser.add_argument("--go-attack-limit", type=int, default=20)
    parser.add_argument("--jgdb-max-scan", type=int, default=300)
    parser.add_argument("--katago-max-scan", type=int, default=300)
    parser.add_argument("--source-fetch-retries", type=int, default=3)
    parser.add_argument("--required-board-size", type=int, default=19)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
