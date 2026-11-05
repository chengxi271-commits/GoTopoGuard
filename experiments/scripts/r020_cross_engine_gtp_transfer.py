#!/usr/bin/env python3
"""Run cross-engine top-move transfer on the R016 D3/D4/D5 subset.

The runner talks to a GTP engine that supports GoTopo's experimental
`gotopo-set_position <B|W> <361-char-board>` command. Leela Zero is adapted
through experiments/patches/leela_zero_gotopo_set_position.patch; ELF OpenGo
is adapted through experiments/patches/elf_opengo_gotopo_set_position.patch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from typing import Iterable


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_script_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R016 = load_script_module("r016_budget_generalization_queries", SCRIPT_DIR / "r016_budget_generalization_queries.py")
R011 = R016.R011_SEARCH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=pathlib.Path, required=True)
    p.add_argument("--engine-name", default="leela-zero")
    p.add_argument("--engine-command", nargs="+")
    p.add_argument("--project-root", type=pathlib.Path, default=pathlib.Path.cwd())
    p.add_argument("--d4-limit", type=int, default=1000)
    p.add_argument("--d5-limit", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--anchor-query-features", type=pathlib.Path, default=pathlib.Path("experiments/results/R016_BUDGET_GENERALIZATION_QUERIES/query_budget_features.jsonl"))
    p.add_argument("--engine-komi", type=float, help="Override board-record komi for engines that only support a fixed komi.")
    p.add_argument("--leela-visits", type=int, default=1600)
    p.add_argument("--leela-threads", type=int, default=2)
    p.add_argument("--leela-cpu-only", action="store_true")
    p.add_argument("--opencl-icd-dir", type=pathlib.Path)
    p.add_argument("--elf-root", type=pathlib.Path, default=pathlib.Path("external/engines/elf/src-v0.0.1-archive"))
    p.add_argument("--elf-model", type=pathlib.Path, default=pathlib.Path("external/engines/elf/models/pretrained-go-19x19-v2.bin"))
    p.add_argument("--elf-python", default=sys.executable)
    p.add_argument("--elf-gpu", type=int, default=0)
    p.add_argument("--elf-port", type=int, default=1234)
    p.add_argument("--elf-rollouts-per-thread", type=int, default=800)
    p.add_argument("--elf-mcts-threads", type=int, default=2)
    p.add_argument("--elf-rollouts-per-batch", type=int, default=16)
    p.add_argument("--elf-batchsize", type=int, default=16)
    p.add_argument("--restart-every", type=int, default=512)
    return p.parse_args()


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_project_path(root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else root / path


def default_leela_command(args: argparse.Namespace) -> list[str]:
    root = args.project_root
    cmd = [
        str(root / "external" / "engines" / "leela-zero" / "src-v0.0.1-archive" / "build" / "leelaz"),
        "--gtp",
        "--noponder",
        "--visits",
        str(args.leela_visits),
        "--threads",
        str(args.leela_threads),
        "--weights",
        str(root / "external" / "engines" / "leela-zero" / "networks" / "0e9ea880fd3c4444695e8ff4b8a36310d2c03f7c858cadd37af3b76df1d1d15f.gz"),
        "--quiet",
    ]
    if args.leela_cpu_only:
        cmd.append("--cpu-only")
    return cmd


def default_elf_command(args: argparse.Namespace) -> list[str]:
    root = args.project_root
    elf_root = resolve_project_path(root, args.elf_root)
    elf_model = resolve_project_path(root, args.elf_model)
    return [
        str(args.elf_python),
        str(elf_root / "scripts" / "elfgames" / "go" / "df_console.py"),
        "--mode",
        "online",
        "--keys_in_reply",
        "V",
        "rv",
        "--use_mcts",
        "--mcts_verbose_time",
        "--mcts_use_prior",
        "--load",
        str(elf_model),
        "--server_addr",
        "localhost",
        "--port",
        str(args.elf_port),
        "--replace_prefix",
        "resnet.module,resnet",
        "init_conv.module,init_conv",
        "--no_check_loaded_options",
        "--no_parameter_print",
        "--gpu",
        str(args.elf_gpu),
        "--num_block",
        "20",
        "--dim",
        "256",
        "--mcts_puct",
        "1.50",
        "--resign_thres",
        "0",
        "--mcts_virtual_loss",
        "1",
        "--batchsize",
        str(args.elf_batchsize),
        "--mcts_rollout_per_batch",
        str(args.elf_rollouts_per_batch),
        "--mcts_threads",
        str(args.elf_mcts_threads),
        "--mcts_rollout_per_thread",
        str(args.elf_rollouts_per_thread),
        "--loglevel",
        "warning",
    ]


def choose_engine_command(args: argparse.Namespace) -> list[str]:
    if args.engine_command:
        return [str(item) for item in args.engine_command]
    engine_name = args.engine_name.lower()
    if engine_name == "leela-zero":
        return default_leela_command(args)
    if engine_name in {"elf", "elf-opengo"}:
        return default_elf_command(args)
    raise ValueError(f"no default command for engine_name={args.engine_name!r}; pass --engine-command")


def default_engine_cwd(args: argparse.Namespace) -> pathlib.Path | None:
    if args.engine_command:
        return None
    if args.engine_name.lower() in {"elf", "elf-opengo"}:
        return resolve_project_path(args.project_root, args.elf_root) / "scripts" / "elfgames" / "go"
    return None


def prepend_path(env: dict[str, str], key: str, paths: Iterable[pathlib.Path | str]) -> None:
    existing = env.get(key, "")
    values = [str(path) for path in paths if pathlib.Path(path).exists()]
    if existing:
        values.append(existing)
    if values:
        env[key] = os.pathsep.join(values)


def build_engine_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.opencl_icd_dir is None:
        args.opencl_icd_dir = args.project_root / "external" / "third_party" / "opencl_icd"
    if args.opencl_icd_dir.exists():
        env["OCL_ICD_VENDORS"] = str(args.opencl_icd_dir)

    if args.engine_name.lower() in {"elf", "elf-opengo"}:
        elf_root = resolve_project_path(args.project_root, args.elf_root)
        env["game"] = "elfgames.go.game"
        env["model"] = "df_pred"
        env["model_file"] = "elfgames.go.df_model3"
        # The caller supplies ELF/CUDA shared-library paths in its own environment.
        prepend_path(
            env,
            "PYTHONPATH",
            [
                elf_root / "src_py",
                elf_root / "build" / "elf",
                elf_root / "build" / "elfgames" / "go",
                elf_root / "scripts",
                elf_root / "scripts" / "elfgames" / "go",
            ],
        )
    return env


def normalize_move(move: str | None) -> str:
    if move is None:
        return ""
    return str(move).strip().lower()


def load_anchors(path: pathlib.Path) -> dict[str, dict[str, object]]:
    anchors: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        position_id = row.get("position_id")
        visits = row.get("visits")
        if not position_id or visits not in (1024, 8192):
            continue
        item = anchors.setdefault(position_id, {})
        item[f"katago_v{visits}_top_move"] = row.get("top_move")
        item[f"katago_v{visits}_winrate"] = row.get("winrate")
        item[f"katago_v{visits}_scoreLead"] = row.get("scoreLead")
        item["initial_player"] = row.get("initial_player")
    return anchors


def board_string(board: list[str]) -> str:
    return "".join("".join(row) for row in board)


class GtpClient:
    def __init__(self, command: list[str], env: dict[str, str], stderr_path: pathlib.Path, cwd: pathlib.Path | None = None):
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_handle = stderr_path.open("a", encoding="utf-8")
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
        )

    def ask(self, command: str, timeout_s: float = 600.0) -> str:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        lines = []
        seen_response = False
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"GTP process ended while waiting for {command!r}")
            stripped = line.strip()
            if not seen_response:
                if not (stripped.startswith("=") or stripped.startswith("?")):
                    continue
                seen_response = True
            lines.append(line.rstrip("\n"))
            if seen_response and stripped == "":
                break
        else:
            raise TimeoutError(command)
        if not seen_response:
            raise TimeoutError(command)
        response = "\n".join(lines).strip()
        if response.startswith("?"):
            raise RuntimeError(f"GTP failure for {command!r}: {response}")
        if response.startswith("="):
            return response[1:].strip()
        return response

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.ask("quit", timeout_s=10.0)
        except Exception:
            if self.proc.poll() is None:
                self.proc.terminate()
        finally:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.stderr_handle.close()


def summarize(rows: list[dict]) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for key in ["ALL", "dataset", "family"]:
        pass
    def add(group_key: str, name: str, row: dict) -> None:
        group = groups.setdefault(f"{group_key}:{name}", {"group": group_key, "name": name, "n": 0, "failures": 0, "disagree_1024": 0, "disagree_8192": 0})
        group["n"] = int(group["n"]) + 1
        if row.get("schema_status") != "valid":
            group["failures"] = int(group["failures"]) + 1
        if row.get("engine_disagrees_katago_v1024"):
            group["disagree_1024"] = int(group["disagree_1024"]) + 1
        if row.get("engine_disagrees_katago_v8192"):
            group["disagree_8192"] = int(group["disagree_8192"]) + 1
    for row in rows:
        add("ALL", "ALL", row)
        add("dataset", row.get("dataset", "unknown"), row)
        add("family", row.get("family", "unknown"), row)
    out = []
    for group in groups.values():
        n = int(group["n"])
        valid = n - int(group["failures"])
        group["valid_n"] = valid
        group["failure_rate"] = int(group["failures"]) / n if n else None
        group["disagree_1024_rate"] = int(group["disagree_1024"]) / valid if valid else None
        group["disagree_8192_rate"] = int(group["disagree_8192"]) / valid if valid else None
        out.append(group)
    return {"num_rows": len(rows), "groups": sorted(out, key=lambda g: (str(g["group"]), str(g["name"])))}


def run(args: argparse.Namespace) -> int:
    output = args.output_root
    records_path = output / "cross_engine_records.jsonl"
    report_path = output / "r020_cross_engine_transfer_report.json"
    stderr_path = output / f"{args.engine_name}_stderr.log"
    selected, skipped = R016.load_selected_records(R016.DEFAULT_PATHS, args.d4_limit, args.d5_limit)
    selected = R016.shard_records(selected, args.shard_index, args.shard_count)
    if args.limit > 0:
        selected = selected[: args.limit]
    anchors = load_anchors(args.anchor_query_features)
    done = {row.get("position_id") for row in read_jsonl(records_path) if row.get("position_id")}
    command = choose_engine_command(args)
    env = build_engine_env(args)
    cwd = default_engine_cwd(args)
    manifest_env_keys = ["OCL_ICD_VENDORS", "game", "model", "model_file", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"]
    engine_visits = args.leela_visits if args.engine_name == "leela-zero" else None
    engine_rollouts = None
    if args.engine_name.lower() in {"elf", "elf-opengo"}:
        engine_rollouts = args.elf_mcts_threads * args.elf_rollouts_per_thread

    manifest = {
        "engine_name": args.engine_name,
        "command": command,
        "cwd": str(cwd) if cwd is not None else None,
        "env": {k: env[k] for k in manifest_env_keys if k in env},
        "engine_komi_override": args.engine_komi,
        "engine_visits": engine_visits,
        "engine_rollouts": engine_rollouts,
        "num_selected": len(selected),
        "num_done": len(done),
        "skipped_input": skipped,
        "anchor_query_features": str(args.anchor_query_features),
    }
    write_json(output / "run_manifest.json", manifest)

    client: GtpClient | None = None
    completed_since_restart = 0
    new_rows = []
    try:
        for record in selected:
            position_id = record["position_id"]
            if position_id in done:
                continue
            if client is None or (args.restart_every > 0 and completed_since_restart >= args.restart_every):
                if client is not None:
                    client.close()
                client = GtpClient(command, env, stderr_path, cwd)
                client.ask("boardsize 19")
                completed_since_restart = 0
            anchor = anchors.get(position_id, {})
            initial_player = str(anchor.get("initial_player") or R011.initial_player_for_record(record)[0])
            raw_record_komi = record.get("komi")
            record_komi = float(raw_record_komi) if raw_record_komi is not None else 6.5
            engine_komi = float(args.engine_komi) if args.engine_komi is not None else record_komi
            started = time.time()
            row = {
                "position_id": position_id,
                "dataset": record.get("dataset"),
                "family": record.get("family"),
                "initial_player": initial_player,
                "engine_name": args.engine_name,
                "record_komi": record_komi,
                "engine_komi": engine_komi,
                "engine_komi_forced": int(args.engine_komi is not None),
                "engine_visits": engine_visits,
                "engine_rollouts": engine_rollouts,
                "katago_v1024_top_move": anchor.get("katago_v1024_top_move"),
                "katago_v8192_top_move": anchor.get("katago_v8192_top_move"),
            }
            try:
                client.ask("clear_board", timeout_s=30.0)
                client.ask(f"komi {engine_komi}", timeout_s=30.0)
                client.ask(f"gotopo-set_position {initial_player} {board_string(record['board'])}", timeout_s=30.0)
                move = client.ask(f"genmove {initial_player}", timeout_s=900.0)
                if not normalize_move(move):
                    raise RuntimeError("empty genmove response")
                row["engine_top_move"] = move
                row["engine_disagrees_katago_v1024"] = int(normalize_move(move) != normalize_move(anchor.get("katago_v1024_top_move")))
                row["engine_disagrees_katago_v8192"] = int(normalize_move(move) != normalize_move(anchor.get("katago_v8192_top_move")))
                row["schema_status"] = "valid"
            except Exception as exc:
                row["schema_status"] = "failed"
                row["error"] = repr(exc)
                if client is not None:
                    client.close()
                    client = None
            row["elapsed_sec"] = time.time() - started
            append_jsonl(records_path, [row])
            new_rows.append(row)
            completed_since_restart += 1
            if len(new_rows) % 32 == 0:
                all_rows = read_jsonl(records_path)
                report = summarize(all_rows)
                report.update({"status": "running", **manifest})
                write_json(report_path, report)
    finally:
        if client is not None:
            client.close()
    all_rows = read_jsonl(records_path)
    report = summarize(all_rows)
    report.update({"status": "done", **manifest})
    write_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
