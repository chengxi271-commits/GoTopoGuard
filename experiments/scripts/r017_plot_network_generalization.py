#!/usr/bin/env python
"""Render R017 network-generalization summary as a deterministic SVG."""

from __future__ import annotations

import argparse
import html
import json
import pathlib


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def bar(x: float, y: float, width: float, height: float, value: float, color: str) -> str:
    actual = max(0.0, min(1.0, value)) * width
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{actual:.1f}" height="{height:.1f}" '
        f'rx="3" fill="{color}"/>'
    )


def label(x: float, y: float, text: str, size: int = 13, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#1f2933">'
        f"{html.escape(text)}</text>"
    )


def render_svg(rows: list[dict]) -> str:
    network = [row for row in rows if row["group"] == "network"]
    dataset = [row for row in rows if row["group"] == "network_dataset"]
    family = sorted(
        [row for row in rows if row["group"] == "network_family"],
        key=lambda row: row.get("top_move_disagreement_rate_vs_b28_1024") or 0.0,
        reverse=True,
    )[:8]

    width = 1040
    height = 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1040" height="760" fill="#f7f9fb"/>',
        label(36, 50, "R017 Network Generalization: top-move disagreement vs b28@1024", 22, "700"),
        label(36, 78, "Comparison anchor only; b28@1024 is not treated as Go ground truth.", 13),
    ]

    colors = {"b18": "#277da1", "b40": "#f3722c"}
    x0 = 36
    y = 126
    bar_w = 280
    parts.append(label(x0, y - 24, "Overall network disagreement", 16, "700"))
    for row in network:
        name = row["name"]
        value = row["top_move_disagreement_rate_vs_b28_1024"]
        parts.append(label(x0, y + 15, name, 13, "700"))
        parts.append(bar(x0 + 58, y, bar_w, 22, value, colors.get(name, "#4b5563")))
        parts.append(label(x0 + 58 + bar_w + 12, y + 16, fmt_pct(value), 13, "700"))
        parts.append(label(x0 + 58, y + 42, f"mean score delta {row['mean_score_abs_delta_vs_b28_1024']:.3f}", 12))
        y += 68

    x1 = 520
    y = 126
    parts.append(label(x1, y - 24, "Dataset-level disagreement", 16, "700"))
    for row in dataset:
        name = row["name"]
        net = name.split("/", 1)[0]
        value = row["top_move_disagreement_rate_vs_b28_1024"]
        parts.append(label(x1, y + 15, name, 13, "700"))
        parts.append(bar(x1 + 96, y, 300, 20, value, colors.get(net, "#4b5563")))
        parts.append(label(x1 + 408, y + 15, fmt_pct(value), 13, "700"))
        y += 42

    y = 365
    parts.append(label(36, y, "Highest-disagreement families", 16, "700"))
    y += 30
    max_family = max((row["top_move_disagreement_rate_vs_b28_1024"] for row in family), default=1.0)
    for row in family:
        value = row["top_move_disagreement_rate_vs_b28_1024"]
        normalized = value / max_family if max_family else 0.0
        name = row["name"]
        net = name.split("/", 1)[0]
        parts.append(label(36, y + 15, name, 12, "700"))
        parts.append(bar(370, y, 390, 20, normalized, colors.get(net, "#4b5563")))
        parts.append(label(780, y + 15, fmt_pct(value), 12, "700"))
        parts.append(label(860, y + 15, f"n={row['position_count']}", 12))
        y += 36

    parts.extend(
        [
            '<line x1="36" y1="700" x2="1004" y2="700" stroke="#c9d2dc" stroke-width="1"/>',
            label(
                36,
                728,
                "R017 subset: D3 136 + D4 1000 + D5 1008 positions; b18/b40 official KataGo networks, 1024 visits.",
                12,
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(pathlib.Path(args.summary))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(rows), encoding="utf-8")
    print(output)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="experiments/results/R017_NETWORK_GENERALIZATION_ANALYSIS/network_stability_summary.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/plots/R017_network_generalization/r017_network_generalization.svg",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
