#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Iterable, Sequence


COLORS = {
    "text": "#111827",
    "muted": "#4B5563",
    "grid": "#E5E7EB",
    "axis": "#374151",
    "support": "#6BAA75",
    "selected": "#D12D2D",
    "before": "#8B95A7",
    "after": "#D12D2D",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, *, size: int = 13, weight: int = 400, anchor: str = "middle", color: str = COLORS["text"]) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def support_panel(
    objects: Sequence[str],
    probs: Sequence[float],
    *,
    selected_idx: int,
    width: int = 430,
    height: int = 250,
) -> str:
    left, right, top, bottom = 58, 24, 32, 46
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(objects)
    gap = 14
    bar_w = max(18, (plot_w - gap * (n - 1)) / max(1, n))
    axis_y = top + plot_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 20, "Support probe", size=15, weight=700),
    ]
    for frac in [0.0, 0.5, 1.0]:
        y = axis_y - frac * plot_h
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        out.append(text(left - 10, y + 4, f"{frac:.1f}", size=10, anchor="end", color=COLORS["muted"]))
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(text(14, top + plot_h / 2, "p(o | I)", size=12, anchor="middle", color=COLORS["muted"]))

    for i, (obj, prob) in enumerate(zip(objects, probs)):
        p = max(0.0, min(1.0, float(prob)))
        x = left + i * (bar_w + gap)
        h = p * plot_h
        y = axis_y - h
        color = COLORS["selected"] if i == selected_idx else COLORS["support"]
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        out.append(text(x + bar_w / 2, max(top + 12, y - 7), f"{p:.2f}", size=11, color=color))
        out.append(text(x + bar_w / 2, axis_y + 22, obj, size=12))
    out.append(text(width / 2, height - 8, "select lowest support", size=11, color=COLORS["muted"]))
    out.append("</svg>")
    return "\n".join(out)


def suppression_panel(
    token_labels: Sequence[str],
    before: Sequence[float],
    after: Sequence[float],
    *,
    width: int = 430,
    height: int = 250,
) -> str:
    left, right, top, bottom = 60, 24, 34, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(token_labels)
    group_gap = 22
    pair_gap = 5
    bar_w = max(15, (plot_w - group_gap * (n - 1) - pair_gap * n) / max(1, 2 * n))
    axis_y = top + plot_h
    ymax = max(1.0, *(float(x) for x in before), *(float(x) for x in after))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 20, "Object-token pathway", size=15, weight=700),
    ]
    for frac in [0.0, 0.5, 1.0]:
        y = axis_y - frac * plot_h
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        out.append(text(left - 10, y + 4, f"{frac*ymax:.1f}", size=10, anchor="end", color=COLORS["muted"]))
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(text(14, top + plot_h / 2, "relative logit", size=12, anchor="middle", color=COLORS["muted"]))

    for i, label in enumerate(token_labels):
        x0 = left + i * (2 * bar_w + pair_gap + group_gap)
        for j, (value, color) in enumerate([(before[i], COLORS["before"]), (after[i], COLORS["after"])]):
            v = max(0.0, min(ymax, float(value)))
            h = (v / ymax) * plot_h
            x = x0 + j * (bar_w + pair_gap)
            y = axis_y - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        out.append(text(x0 + bar_w + pair_gap / 2, axis_y + 22, label, size=12))
    legend_y = height - 13
    out.append(f'<rect x="{left+70}" y="{legend_y-9}" width="16" height="8" rx="2" fill="{COLORS["before"]}"/>')
    out.append(text(left + 93, legend_y, "before", size=11, anchor="start", color=COLORS["muted"]))
    out.append(f'<rect x="{left+152}" y="{legend_y-9}" width="16" height="8" rx="2" fill="{COLORS["after"]}"/>')
    out.append(text(left + 175, legend_y, "after", size=11, anchor="start", color=COLORS["muted"]))
    out.append("</svg>")
    return "\n".join(out)


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print("[saved]", path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create small SVG bar panels for the generative overview figure.")
    ap.add_argument("--out_dir", default="experiments/generative_overview_panels")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    objects = ["cat", "table", "laptop", "TV"]
    support_probs = [0.91, 0.74, 0.68, 0.22]
    selected_idx = 3
    write(out_dir / "support_probe_bars.svg", support_panel(objects, support_probs, selected_idx=selected_idx))

    token_labels = ["TV", "tv", "television"]
    before = [1.00, 0.82, 0.64]
    after = [0.28, 0.22, 0.18]
    write(out_dir / "object_token_suppression_bars.svg", suppression_panel(token_labels, before, after))


if __name__ == "__main__":
    main()
