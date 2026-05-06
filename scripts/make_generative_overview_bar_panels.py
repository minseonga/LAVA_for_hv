#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


COLORS = {
    "text": "#111827",
    "muted": "#4B5563",
    "grid": "#E5E7EB",
    "axis": "#374151",
    "support": "#6BAA75",
    "selected": "#D12D2D",
    "caption_logit": "#52637A",
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digits_only(value: object) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def row_matches(row: dict[str, Any], *, image: str, question_id: str) -> bool:
    if question_id:
        row_qid = str(row.get("question_id") or row.get("id") or "").strip()
        if row_qid == str(question_id).strip():
            return True

    image_name = Path(str(image or "")).name
    image_digits = digits_only(image_name)
    row_image = Path(str(row.get("image") or row.get("image_path") or "")).name
    if image_name and row_image == image_name:
        return True

    row_image_id = row.get("image_id") or row.get("image") or ""
    return bool(image_digits and digits_only(row_image_id) == image_digits)


def caption_text(row: dict[str, Any]) -> str:
    for key in ("output", "caption", "text", "answer", "prediction", "pred"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def find_caption_in_file(path: Path, *, image: str, question_id: str) -> tuple[str, dict[str, Any]] | None:
    try:
        for row in read_jsonl(path):
            if row_matches(row, image=image, question_id=question_id):
                cap = caption_text(row)
                if cap:
                    return cap, row
    except Exception:
        return None
    return None


def default_raw_pred(panel_root: Path, target: str, split: str) -> Path:
    name_by_target = {
        "llava15_vga": "pred_vga_caption.jsonl",
        "qwen25_vga": "pred_vga_caption.jsonl",
        "llava15_pai_attn": "pred_pai_attn_caption.jsonl",
        "qwen25_pai_attn": "pred_pai_attn_caption.jsonl",
        "llava15_vaf": "pred_vaf_caption.jsonl",
        "qwen25_vaf": "pred_vaf_caption.jsonl",
    }
    return panel_root / "raw_sources" / target / split / name_by_target.get(target, "pred_vga_caption.jsonl")


def default_repaired_candidates(panel_root: Path, target: str, split: str, threshold: str) -> list[Path]:
    th_short = str(float(threshold)).rstrip("0").rstrip(".")
    th_fixed = f"{float(threshold):.2f}"
    apply_dir = f"{split}_apply_next_token_yesno_yp{th_short}"
    names = [
        f"pred_object_token_suppression_merged_max8_vocab_first_token_bias-1.0_yp{th_fixed}.jsonl",
        f"pred_object_token_suppression_merged_max8_vocab_first_token_bias-1.0_yp{th_short}.jsonl",
    ]
    roots = [
        panel_root / "ours_oldv84_fixedyp06" / target / apply_dir / split,
        panel_root / "ours_fixedyp06" / target / apply_dir / split,
    ]
    if target == "llava15_vga":
        roots.append(
            Path("/home/kms/LLaVA_calibration/experiments/rapic_generative_v84_valcalib_vga_token_suppression")
            / apply_dir
            / split
        )
    return [root / name for root in roots for name in names]


def search_caption(
    *,
    explicit_path: str,
    candidates: Sequence[Path],
    glob_roots: Sequence[Path],
    image: str,
    question_id: str,
    prefer_name: str,
) -> tuple[str, Path | None, dict[str, Any] | None]:
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path))
    paths.extend(candidates)
    for root in glob_roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*.jsonl")))

    seen: set[str] = set()
    ordered = sorted(
        [p for p in paths if str(p) not in seen and not seen.add(str(p))],
        key=lambda p: (prefer_name not in p.name, len(str(p))),
    )
    for path in ordered:
        if not path.exists():
            continue
        found = find_caption_in_file(path, image=image, question_id=question_id)
        if found:
            cap, row = found
            return cap, path, row
    return "", None, None


def wrap_words(value: str, *, max_chars: int = 68) -> list[str]:
    words = str(value or "").split()
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else f"{cur} {word}"
        if len(nxt) <= max_chars:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or ["caption not found"]


def multiline_text(
    x: float,
    y: float,
    lines: Sequence[str],
    *,
    size: int = 13,
    weight: int = 400,
    anchor: str = "start",
    color: str = COLORS["text"],
    line_gap: float = 1.28,
) -> str:
    out = [
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else f"{size * line_gap:.1f}"
        out.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    out.append("</text>")
    return "\n".join(out)


def caption_pair_panel(method_caption: str, repaired_caption: str, *, width: int = 860, height: int = 230) -> str:
    margin = 24
    gap = 22
    box_w = (width - 2 * margin - gap) / 2
    box_h = height - 58
    y = 38
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 22, "Caption repair example", size=15, weight=700),
    ]
    for idx, (title, body, color) in enumerate(
        [
            ("Method caption c_M", method_caption, COLORS["before"]),
            ("Repaired caption c_R", repaired_caption, COLORS["after"]),
        ]
    ):
        x = margin + idx * (box_w + gap)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="6" fill="#FFFFFF" stroke="{color}" stroke-width="1.8"/>')
        out.append(text(x + 16, y + 26, title, size=13, weight=700, anchor="start", color=color))
        out.append(multiline_text(x + 16, y + 54, wrap_words(body, max_chars=46), size=13, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def sentence_excerpt(value: str, terms: Sequence[str], *, max_chars: int = 78) -> str:
    clean = " ".join(str(value or "").split())
    if not clean:
        return '"caption not found"'
    sentences = [part.strip() for part in clean.split(".") if part.strip()]
    if not sentences:
        sentences = [clean]
    lowered_terms = [str(term).strip().lower() for term in terms if str(term).strip()]
    chosen = sentences[0]
    for sentence in sentences:
        low = sentence.lower()
        if any(term in low for term in lowered_terms):
            chosen = sentence
            break
    if len(chosen) > max_chars:
        chosen = chosen[: max_chars - 3].rstrip() + "..."
        return f'"... {chosen}"'
    return f'"... {chosen}."'


def repair_caption_panel(
    method_caption: str,
    repaired_caption: str,
    token_labels: Sequence[str],
    *,
    selected_object: str,
    suppress_bias: float = -1.0,
    width: int = 860,
    height: int = 250,
) -> str:
    margin = 26
    top = 42
    row_h = 62
    inset_w = 245
    text_w = width - 2 * margin - inset_w - 22
    selected = str(selected_object or "object").strip()
    before_excerpt = sentence_excerpt(method_caption, [selected, "contains"])
    after_terms = ["cabinet", "countertop", "contains"] if selected.lower() == "sink" else ["contains"]
    after_excerpt = sentence_excerpt(repaired_caption, after_terms)
    token_text = ", ".join(str(x) for x in token_labels[:4]) or selected

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 24, "Local object-token suppression", size=16, weight=700),
    ]
    rows = [
        ("Before c_M", before_excerpt, COLORS["before"]),
        ("After c_R", after_excerpt, COLORS["after"]),
    ]
    for idx, (label, excerpt, color) in enumerate(rows):
        y = top + idx * (row_h + 16)
        out.append(f'<rect x="{margin}" y="{y}" width="{text_w}" height="{row_h}" rx="6" fill="#FFFFFF" stroke="{color}" stroke-width="1.8"/>')
        out.append(text(margin + 16, y + 24, label, size=13, weight=700, anchor="start", color=color))
        out.append(multiline_text(margin + 16, y + 47, wrap_words(excerpt, max_chars=62), size=13, anchor="start"))

    inset_x = margin + text_w + 22
    inset_y = top
    inset_h = row_h * 2 + 16
    out.append(f'<rect x="{inset_x}" y="{inset_y}" width="{inset_w}" height="{inset_h}" rx="6" fill="#F9FAFB" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(text(inset_x + 16, inset_y + 28, f"Suppress token set T({selected})", size=13, weight=700, anchor="start"))
    out.append(text(inset_x + 16, inset_y + 60, token_text, size=16, weight=700, anchor="start", color=COLORS["selected"]))
    out.append(text(inset_x + 16, inset_y + 92, f"bias b = {suppress_bias:.1f}", size=13, anchor="start", color=COLORS["muted"]))
    out.append(text(inset_x + 16, inset_y + 118, "local decoding-time edit", size=12, anchor="start", color=COLORS["muted"]))
    out.append("</svg>")
    return "\n".join(out)


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value or "").split(",") if x.strip()]


def plot_preset(image: str, question_id: str) -> dict[str, Any]:
    image_name = Path(str(image or "")).name
    qid = str(question_id or "").strip()
    if qid == "8170" or image_name == "COCO_val2014_000000008170.jpg":
        return {
            "objects": ["refrigerator", "microwave", "sink"],
            "method_logits": [1.0, 0.8233025523097742, 0.0],
            "support_probs": [0.9852713842581693, 0.9678992896948747, 0.03567855254395119],
            "selected_object": "sink",
            "token_labels": ["sink", "s", "S"],
            "before": [1.0, 0.5102652480143863, 0.07672710924621609],
            "after": [0.9232728907537839, 0.4335381387681702, 0.0],
        }
    if image_name == "COCO_val2014_000000304819.jpg":
        return {
            "objects": ["cat", "table", "laptop", "TV"],
            "method_logits": [0.88, 0.65, 0.74, 0.91],
            "support_probs": [0.91, 0.74, 0.68, 0.22],
            "selected_object": "TV",
            "token_labels": ["TV", "tv", "television"],
            "before": [1.00, 0.82, 0.64],
            "after": [0.28, 0.22, 0.18],
        }
    return {
        "objects": ["object A", "object B", "object C", "object D"],
        "method_logits": [0.70, 0.62, 0.58, 0.90],
        "support_probs": [0.88, 0.72, 0.61, 0.18],
        "selected_object": "object D",
        "token_labels": ["obj", "Obj", "object"],
        "before": [1.00, 0.76, 0.58],
        "after": [0.24, 0.18, 0.14],
    }


def selected_index(objects: Sequence[str], selected_object: str) -> int:
    selected = str(selected_object or "").strip().lower()
    for idx, obj in enumerate(objects):
        if str(obj).strip().lower() == selected:
            return idx
    return max(0, len(objects) - 1)


def support_panel(
    objects: Sequence[str],
    probs: Sequence[float],
    *,
    selected_idx: int,
    width: int = 520,
    height: int = 240,
) -> str:
    left, right, top, bottom = 128, 58, 40, 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(objects)
    row_gap = plot_h / max(1, n)
    bar_h = min(30, row_gap * 0.55)
    axis_y = top + plot_h + 8
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 20, "Object support probe", size=15, weight=700),
    ]
    for frac in [0.0, 0.5, 1.0]:
        x = left + frac * plot_w
        out.append(f'<line x1="{x:.1f}" y1="{top - 8:.1f}" x2="{x:.1f}" y2="{axis_y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        out.append(text(x, axis_y + 17, f"{frac:.1f}", size=10, color=COLORS["muted"]))
    out.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(text(left + plot_w / 2, height - 7, "Visual support p_yes(o | I)", size=11, color=COLORS["muted"]))

    for i, (obj, prob) in enumerate(zip(objects, probs)):
        p = max(0.0, min(1.0, float(prob)))
        y = top + row_gap * i + row_gap / 2
        w = p * plot_w
        color = COLORS["selected"] if i == selected_idx else COLORS["support"]
        out.append(text(left - 12, y + 4, obj, size=12, anchor="end"))
        out.append(f'<rect x="{left:.1f}" y="{y - bar_h / 2:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
        value_x = min(left + w + 8, width - right + 8)
        out.append(text(value_x, y + 4, f"{p:.3f}", size=11, anchor="start", color=color))
    out.append("</svg>")
    return "\n".join(out)


def method_support_panel(
    objects: Sequence[str],
    method_logits: Sequence[float],
    support_probs: Sequence[float],
    *,
    selected_idx: int,
    width: int = 520,
    height: int = 270,
) -> str:
    left, right, top, bottom = 58, 24, 36, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(objects)
    group_gap = 18
    pair_gap = 5
    bar_w = max(16, (plot_w - group_gap * (n - 1) - pair_gap * n) / max(1, 2 * n))
    axis_y = top + plot_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 21, "Generation vs visual support", size=15, weight=700),
    ]
    for frac in [0.0, 0.5, 1.0]:
        y = axis_y - frac * plot_h
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        out.append(text(left - 10, y + 4, f"{frac:.1f}", size=10, anchor="end", color=COLORS["muted"]))
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(text(14, top + plot_h / 2, "normalized score", size=12, anchor="middle", color=COLORS["muted"]))

    for i, obj in enumerate(objects):
        x0 = left + i * (2 * bar_w + pair_gap + group_gap)
        vals = [
            (max(0.0, min(1.0, float(method_logits[i]))), COLORS["caption_logit"]),
            (max(0.0, min(1.0, float(support_probs[i]))), COLORS["selected"] if i == selected_idx else COLORS["support"]),
        ]
        for j, (value, color) in enumerate(vals):
            h = value * plot_h
            x = x0 + j * (bar_w + pair_gap)
            y = axis_y - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        if i == selected_idx:
            out.append(text(x0 + bar_w + pair_gap / 2, top + 12, "high logit / low support", size=10, color=COLORS["selected"]))
        out.append(text(x0 + bar_w + pair_gap / 2, axis_y + 22, obj, size=12))

    legend_y = height - 16
    out.append(f'<rect x="{left+46}" y="{legend_y-9}" width="16" height="8" rx="2" fill="{COLORS["caption_logit"]}"/>')
    out.append(text(left + 69, legend_y, "caption logit", size=11, anchor="start", color=COLORS["muted"]))
    out.append(f'<rect x="{left+165}" y="{legend_y-9}" width="16" height="8" rx="2" fill="{COLORS["support"]}"/>')
    out.append(text(left + 188, legend_y, "visual support", size=11, anchor="start", color=COLORS["muted"]))
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


def token_logit_change_panel(
    token_labels: Sequence[str],
    before: Sequence[float],
    after: Sequence[float],
    *,
    width: int = 560,
    height: int = 260,
) -> str:
    left, right, top, bottom = 86, 56, 44, 46
    plot_w = width - left - right
    row_gap = 50
    bar_h = 14
    ymax = max(1.0, *(float(x) for x in before), *(float(x) for x in after))
    ymax = max(1.0, (int(ymax / 5) + 1) * 5)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 22, "Token logits before/after suppression", size=15, weight=700),
    ]
    axis_y = top + row_gap * len(token_labels) + 4
    for tick in [0.0, 0.5, 1.0]:
        x = left + tick * plot_w
        out.append(f'<line x1="{x:.1f}" y1="{top - 14:.1f}" x2="{x:.1f}" y2="{axis_y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        out.append(text(x, axis_y + 18, f"{tick * ymax:.0f}", size=10, color=COLORS["muted"]))
    out.append(f'<line x1="{left}" y1="{axis_y:.1f}" x2="{left + plot_w:.1f}" y2="{axis_y:.1f}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    out.append(text(left + plot_w / 2, height - 8, "raw logit", size=11, color=COLORS["muted"]))

    for i, label in enumerate(token_labels):
        y = top + i * row_gap
        b = float(before[i])
        a = float(after[i])
        out.append(text(left - 12, y + 12, label, size=12, anchor="end"))
        for dy, value, color, name in [
            (-8, b, COLORS["before"], "before"),
            (10, a, COLORS["after"], "after"),
        ]:
            v = max(0.0, min(ymax, value))
            w = (v / ymax) * plot_w
            out.append(f'<rect x="{left:.1f}" y="{y + dy:.1f}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
            out.append(text(left + w + 7, y + dy + 11, f"{value:.2f}", size=10, anchor="start", color=color))
            if i == 0:
                out.append(text(left + plot_w - (96 if name == "before" else 36), top - 18 + (0 if name == "before" else 18), name, size=10, anchor="start", color=color))
    out.append("</svg>")
    return "\n".join(out)


def raw_suppression_series(values: dict[str, Any]) -> tuple[list[str], list[float], list[float]] | None:
    raw_values = values.get("raw_values", {}) if isinstance(values.get("raw_values"), dict) else {}
    rows = raw_values.get("suppression_token_logits")
    if not isinstance(rows, list) or not rows:
        return None
    labels: list[str] = []
    before: list[float] = []
    after: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("token_text") or row.get("token_id") or "").strip()
        if not label:
            continue
        try:
            b = float(row.get("before_logit"))
            a = float(row.get("after_logit"))
        except Exception:
            continue
        labels.append(label)
        before.append(b)
        after.append(a)
    return (labels, before, after) if labels else None


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print("[saved]", path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create small SVG bar panels for the generative overview figure.")
    ap.add_argument("--out_dir", default="experiments/generative_overview_panels")
    ap.add_argument(
        "--panel_root",
        default=os.environ.get(
            "PANEL_ROOT",
            "/home/kms/LLaVA_calibration/experiments/coco_chair_multibackbone_method_ours_panel",
        ),
    )
    ap.add_argument("--target", default="llava15_vga")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", default="0.60")
    ap.add_argument("--image", default="COCO_val2014_000000304819.jpg")
    ap.add_argument("--question_id", default="")
    ap.add_argument("--raw_pred_jsonl", default="")
    ap.add_argument("--repaired_pred_jsonl", default="")
    ap.add_argument("--method_caption", default="")
    ap.add_argument("--repaired_caption", default="")
    ap.add_argument("--values_json", default="", help="JSON from scripts/extract_generative_overview_values.py.")
    ap.add_argument("--objects", default="", help="Comma-separated object labels for support_probe_bars.svg.")
    ap.add_argument("--method_logits", default="", help="Comma-separated normalized caption-logit scores matching --objects.")
    ap.add_argument("--support_probs", default="", help="Comma-separated p(o|I) values matching --objects.")
    ap.add_argument("--selected_object", default="", help="Object label to highlight as lowest support.")
    ap.add_argument("--token_labels", default="", help="Comma-separated token labels for object_token_suppression_bars.svg.")
    ap.add_argument("--before", default="", help="Comma-separated before-suppression token scores.")
    ap.add_argument("--after", default="", help="Comma-separated after-suppression token scores.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    values: dict[str, Any] = {}
    if str(args.values_json or "").strip():
        with open(os.path.abspath(args.values_json), "r", encoding="utf-8") as f:
            values = json.load(f)
        plot_values = values.get("plot", {}) if isinstance(values.get("plot"), dict) else {}
        sample_values = values.get("sample", {}) if isinstance(values.get("sample"), dict) else {}
        if not args.method_caption:
            args.method_caption = str(sample_values.get("method_caption", "") or "")
        if not args.repaired_caption:
            args.repaired_caption = str(sample_values.get("repaired_caption", "") or "")
    else:
        plot_values = {}

    preset = plot_preset(str(args.image), str(args.question_id))
    objects = parse_csv_list(args.objects) or list(plot_values.get("objects") or preset["objects"])
    method_logits = parse_float_list(args.method_logits) or [float(x) for x in (plot_values.get("method_logits") or preset["method_logits"])]
    support_probs = parse_float_list(args.support_probs) or [float(x) for x in (plot_values.get("support_probs") or preset["support_probs"])]
    selected_object = str(args.selected_object or plot_values.get("selected_object") or preset["selected_object"])
    if len(support_probs) != len(objects):
        raise ValueError(f"--support_probs length ({len(support_probs)}) must match --objects length ({len(objects)})")
    if len(method_logits) != len(objects):
        method_logits = [0.0 for _ in objects]
    selected_idx = selected_index(objects, selected_object)
    print(f"[bars] support_objects={objects}")
    print(f"[bars] method_logits={method_logits}")
    print(f"[bars] support_probs={support_probs}")
    print(f"[bars] selected_object={objects[selected_idx] if objects else ''}")
    support = support_panel(objects, support_probs, selected_idx=selected_idx)
    write(out_dir / "support_probe_bars.svg", support)
    write(out_dir / "object_support_probe.svg", support)
    write(out_dir / "support_probe_only_bars.svg", support)
    mismatch = method_support_panel(objects, method_logits, support_probs, selected_idx=selected_idx)
    write(out_dir / "method_support_mismatch_bars.svg", mismatch)

    token_labels = parse_csv_list(args.token_labels) or list(plot_values.get("token_labels") or preset["token_labels"])
    before = parse_float_list(args.before) or [float(x) for x in (plot_values.get("before") or preset["before"])]
    after = parse_float_list(args.after) or [float(x) for x in (plot_values.get("after") or preset["after"])]
    raw_series = raw_suppression_series(values)
    if raw_series and not (args.token_labels or args.before or args.after):
        token_labels, before, after = raw_series
    if len(before) != len(token_labels) or len(after) != len(token_labels):
        raise ValueError("--before and --after lengths must match --token_labels length")
    print(f"[bars] token_labels={token_labels}")
    print(f"[bars] before={before}")
    print(f"[bars] after={after}")
    suppression = token_logit_change_panel(token_labels, before, after)
    write(out_dir / "object_token_logit_change.svg", suppression)
    write(out_dir / "object_token_raw_logit_change.svg", suppression)

    panel_root = Path(args.panel_root)
    method_caption = str(args.method_caption or "").strip()
    method_path: Path | None = None
    method_row: dict[str, Any] | None = None
    if not method_caption:
        method_caption, method_path, method_row = search_caption(
            explicit_path=str(args.raw_pred_jsonl or ""),
            candidates=[default_raw_pred(panel_root, str(args.target), str(args.split))],
            glob_roots=[panel_root / "raw_sources" / str(args.target)],
            image=str(args.image),
            question_id=str(args.question_id),
            prefer_name="pred_vga_caption",
        )

    repaired_caption = str(args.repaired_caption or "").strip()
    repaired_path: Path | None = None
    repaired_row: dict[str, Any] | None = None
    if not repaired_caption:
        repaired_caption, repaired_path, repaired_row = search_caption(
            explicit_path=str(args.repaired_pred_jsonl or ""),
            candidates=default_repaired_candidates(panel_root, str(args.target), str(args.split), str(args.threshold)),
            glob_roots=[
                panel_root / "ours_oldv84_fixedyp06" / str(args.target),
                panel_root / "ours_fixedyp06" / str(args.target),
            ],
            image=str(args.image),
            question_id=str(args.question_id),
            prefer_name="pred_object_token_suppression_merged",
        )

    if method_caption or repaired_caption:
        print(f"[caption] method_source={method_path or ''}")
        print(f"[caption] method={method_caption}")
        print(f"[caption] repaired_source={repaired_path or ''}")
        print(f"[caption] repaired={repaired_caption}")
        write(out_dir / "caption_pair.svg", caption_pair_panel(method_caption, repaired_caption))
        write(
            out_dir / "object_token_suppression_bars.svg",
            repair_caption_panel(
                method_caption,
                repaired_caption,
                token_labels,
                selected_object=objects[selected_idx] if objects else selected_object,
            ),
        )
        write(
            out_dir / "local_suppression_repair.svg",
            repair_caption_panel(
                method_caption,
                repaired_caption,
                token_labels,
                selected_object=objects[selected_idx] if objects else selected_object,
            ),
        )
        write(
            out_dir / "caption_pair.json",
            json.dumps(
                {
                    "target": args.target,
                    "split": args.split,
                    "threshold": args.threshold,
                    "image": args.image,
                    "question_id": args.question_id,
                    "method_caption": method_caption,
                    "method_source": str(method_path) if method_path else "",
                    "method_row": method_row or {},
                    "repaired_caption": repaired_caption,
                    "repaired_source": str(repaired_path) if repaired_path else "",
                    "repaired_row": repaired_row or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        write(
            out_dir / "caption_pair.txt",
            "\n".join(
                [
                    f"image: {args.image}",
                    f"target: {args.target}",
                    "",
                    f"method_caption_source: {method_path or ''}",
                    f"method_caption: {method_caption}",
                    "",
                    f"repaired_caption_source: {repaired_path or ''}",
                    f"repaired_caption: {repaired_caption}",
                    "",
                ]
            ),
        )
    else:
        print(f"[warn] caption row not found for image={args.image!r} question_id={args.question_id!r}")


if __name__ == "__main__":
    main()
