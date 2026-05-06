#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_IMAGE = "/home/kms/data/pope/val2014/COCO_val2014_000000008170.jpg"
DEFAULT_OBJECTS = ["refrigerator", "microwave", "sink"]
DEFAULT_SUPPORT = [0.9852713842581693, 0.9678992896948747, 0.03567855254395119]
DEFAULT_SELECTED = "sink"
DEFAULT_TOKENS = ["sink", "s", "S"]
DEFAULT_METHOD_CAPTION = (
    "The image features a small kitchen with a white refrigerator and a microwave oven. "
    "The refrigerator is covered in magnets and papers, giving it a cluttered appearance. "
    "The microwave is placed on top of the refrigerator, and the refrigerator is located next to a wall. "
    "The kitchen also contains a sink, a cabinet, and a countertop. "
    "The refrigerator is situated in the corner of the kitchen, and the microwave is placed on top of it."
)
DEFAULT_REPAIRED_CAPTION = (
    "The image features a small kitchen with a white refrigerator and a microwave oven. "
    "The refrigerator is covered in magnets and papers, giving it a cluttered appearance. "
    "The microwave is placed on top of the refrigerator, and the refrigerator is located next to a wall. "
    "The kitchen also contains a cabinet and a countertop. "
    "The refrigerator is situated in the corner of the room, and the microwave is placed on top of it."
)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 14,
    weight: str = "400",
    anchor: str = "middle",
    color: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def multiline_text(
    x: float,
    y: float,
    lines: Sequence[str],
    *,
    size: int = 14,
    weight: str = "400",
    color: str = "#111827",
    line_gap: float = 1.25,
) -> str:
    out = [
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="start" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else f"{size * line_gap:.1f}"
        out.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    out.append("</text>")
    return "\n".join(out)


def load_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def clean_sentence_text(value: str) -> str:
    return " ".join(str(value or "").split())


def sentence_with(value: str, terms: Sequence[str], fallback: str) -> str:
    clean = clean_sentence_text(value)
    if not clean:
        return fallback
    sentences = [part.strip() for part in clean.split(".") if part.strip()]
    lowered_terms = [str(term).lower() for term in terms if str(term).strip()]
    for sentence in sentences:
        low = sentence.lower()
        if any(term in low for term in lowered_terms):
            return sentence
    return sentences[0] if sentences else fallback


def caption_excerpt(method_caption: str, repaired_caption: str, selected: str) -> tuple[str, str]:
    before = sentence_with(
        method_caption,
        [selected, "contains"],
        "The kitchen also contains a sink, a cabinet, and a countertop",
    )
    after_terms = ["cabinet", "countertop", "contains"] if selected.lower() == "sink" else ["contains"]
    after = sentence_with(
        repaired_caption,
        after_terms,
        "The kitchen also contains a cabinet and a countertop",
    )
    return before, after


def read_values(values: dict[str, Any], image_path_arg: str) -> dict[str, Any]:
    plot = values.get("plot") if isinstance(values.get("plot"), dict) else {}
    sample = values.get("sample") if isinstance(values.get("sample"), dict) else {}
    risk = values.get("risk") if isinstance(values.get("risk"), dict) else {}
    objects = list(plot.get("objects") or DEFAULT_OBJECTS)
    support = [float(x) for x in (plot.get("support_probs") or DEFAULT_SUPPORT)]
    selected = str(plot.get("selected_object") or risk.get("selected_object") or DEFAULT_SELECTED)
    tokens = list(plot.get("token_labels") or DEFAULT_TOKENS)
    image_path = str(image_path_arg or sample.get("image_abs_path") or DEFAULT_IMAGE)
    method_caption = str(sample.get("method_caption") or DEFAULT_METHOD_CAPTION)
    repaired_caption = str(sample.get("repaired_caption") or DEFAULT_REPAIRED_CAPTION)
    if len(objects) != len(support):
        raise ValueError(f"objects length ({len(objects)}) and support length ({len(support)}) differ")
    return {
        "objects": objects,
        "support": support,
        "selected": selected,
        "tokens": tokens,
        "image_path": image_path,
        "method_caption": method_caption,
        "repaired_caption": repaired_caption,
    }


def image_href(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def bold_object_list(objects: Sequence[str], selected: str) -> str:
    return ", ".join(f"[{obj}]" if str(obj).lower() == selected.lower() else str(obj) for obj in objects)


def repair_card_text(before_sentence: str, after_sentence: str, selected: str) -> tuple[list[str], list[str]]:
    if selected.lower() == "sink":
        return (
            ['Before c_M:', '"... contains a [sink], a cabinet,', 'and a countertop."'],
            ['After c_R:', '"... contains a cabinet and', 'a countertop."'],
        )
    before = before_sentence.replace(selected, "[" + selected + "]")
    return (
        ["Before c_M:", f'"... {before}."'],
        ["After c_R:", f'"... {after_sentence}."'],
    )


def bar_panel(x0: float, y0: float, objects: Sequence[str], support: Sequence[float], selected: str) -> list[str]:
    left = x0 + 125
    top = y0 + 68
    plot_w = 300
    row_gap = 64
    selected_l = selected.lower()
    out = [
        text(x0 + 230, y0 + 24, "(b) Object support probe", size=17, weight="700"),
        text(x0 + 255, y0 + 330, "Visual support p_yes(o | I)", size=14),
    ]
    for tick in [0.0, 0.5, 1.0]:
        tx = left + tick * plot_w
        out.append(f'<line x1="{tx:.1f}" y1="{top - 20:.1f}" x2="{tx:.1f}" y2="{top + row_gap * (len(objects) - 1) + 32:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        out.append(text(tx, top + row_gap * (len(objects) - 1) + 56, f"{tick:.1f}", size=11, color="#4B5563"))
    for idx, (obj, prob) in enumerate(zip(objects, support)):
        y = top + idx * row_gap
        val = max(0.0, min(1.0, float(prob)))
        color = "#D62728" if str(obj).lower() == selected_l else "#6BAA75"
        out.append(text(left - 12, y + 5, obj, size=14, anchor="end"))
        out.append(f'<rect x="{left:.1f}" y="{y - 16:.1f}" width="{val * plot_w:.1f}" height="32" rx="4" fill="{color}"/>')
        out.append(text(min(left + val * plot_w + 32, left + plot_w + 20), y + 5, f"{val:.3f}", size=13, anchor="start", color=color))
        if str(obj).lower() == selected_l:
            out.append(text(left + 70, y + 5, "top-k risk", size=13, weight="700", anchor="start", color="#D62728"))
    out.append(f'<line x1="{left:.1f}" y1="{top + row_gap * (len(objects) - 1) + 32:.1f}" x2="{left + plot_w:.1f}" y2="{top + row_gap * (len(objects) - 1) + 32:.1f}" stroke="#374151" stroke-width="1.2"/>')
    return out


def figure_svg(values: dict[str, Any]) -> str:
    objects = [str(x) for x in values["objects"]]
    support = [float(x) for x in values["support"]]
    selected = str(values["selected"])
    tokens = [str(x) for x in values["tokens"]]
    before_sentence, after_sentence = caption_excerpt(str(values["method_caption"]), str(values["repaired_caption"]), selected)
    before_lines, after_lines = repair_card_text(before_sentence, after_sentence, selected)
    href = image_href(str(values["image_path"]))
    width, height = 1500, 420
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    x0, y0 = 30, 26
    out.append(text(x0 + 225, y0 + 18, "(a) Method caption objects", size=17, weight="700"))
    out.append(f'<rect x="{x0}" y="{y0 + 38}" width="440" height="258" rx="5" fill="#F1F5F9" stroke="#CBD5E1"/>')
    if href:
        out.append(f'<image x="{x0}" y="{y0 + 38}" width="440" height="258" href="{href}" preserveAspectRatio="xMidYMid meet"/>')
    else:
        out.append(text(x0 + 220, y0 + 168, "image not found", size=15, color="#64748B"))
    out.append(multiline_text(x0 + 8, y0 + 322, ['c_M: "... contains a sink, a cabinet,', 'and a countertop."'], size=13))
    out.append(text(x0 + 8, y0 + 385, f"Objects: {bold_object_list(objects, selected)}", size=13, anchor="start", color="#334155"))

    out.extend(bar_panel(520, 26, objects, support, selected))

    x2, y2 = 930, 26
    out.append(text(x2 + 270, y2 + 18, "(c) Local residual repair", size=17, weight="700"))
    out.append(f'<rect x="{x2 + 16}" y="{y2 + 52}" width="520" height="310" rx="14" fill="#F8FAFC" stroke="#334155" stroke-width="1.2"/>')
    out.append(text(x2 + 48, y2 + 92, f"Selected object: {selected}", size=15, weight="700", anchor="start", color="#D62728"))
    out.append(text(x2 + 48, y2 + 136, f"Suppress token set T({selected})", size=14, anchor="start", color="#334155"))
    out.append(text(x2 + 48, y2 + 165, f"{', '.join(tokens)}    bias b = -1.0", size=14, anchor="start", color="#334155"))
    out.append(multiline_text(x2 + 48, y2 + 215, before_lines, size=14))
    out.append(multiline_text(x2 + 48, y2 + 305, after_lines, size=14))
    out.append("</svg>")
    return "\n".join(out)


def svg_output_path(path: str) -> Path:
    out = Path(path)
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Make the 3-panel generative RAPIC overview example figure as dependency-free SVG.")
    ap.add_argument("--values_json", default="", help="JSON from scripts/extract_generative_overview_values.py.")
    ap.add_argument("--image_path", default="", help="Override image path. Defaults to values_json sample image path.")
    ap.add_argument("--out", default="experiments/generative_overview_panels/generative_rapic_example.svg")
    ap.add_argument("--out_png", default="", help="Ignored by the dependency-free SVG renderer.")
    args = ap.parse_args()

    values = read_values(load_json(args.values_json), args.image_path)
    out = svg_output_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(figure_svg(values), encoding="utf-8")
    print("[saved]", out)
    if args.out_png:
        print("[warn] --out_png ignored; this dependency-free renderer writes SVG. Convert SVG externally if PNG/PDF is required.")


if __name__ == "__main__":
    main()
