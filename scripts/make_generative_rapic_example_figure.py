#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image


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
    objects = list(plot.get("objects") or DEFAULT_OBJECTS)
    support = [float(x) for x in (plot.get("support_probs") or DEFAULT_SUPPORT)]
    selected = str(plot.get("selected_object") or values.get("risk", {}).get("selected_object") or DEFAULT_SELECTED)
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


def load_image_or_placeholder(path: str) -> np.ndarray:
    p = Path(path).expanduser()
    if p.exists():
        return np.asarray(Image.open(p).convert("RGB"))
    image = np.ones((360, 480, 3), dtype=np.float32)
    image[:, :, :] = np.array([0.95, 0.96, 0.98])
    return image


def bold_object_list(objects: Sequence[str], selected: str) -> str:
    parts = []
    selected_l = selected.lower()
    for obj in objects:
        if str(obj).lower() == selected_l:
            parts.append(rf"$\mathbf{{{obj}}}$")
        else:
            parts.append(str(obj))
    return ", ".join(parts)


def repair_card_text(before_sentence: str, after_sentence: str, selected: str) -> tuple[str, str]:
    if selected.lower() == "sink":
        return (
            'Before $c_M$:\n"... contains a [sink], a cabinet,\nand a countertop."',
            'After $c_R$:\n"... contains a cabinet and\na countertop."',
        )
    before = before_sentence.replace(selected, "[" + selected + "]")
    return (
        f'Before $c_M$:\n"... {before}."',
        f'After $c_R$:\n"... {after_sentence}."',
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Make the 3-panel generative RAPIC overview example figure.")
    ap.add_argument("--values_json", default="", help="JSON from scripts/extract_generative_overview_values.py.")
    ap.add_argument("--image_path", default="", help="Override image path. Defaults to values_json sample image path.")
    ap.add_argument("--out", default="experiments/generative_overview_panels/generative_rapic_example.pdf")
    ap.add_argument("--out_png", default="", help="Optional PNG copy.")
    args = ap.parse_args()

    values = read_values(load_json(args.values_json), args.image_path)
    objects = values["objects"]
    support_probs = np.asarray(values["support"], dtype=float)
    selected_object = str(values["selected"])
    token_labels = [str(x) for x in values["tokens"]]
    method_caption = str(values["method_caption"])
    repaired_caption = str(values["repaired_caption"])
    before_sentence, after_sentence = caption_excerpt(method_caption, repaired_caption, selected_object)
    before_text, after_text = repair_card_text(before_sentence, after_sentence, selected_object)
    image = load_image_or_placeholder(str(values["image_path"]))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.35]},
    )

    ax = axes[0]
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("(a) Method caption objects", fontsize=12, weight="bold")
    ax.text(
        0.02,
        -0.10,
        'c_M: "... contains a sink, a cabinet,\nand a countertop."',
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
    )
    ax.text(
        0.02,
        -0.28,
        f"Objects: {bold_object_list(objects, selected_object)}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#334155",
        va="top",
    )

    ax = axes[1]
    colors = ["#6BAA75" if obj != selected_object else "#D62728" for obj in objects]
    y = np.arange(len(objects))
    ax.barh(y, support_probs, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(objects, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(r"Visual support $p_{\mathrm{yes}}(o\mid I)$", fontsize=10)
    ax.set_title("(b) Object support probe", fontsize=12, weight="bold")
    for i, val in enumerate(support_probs):
        ax.text(min(float(val) + 0.025, 1.02), i, f"{float(val):.3f}", va="center", fontsize=9)
    risk_idx = objects.index(selected_object) if selected_object in objects else int(np.argmin(support_probs))
    ax.text(
        0.20,
        risk_idx,
        "top-k risk",
        va="center",
        ha="left",
        fontsize=9,
        color="#D62728",
        weight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.axis("off")
    ax.set_title("(c) Local residual repair", fontsize=12, weight="bold")
    box = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.1,
        edgecolor="#334155",
        facecolor="#F8FAFC",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        0.07,
        0.82,
        rf"Selected object: $\mathbf{{{selected_object}}}$",
        fontsize=10,
        color="#D62728",
        transform=ax.transAxes,
    )
    ax.text(
        0.07,
        0.68,
        rf"Suppress token set $T(\mathrm{{{selected_object}}})$",
        fontsize=9.5,
        color="#334155",
        transform=ax.transAxes,
    )
    ax.text(
        0.07,
        0.61,
        f"{', '.join(token_labels)}    bias b = -1.0",
        fontsize=9.5,
        color="#334155",
        transform=ax.transAxes,
    )
    ax.text(
        0.07,
        0.50,
        before_text,
        fontsize=9.2,
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.07,
        0.25,
        after_text,
        fontsize=9.2,
        transform=ax.transAxes,
        va="top",
    )

    plt.subplots_adjust(wspace=0.35, bottom=0.22)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print("[saved]", out)
    if args.out_png:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.03)
        print("[saved]", out_png)


if __name__ == "__main__":
    main()
