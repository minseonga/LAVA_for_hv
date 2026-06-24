#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def maybe_int(value: Any) -> int:
    value_f = maybe_float(value)
    return int(round(value_f)) if value_f is not None else 0


def safe_float(value: Any, default: float = 0.0) -> float:
    value_f = maybe_float(value)
    return float(default if value_f is None else value_f)


def canonical_object(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def split_objects(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [canonical_object(v) for v in value if canonical_object(v)]
    text = str(value or "").strip()
    if not text:
        return []
    return [canonical_object(x) for x in re.split(r"\s*\|\s*|\s*,\s*", text) if canonical_object(x)]


def object_match(a: str, b: str, *, substring: bool) -> bool:
    a = canonical_object(a)
    b = canonical_object(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith("s") and a[:-1] == b:
        return True
    if b.endswith("s") and b[:-1] == a:
        return True
    return bool(substring and (a in b or b in a))


def logsumexp2(a: float, b: float) -> float:
    m = max(float(a), float(b))
    return float(m + math.log(math.exp(float(a) - m) + math.exp(float(b) - m)))


def binary_entropy(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) != len(labels) or not scores:
        return None
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, labels) if int(y) == 1)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    pos = min(1.0, max(0.0, float(q))) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(sum(values) / len(values))


def parse_details(row: Dict[str, str]) -> List[Dict[str, Any]]:
    raw = str(row.get("risk_details_json", "") or "").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def add_metric_fields(out: Dict[str, Any], *, yes_prob: float, lp_margin: float, yes_lp: float, no_lp: float) -> None:
    yn_mass_lp = logsumexp2(yes_lp, no_lp)
    out.update(
        {
            "yes_prob": float(yes_prob),
            "lp_margin": float(lp_margin),
            "yes_lp": float(yes_lp),
            "no_lp": float(no_lp),
            "yn_mass_lp": float(yn_mass_lp),
            "binary_entropy": binary_entropy(yes_prob),
            "risk_1_minus_yes_prob": float(1.0 - yes_prob),
            "risk_neg_lp_margin": float(-lp_margin),
            "risk_neg_yn_mass_lp": float(-yn_mass_lp),
            "risk_binary_entropy": binary_entropy(yes_prob),
        }
    )


def infer_task_kind(rows: Sequence[Dict[str, str]]) -> str:
    if rows and "oracle_hallucinated_objects" in rows[0]:
        return "captioning_chair_object_support"
    if rows and any(k in rows[0] for k in ("answer", "label", "gt", "category")):
        return "pope_or_discriminative_yesno_like"
    return "unknown"


def build_sample_top_rows(rows: Sequence[Dict[str, str]], *, label_col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        label = maybe_int(row.get(label_col))
        yes_prob = safe_float(row.get("risk_top_yes_prob"), 1.0)
        lp_margin = safe_float(row.get("risk_top_lp_margin"), 0.0)
        yes_lp = safe_float(row.get("risk_top_yes_lp"), 0.0)
        no_lp = safe_float(row.get("risk_top_no_lp"), 0.0)
        item: Dict[str, Any] = {
            "level": "sample_top",
            "id": str(row.get("id") or row.get("image_id") or ""),
            "image": str(row.get("image") or ""),
            "object": str(row.get("risk_top_object") or ""),
            "label_hallucinated": int(label),
            "oracle_hallucinated_objects": str(row.get("oracle_hallucinated_objects") or ""),
        }
        add_metric_fields(item, yes_prob=yes_prob, lp_margin=lp_margin, yes_lp=yes_lp, no_lp=no_lp)
        out.append(item)
    return out


def build_object_level_rows(rows: Sequence[Dict[str, str]], *, substring_match: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        gold = split_objects(row.get("oracle_hallucinated_objects"))
        details = parse_details(row)
        for item in details:
            obj_name = canonical_object(item.get("object"))
            label = int(any(object_match(obj_name, g, substring=substring_match) for g in gold))
            yes_prob = safe_float(item.get("yesno_prob"), 1.0)
            lp_margin = safe_float(item.get("yesno_lp_margin"), 0.0)
            yes_lp = safe_float(item.get("yesno_yes_lp"), 0.0)
            no_lp = safe_float(item.get("yesno_no_lp"), 0.0)
            out_item: Dict[str, Any] = {
                "level": "object",
                "id": str(row.get("id") or row.get("image_id") or ""),
                "image": str(row.get("image") or ""),
                "object": obj_name,
                "label_hallucinated": int(label),
                "oracle_hallucinated_objects": " | ".join(gold),
            }
            add_metric_fields(out_item, yes_prob=yes_prob, lp_margin=lp_margin, yes_lp=yes_lp, no_lp=no_lp)
            out.append(out_item)
    return out


def group_summary(name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ys = [int(r["label_hallucinated"]) for r in rows]
    pos = [r for r in rows if int(r["label_hallucinated"]) == 1]
    neg = [r for r in rows if int(r["label_hallucinated"]) == 0]

    def metric_block(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        vals_prob = [float(r["yes_prob"]) for r in rs]
        vals_margin = [float(r["lp_margin"]) for r in rs]
        low = [r for r in rs if float(r["yes_prob"]) <= 0.1]
        sharp = [r for r in rs if float(r["yes_prob"]) >= 0.9 and float(r["lp_margin"]) >= 2.0]
        return {
            "n": int(len(rs)),
            "yes_prob_mean": mean(vals_prob),
            "yes_prob_p50": quantile(vals_prob, 0.5),
            "yes_prob_p90": quantile(vals_prob, 0.9),
            "lp_margin_mean": mean(vals_margin),
            "lp_margin_p50": quantile(vals_margin, 0.5),
            "lp_margin_p90": quantile(vals_margin, 0.9),
            "low_support_yes_prob_le_0p1_count": int(len(low)),
            "low_support_yes_prob_le_0p1_rate": float(len(low) / max(1, len(rs))),
            "sharp_yes_count": int(len(sharp)),
            "sharp_yes_rate": float(len(sharp) / max(1, len(rs))),
        }

    features = [
        "risk_1_minus_yes_prob",
        "risk_neg_lp_margin",
        "risk_neg_yn_mass_lp",
        "risk_binary_entropy",
    ]
    auc_rows = []
    for feature in features:
        xs = [float(r[feature]) for r in rows]
        auc_rows.append({"level": name, "feature": feature, "auroc_hallucinated": auroc(xs, ys)})

    corr_rows = []
    raw_features = ["yes_prob", "lp_margin", "yn_mass_lp", "binary_entropy"]
    for i, a in enumerate(raw_features):
        for b in raw_features[i + 1 :]:
            xs = [float(r[a]) for r in rows]
            zs = [float(r[b]) for r in rows]
            corr_rows.append(
                {
                    "level": name,
                    "feature_a": a,
                    "feature_b": b,
                    "pearson": pearson(xs, zs),
                    "spearman": spearman(xs, zs),
                }
            )

    sharp_pos = [r for r in pos if float(r["yes_prob"]) >= 0.9 and float(r["lp_margin"]) >= 2.0]
    top_objects = Counter(str(r.get("object", "")) for r in sharp_pos if str(r.get("object", ""))).most_common(30)
    examples = sorted(sharp_pos, key=lambda r: (-float(r["yes_prob"]), -float(r["lp_margin"]), str(r["id"])))[:30]
    return {
        "level": name,
        "counts": {
            "n": int(len(rows)),
            "n_hallucinated": int(sum(ys)),
            "n_non_hallucinated": int(len(ys) - sum(ys)),
        },
        "hallucinated": metric_block(pos),
        "non_hallucinated": metric_block(neg),
        "auroc": auc_rows,
        "correlations": corr_rows,
        "sharp_hallucinated_top_objects": [{"object": obj, "count": count} for obj, count in top_objects],
        "sharp_hallucinated_examples": examples,
    }


def flatten_prefixed(prefix: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append({"section": prefix, **row})
    return out


def markdown_report(summary: Dict[str, Any]) -> str:
    lines = [
        f"# Generative Support/Energy-Proxy Audit",
        "",
        f"- task_kind: `{summary['task_kind']}`",
        f"- risk_csv: `{summary['risk_csv']}`",
        "",
        "Note: `yn_mass_lp = logsumexp(yes_logprob, no_logprob)` is a yes/no answer-mass proxy, not true raw-logit energy. True energy requires full-vocabulary raw logits.",
        "",
    ]
    for level in summary["levels"]:
        lines += [
            f"## {level['level']}",
            "",
            "| Group | n | mean p_yes | p50 p_yes | p90 p_yes | mean margin | p50 margin | p90 margin | low-support % | sharp-yes % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for key, label in [("hallucinated", "hallucinated"), ("non_hallucinated", "non-hallucinated")]:
            block = level[key]
            lines.append(
                f"| {label} | {block['n']} | "
                f"{fmt(block['yes_prob_mean'])} | {fmt(block['yes_prob_p50'])} | {fmt(block['yes_prob_p90'])} | "
                f"{fmt(block['lp_margin_mean'])} | {fmt(block['lp_margin_p50'])} | {fmt(block['lp_margin_p90'])} | "
                f"{100.0 * block['low_support_yes_prob_le_0p1_rate']:.2f} | "
                f"{100.0 * block['sharp_yes_rate']:.2f} |"
            )
        lines += ["", "### AUROC", ""]
        lines += ["| Feature | AUROC |", "| --- | ---: |"]
        for row in level["auroc"]:
            lines.append(f"| {row['feature']} | {fmt(row['auroc_hallucinated'])} |")
        lines += ["", "### Correlation", ""]
        lines += ["| A | B | Pearson | Spearman |", "| --- | --- | ---: | ---: |"]
        for row in level["correlations"]:
            lines.append(f"| {row['feature_a']} | {row['feature_b']} | {fmt(row['pearson'])} | {fmt(row['spearman'])} |")
        lines += ["", "### Sharp Hallucinated Top Objects", ""]
        top = ", ".join(f"{r['object']} ({r['count']})" for r in level["sharp_hallucinated_top_objects"][:15])
        lines += [top or "_none_", ""]
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CPU-only audit of VLM object-support distributions for hallucination vs non-hallucination."
    )
    ap.add_argument("--risk_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_col", default="oracle_any_hallucinated_object")
    ap.add_argument("--substring_match", action="store_true")
    args = ap.parse_args()

    risk_csv = Path(args.risk_csv).expanduser().resolve()
    rows = read_csv_rows(risk_csv)
    sample_top = build_sample_top_rows(rows, label_col=str(args.label_col))
    object_level = build_object_level_rows(rows, substring_match=bool(args.substring_match))

    levels = []
    if sample_top:
        levels.append(group_summary("sample_top", sample_top))
    if object_level:
        levels.append(group_summary("object_level", object_level))

    summary = {
        "task_kind": infer_task_kind(rows),
        "risk_csv": str(risk_csv),
        "notes": {
            "low_support": "yes_prob <= 0.1; yes_prob is sigmoid(yes_logprob - no_logprob), a binary yes-vs-no support score.",
            "sharp_yes": "yes_prob >= 0.9 and lp_margin >= 2.0.",
            "yn_mass_lp": "logsumexp(yes_logprob, no_logprob); this is a yes/no log-probability mass proxy, not true raw-logit energy.",
            "energy_limitation": "True energy needs full-vocabulary raw logits or at least raw selected logits plus the normalizer; existing CSV stores log-probability summaries.",
        },
        "levels": levels,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "support_energy_proxy_sample_top_rows.csv", sample_top)
    if object_level:
        write_csv(out_dir / "support_energy_proxy_object_level_rows.csv", object_level)
    write_json(out_dir / "support_energy_proxy_audit.json", summary)
    (out_dir / "support_energy_proxy_audit.md").write_text(markdown_report(summary) + "\n", encoding="utf-8")
    for level in levels:
        write_csv(out_dir / f"{level['level']}_auroc.csv", level["auroc"])
        write_csv(out_dir / f"{level['level']}_correlations.csv", level["correlations"])
        write_csv(out_dir / f"{level['level']}_sharp_hallucinated_examples.csv", level["sharp_hallucinated_examples"])
    print("[saved]", out_dir / "support_energy_proxy_audit.json")
    print("[saved]", out_dir / "support_energy_proxy_audit.md")


if __name__ == "__main__":
    main()
