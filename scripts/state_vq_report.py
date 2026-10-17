"""SM32 summary: student state-quantization variants vs the fork (scripts/small_model_heldout.py rows).

Review-weighted benchmark LogLoss (curve = ahead head, instant) on the users every listed label has, Δ% vs the fork
w8+s8 reference (95% user-bootstrap CI, scripts/small_model_report.py), Δ vs the student's w8+s8, bits per card / note
entity and the implied card+note state of the largest dataset user (246,801 cards, 76,887 notes).

usage: state_vq_report.py --jsonl a.jsonl --jsonl b.jsonl --labels st/fp32,st/w8+s8,... [--out DIR/summary]
"""
import argparse
import json
from pathlib import Path

from scripts import compress_quant as cq
from scripts.small_model_report import arrays, boot_delta_pct, rw

ROOT = Path(__file__).resolve().parents[1]
CARDS, NOTES = 246_801, 76_887
# Student (w64: card 2 WKV layers + 3 shift vectors, note 1 + 2; 4 heads x 16) state bits per entity without VQ.
FLOAT_BITS = {"card": 2 * 4 * 256 * 32 + 3 * 64 * 32, "note": 4 * 256 * 32 + 2 * 64 * 32}
# +s8: int8 codes + one f32 scale per 16-element WKV row; shift vectors stay f32 on the student path.
S8_BITS = {"card": 2 * 64 * (16 * 8 + 32) + 3 * 64 * 32, "note": 64 * (16 * 8 + 32) + 2 * 64 * 32}


def load(paths):
    per, meta = {}, {}
    for path in paths:
        for row in map(json.loads, Path(path).read_text().splitlines()):
            if "benchmark" not in row or "ahead" not in row["benchmark"]:
                continue
            per.setdefault(row["label"], {})[row["user_id"]] = row["benchmark"]
            meta[row["label"]] = row
    return per, meta


def bits_of(row):
    vq = row.get("state_vq")
    if vq:
        return vq["card"]["bits"], vq["note"]["bits"]
    if row.get("kind") == "fork":  # fork runtime state (scripts/compress_quant.state_bytes; card 3, note 2 layers)
        precision = "int8" if row.get("state_bits") == 8 else "f32"
        return 8 * cq.state_bytes(3, precision=precision), 8 * cq.state_bytes(2, precision=precision)
    if row.get("state_bits") == 8:
        return S8_BITS["card"], S8_BITS["note"]
    return FLOAT_BITS["card"], FLOAT_BITS["note"]


def size(card, note):
    if card is None or note is None:
        return "–"
    b = (CARDS * card + NOTES * note) / 8
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference", default="fork/w8+s8")
    parser.add_argument("--student", default="st/w8+s8")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    per, meta = load(args.jsonl)
    labels = args.labels.split(",")
    users = sorted(set.intersection(*(set(per[label]) for label in labels + [args.reference])))
    lines = [f"{len(users)} users; review-weighted benchmark LogLoss; Δ% vs `{args.reference}` [95% user-bootstrap CI]; "
             f"Δ vs `{args.student}` (absolute); state = card+note state of the largest user "
             f"({CARDS:,} cards, {NOTES:,} notes).", "",
             "| Label | Card bits | Note bits | State (largest user) | Curve LogLoss | Δ% vs fork | Δ vs st w8+s8 | "
             "Instant LogLoss | Δ% vs fork | Δ vs st w8+s8 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for label in [args.reference] + labels:
        card, note = bits_of(meta[label])
        cells = [label, "–" if card is None else f"{card:,}", "–" if note is None else f"{note:,}", size(card, note)]
        row = {"label": label, "card_bits": card, "note_bits": note}
        for head in ("ahead", "instant"):
            cand, sizes = arrays(per, label, users, head, "LogLoss")
            ref, _ = arrays(per, args.reference, users, head, "LogLoss")
            base, _ = arrays(per, args.student, users, head, "LogLoss") if args.student in per else (None, None)
            value = rw(cand, sizes)
            pct = (value / rw(ref, sizes) - 1) * 100
            lo, hi = boot_delta_pct(cand, ref, sizes)
            delta = value - rw(base, sizes) if base is not None else float("nan")
            cells += [f"{value:.5f}", f"{pct:+.2f}% [{lo:+.2f}, {hi:+.2f}]", f"{delta:+.5f}"]
            row[head] = {"LogLoss": value, "delta_pct_vs_reference": pct, "ci95": [lo, hi], "delta_vs_student": delta}
        rows.append(row)
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        out = ROOT / args.out
        out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps({"users": users, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
