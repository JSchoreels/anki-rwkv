#!/usr/bin/env python
"""Aggressive state tiers for Tiny (SM146): state bytes of every scored label for the six-layer Tiny models (CPU, seconds).

Method of the int3 young tier on the early switch (SM145) (sm145_bytes.py): the fork's compact ModuleState layout; Tiny card
and note = 1 layer + 1 channel-mixer layer, H = 4, K = 16, D = 64; a matrix row = float16 scale + packed codes at the tier
width (int8 1 byte per code, int4 2 per byte, int3 8 codes in 3 bytes, int2 4 per byte), shift vectors int8 in every tier.
Bytes per entity: 1,315 / 803 / 675 / 547 at int8 / int4 / int3 / int2. Weighted by the end-of-history tier shares of the 723
users of the 732 set (sm146-dist.json = sm145-dist.json: answered cards 6,136,137, notes with answers 5,461,876).

Checks (exit 1 on failure): the formula reproduces SM136's Tiny int8 / int4 sizes (as SM145); the order of the labels by bytes
is the order the report assumes; consecutive labels differ by at least 1.5% of the int8 bytes (the pre-registered spread rule).

  python sm146_bytes.py [--out sm146-bytes.json]
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SM136 = HERE.parent / "sm136" / "sm136-bytes.json"
H, K, D, LAYERS, CM = 4, 16, 64, 1, 1          # Tiny (w64-c1): card and note alike
# label: (young bits, M, mid bits, N, old bits): young for n <= M, old for n >= N
SCHEMES = {
    "REF": (8, 0, 8, None, 8),
    "AN16": (4, 0, 4, 16, 8),
    "I3a": (3, 4, 4, 16, 8),
    "I3b": (3, 8, 4, 16, 8),
    "I2a": (2, 4, 4, 16, 8),
    "I2b": (2, 4, 3, 16, 8),
}
# report only (candidates that were not scored): int3 below 16 answers, int2 for 1-8
EXTRA = {"T3_16 (t3.16.8)": (3, 0, 3, 16, 8), "I2c (t2.9.3.16.8)": (2, 8, 3, 16, 8), "I2d (t2.9.4.16.8)": (2, 8, 4, 16, 8)}
ORDER = ["I2b", "I2a", "I3b", "I3a", "AN16"]   # ascending bytes, the order sm146_report.py assumes
MIN_STEP = 1.5                                 # % of the int8 bytes between consecutive labels (design.md)


def entity_bytes(bits):
    return 14 + 1 + 4 + LAYERS * 12 + (LAYERS + CM) * (2 + D) + LAYERS * H * K * (2 + math.ceil(K * bits / 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "sm146-bytes.json"))
    args = ap.parse_args()
    arm = json.loads(SM136.read_text())["arms"]["Tiny (w64-c1)"]
    want = {"card8": arm["w8+s8hx8"]["card"]["bytes"], "note8": arm["w8+s8hx8"]["note"]["bytes"],
            "card4": arm["w8+s4c8nhx8"]["card"]["bytes"]}
    got = {"card8": entity_bytes(8), "note8": entity_bytes(8), "card4": entity_bytes(4)}
    ok = want == got
    dist = json.loads((HERE / "sm146-dist.json").read_text())
    cards, notes = dist["answered_cards"], dist["notes_with_answers"]
    rows = {(t["M"] if t["young_bits"] else 0, t["N"]): t for t in dist["tiers"]}
    ref_total = (cards + notes) * entity_bytes(8)
    out = {"layout": "fork compact ModuleState; Tiny card = note = 1 layer + 1 channel-mixer layer; int8 shift vectors in every tier",
           "bytes_per_entity": {str(b): entity_bytes(b) for b in (8, 4, 3, 2)}, "check_vs_sm136": {"sm136": want, "formula": got, "equal": want == got},
           "entities_732": {"answered_cards": cards, "notes_with_answers": notes}, "labels": {}, "report_only": {}}
    for table, schemes in ((out["labels"], SCHEMES), (out["report_only"], EXTRA)):
        for name, (young, M, mid, N, old) in schemes.items():
            if N is None:
                mean = {"card": entity_bytes(mid), "note": entity_bytes(mid)}
            else:
                mean = {}
                for kind in ("card", "note"):
                    e = rows[(M, N)][kind]["entities"]
                    mean[kind] = e[0] * entity_bytes(young) + e[1] * entity_bytes(mid) + e[2] * entity_bytes(old)
            total = cards * mean["card"] + notes * mean["note"]
            table[name] = {"spec": [young, M, mid, N, old], "mean_card_bytes": round(mean["card"], 1),
                           "mean_note_bytes": round(mean["note"], 1), "total_mb_732": round(total / 1e6, 1),
                           "vs_REF_pct": round(100 * (total / ref_total - 1), 2)}
    pct = {k: v["vs_REF_pct"] for k, v in out["labels"].items()}
    ok &= sorted(ORDER, key=lambda s: pct[s]) == ORDER
    steps = {ORDER[i]: round(pct[ORDER[i + 1]] - pct[ORDER[i]], 2) for i in range(len(ORDER) - 1)}   # extra saving vs the next less aggressive label
    out["extra_saving_vs_next_less_aggressive_label_pct_of_int8"] = steps
    out["step_rule_ok"] = all(s >= MIN_STEP for s in steps.values())
    ok &= out["step_rule_ok"]
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[sm146-bytes] {'OK' if ok else 'FAILED'}: formula vs SM136 {want == got}; vs int8: "
          + ", ".join(f"{k} {v:+.2f}%" for k, v in pct.items()) + f"; extra saving vs the next less aggressive label {steps}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
